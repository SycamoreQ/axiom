use crate::metal::context::MetalContext;
use crate::metal::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLDevice;
use objc2_metal::MTLResourceOptions;
use std::cell::{Cell, RefCell};

#[derive(Clone, Debug)]
pub struct BlockHandle {
    pub index: usize,
    pub ptr: *mut u8,
    pub offset_bytes: usize,
    pub size_bytes: usize,
    pub owned_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

unsafe impl Send for BlockHandle {}
unsafe impl Sync for BlockHandle {}

impl BlockHandle {
    pub fn metal_buffer<'a>(
        &'a self,
        allocator: &'a MetalAllocator,
    ) -> &'a ProtocolObject<dyn MTLBuffer> {
        self.owned_buffer
            .as_deref()
            .unwrap_or_else(|| allocator.buffer())
    }
}

pub struct FreeBlock {
    pub offset_bytes: usize,
    pub size: usize,
}

pub struct MetalAllocator {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    capacity: usize,
    current_offset: Cell<usize>,
    free_list: RefCell<Vec<FreeBlock>>,
}

impl std::fmt::Debug for MetalAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalAllocator").finish()
    }
}

unsafe impl Send for MetalAllocator {}
unsafe impl Sync for MetalAllocator {}

impl MetalAllocator {
    pub fn new(ctx: &MetalContext, total_size_bytes: usize) -> Result<Self> {
        let options = MTLResourceOptions::StorageModeShared;
        let buffer = ctx
            .device
            .raw()
            .newBufferWithLength_options(total_size_bytes, options)
            .ok_or(MetalError::AllocationFailed)?;

        assert!(
            !buffer.contents().as_ptr().is_null(),
            "MTLBuffer contents() is null — allocation failed"
        );

        Ok(Self {
            buffer,
            capacity: total_size_bytes,
            current_offset: Cell::new(0),
            free_list: RefCell::new(Vec::new()),
        })
    }

    pub fn alloc(&self, size: usize, alignment: usize) -> Result<BlockHandle> {
        // Split the lookup from the removal on purpose: `self.free_list.borrow()`
        // inside an `if let` scrutinee has its temporary Ref kept alive for the
        // whole if-let body (same rule behind the classic
        // `match mutex.lock().unwrap() { .. }` footgun), so calling
        // `.borrow_mut()` inside that body panics with "already borrowed:
        // BorrowMutError" the moment this branch is taken. Binding `found`
        // first drops the Ref before the body runs.
        let found = self.free_list.borrow().iter().position(|b| {
            let remainder = b.offset_bytes % alignment;
            let padding = if remainder == 0 {
                0
            } else {
                alignment - remainder
            };
            b.size >= size + padding
        });
        if let Some(pos) = found {
            let block = self.free_list.borrow_mut().remove(pos);
            let remainder = block.offset_bytes % alignment;
            let padding = if remainder == 0 {
                0
            } else {
                alignment - remainder
            };
            let start_offset = block.offset_bytes + padding;
            let base_ptr = self.buffer.contents().as_ptr() as *mut u8;
            let ptr = unsafe { base_ptr.add(start_offset) };
            return Ok(BlockHandle {
                index: pos,
                offset_bytes: start_offset,
                size_bytes: size,
                ptr,
                owned_buffer: None,
            });
        }

        let current = self.current_offset.get();
        let remainder = current % alignment;
        let padding = if remainder == 0 {
            0
        } else {
            alignment - remainder
        };
        let start_offset = current + padding;

        if start_offset + size > self.capacity {
            return Err(MetalError::OutOfMemory {
                requested: size,
                available: self.capacity - current,
            });
        }

        self.current_offset.set(start_offset + size);

        let base_ptr = self.buffer.contents().as_ptr() as *mut u8;
        let block_ptr = unsafe { base_ptr.add(start_offset) };

        Ok(BlockHandle {
            index: 0,
            offset_bytes: start_offset,
            size_bytes: size,
            ptr: block_ptr,
            owned_buffer: None,
        })
    }

    pub fn free(&self, handle: BlockHandle) {
        let mut free_list = self.free_list.borrow_mut();
        free_list.push(FreeBlock {
            offset_bytes: handle.offset_bytes,
            size: handle.size_bytes,
        });
        free_list.sort_by_key(|b| b.offset_bytes);

        let mut merged: Vec<FreeBlock> = Vec::new();
        for block in free_list.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.offset_bytes + last.size == block.offset_bytes {
                    last.size += block.size;
                    continue;
                }
            }
            merged.push(block);
        }
        *free_list = merged;

        if let Some(last) = free_list.last() {
            if last.offset_bytes + last.size == self.current_offset.get() {
                self.current_offset.set(last.offset_bytes);
                free_list.pop();
            }
        }
    }

    /// Reclaims the entire pool at once: bump cursor back to 0, free list
    /// cleared. Safe whenever nothing alive still needs pool-backed memory --
    /// e.g. at the start of a forward_metal call, since the KV cache (the only
    /// thing that must survive across calls) lives in its own dedicated
    /// MTLBuffer via zeros()/from_bytes_direct, never in this pool.
    pub fn reset(&self) {
        self.current_offset.set(0);
        self.free_list.borrow_mut().clear();
    }

    pub fn free_count(&self) -> usize {
        self.free_list.borrow().len()
    }

    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::context::MetalContext;
    use crate::metal::device::MetalDevice;

    fn make_allocator(size: usize) -> MetalAllocator {
        let device = MetalDevice::system_default().unwrap();
        let ctx = MetalContext::new(device).unwrap();
        MetalAllocator::new(&ctx, size).unwrap()
    }

    #[test]
    fn test_basic_alloc() {
        let alloc = make_allocator(1024);
        let h = alloc.alloc(64, 16).unwrap();
        assert_eq!(h.offset_bytes, 0);
        assert_eq!(h.size_bytes, 64);
        assert!(!h.ptr.is_null());
    }

    #[test]
    fn test_alignment_padding() {
        let alloc = make_allocator(1024);
        let _h1 = alloc.alloc(3, 1).unwrap();
        let h2 = alloc.alloc(16, 16).unwrap();
        assert_eq!(h2.offset_bytes % 16, 0);
    }

    #[test]
    fn test_free_reuse() {
        let alloc = make_allocator(1024);
        let h1 = alloc.alloc(64, 16).unwrap();
        let offset = h1.offset_bytes;
        alloc.free(h1);
        let h2 = alloc.alloc(64, 16).unwrap();
        assert_eq!(h2.offset_bytes, offset);
    }

    #[test]
    fn test_cursor_reclaim() {
        let alloc = make_allocator(1024);
        let h = alloc.alloc(64, 16).unwrap();
        let cursor_before = alloc.current_offset.get();
        alloc.free(h);
        assert!(alloc.current_offset.get() < cursor_before);
        assert_eq!(alloc.free_list.borrow().len(), 0);
    }

    #[test]
    fn test_exhaustion() {
        let alloc = make_allocator(128);
        let _h = alloc.alloc(128, 1).unwrap();
        let err = alloc.alloc(1, 1).unwrap_err();
        assert!(matches!(err, MetalError::OutOfMemory { .. }));
    }

    #[test]
    fn test_cpu_write_read() {
        let alloc = make_allocator(1024);
        let h = alloc.alloc(4, 4).unwrap();
        unsafe {
            let p = h.ptr as *mut f32;
            p.write(42.0f32);
            assert_eq!(p.read(), 42.0f32);
        }
    }

    #[test]
    fn test_reset_reclaims_whole_pool() {
        let alloc = make_allocator(1024);
        let _h1 = alloc.alloc(200, 16).unwrap();
        let _h2 = alloc.alloc(200, 16).unwrap();
        alloc.reset();
        assert_eq!(alloc.current_offset.get(), 0);
        assert_eq!(alloc.free_list.borrow().len(), 0);
        let h3 = alloc.alloc(1000, 16).unwrap();
        assert_eq!(h3.offset_bytes, 0);
    }
}
