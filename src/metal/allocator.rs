use crate::metal::context::MetalContext;
use crate::metal::error::{MetalError, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLDevice;
use objc2_metal::MTLResourceOptions;

#[derive(Debug)]
pub struct BlockHandle {
    pub index: usize,
    pub ptr: *mut u8,
    pub offset_bytes: usize,
    pub size_bytes: usize,
}

unsafe impl Send for BlockHandle {}

pub struct FreeBlock {
    pub offset_bytes: usize,
    pub size: usize,
}

pub struct MetalAllocator {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    capacity: usize,
    current_offset: usize,
    free_list: Vec<FreeBlock>,
}

impl MetalAllocator {
    pub fn new(ctx: &MetalContext, total_size_bytes: usize) -> Result<Self> {
        let options = MTLResourceOptions::StorageModeShared; // Unified memory for Apple Silicon
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
            current_offset: 0,
            free_list: Vec::new(),
        })
    }

    pub fn alloc(&mut self, size: usize, alignment: usize) -> Result<BlockHandle> {
        if let Some(pos) = self.free_list.iter().position(|b| {
            let remainder = b.offset_bytes % alignment;
            let padding = if remainder == 0 {
                0
            } else {
                alignment - remainder
            };
            b.size >= size + padding
        }) {
            let block = self.free_list.remove(pos);
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
            });
        }

        let remainder = self.current_offset % alignment;
        let padding = if remainder == 0 {
            0
        } else {
            alignment - remainder
        };
        let start_offset = self.current_offset + padding;

        if start_offset + size > self.capacity {
            return Err(MetalError::OutOfMemory {
                requested: size,
                available: self.capacity - self.current_offset,
            });
        }

        self.current_offset = start_offset + size;

        let base_ptr = self.buffer.contents().as_ptr() as *mut u8;
        let block_ptr = unsafe { base_ptr.add(start_offset) };

        Ok(BlockHandle {
            index: 0,
            offset_bytes: start_offset,
            size_bytes: size,
            ptr: block_ptr,
        })
    }

    pub fn free(&mut self, handle: BlockHandle) {
        self.free_list.push(FreeBlock {
            offset_bytes: handle.offset_bytes,
            size: handle.size_bytes,
        });
        self.free_list.sort_by_key(|b| b.offset_bytes);

        // merge contiguous blocks
        let mut merged: Vec<FreeBlock> = Vec::new();
        for block in self.free_list.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.offset_bytes + last.size == block.offset_bytes {
                    // contiguous — extend the last block instead of pushing a new one
                    last.size += block.size;
                    continue;
                }
            }
            merged.push(block);
        }
        self.free_list = merged;

        // if the last free block touches the bump cursor, reclaim it
        if let Some(last) = self.free_list.last() {
            if last.offset_bytes + last.size == self.current_offset {
                self.current_offset = last.offset_bytes;
                self.free_list.pop();
            }
        }
    }
    pub fn free_count(&self) -> usize {
        self.free_list.len()
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
        let mut alloc = make_allocator(1024);
        let h = alloc.alloc(64, 16).unwrap();
        assert_eq!(h.offset_bytes, 0);
        assert_eq!(h.size_bytes, 64);
        assert!(!h.ptr.is_null());
    }

    #[test]
    fn test_alignment_padding() {
        let mut alloc = make_allocator(1024);
        let _h1 = alloc.alloc(3, 1).unwrap(); // bump cursor to 3
        let h2 = alloc.alloc(16, 16).unwrap(); // should pad to offset 16
        assert_eq!(h2.offset_bytes % 16, 0);
    }

    #[test]
    fn test_free_reuse() {
        let mut alloc = make_allocator(1024);
        let h1 = alloc.alloc(64, 16).unwrap();
        let offset = h1.offset_bytes;
        alloc.free(h1);
        let h2 = alloc.alloc(64, 16).unwrap();
        assert_eq!(h2.offset_bytes, offset); // same block reused
    }

    #[test]
    fn test_cursor_reclaim() {
        let mut alloc = make_allocator(1024);
        let h = alloc.alloc(64, 16).unwrap();
        let cursor_before = alloc.current_offset;
        alloc.free(h);
        // cursor should walk back since the freed block was at the top
        assert!(alloc.current_offset < cursor_before);
        assert_eq!(alloc.free_list.len(), 0);
    }

    #[test]
    fn test_exhaustion() {
        let mut alloc = make_allocator(128);
        let _h = alloc.alloc(128, 1).unwrap();
        let err = alloc.alloc(1, 1).unwrap_err();
        assert!(matches!(err, MetalError::OutOfMemory { .. }));
    }

    #[test]
    fn test_cpu_write_read() {
        let mut alloc = make_allocator(1024);
        let h = alloc.alloc(4, 4).unwrap();
        // write via CPU pointer, read back — validates StorageModeShared is working
        unsafe {
            let p = h.ptr as *mut f32;
            p.write(42.0f32);
            assert_eq!(p.read(), 42.0f32);
        }
    }
}
