use axiom::core::backend::{Backend, MetalBackend};
use axiom::core::device::Device;
use axiom::core::dtype::DType;
use axiom::core::shape::Shape;
use axiom::core::tensor::TensorOps;

fn main() {
    let pool_size = 512usize * 1024 * 1024;
    axiom::metal::state::init_global_metal_state(pool_size)
        .expect("failed to initialize Metal state");

    let device = Device::Metal(0);

    type T = <MetalBackend as Backend>::Tensor;
    let a = T::zeros(&Shape::new(&[1, 2048]), DType::F32, &device).expect("failed to alloc a");
    let w = T::zeros(&Shape::new(&[2048, 768]), DType::F32, &device).expect("failed to alloc w");

    // warm up — first call often eats one-time driver/pipeline setup cost,
    // which would otherwise skew the average
    let _ = a.broadcast_matmul(&w).expect("warmup matmul failed");

    let n = 200;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        let _ = a.broadcast_matmul(&w).expect("matmul failed");
    }
    let elapsed = t0.elapsed();
    println!("total: {:?}, avg per matmul: {:?}", elapsed, elapsed / n);
}
