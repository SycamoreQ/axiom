pub mod engine_profiler;
pub mod metrics;
pub mod report;
pub mod system;

pub use engine_profiler::EngineProfiler;
pub use metrics::{InferenceProfile, LatencyStats, MemoryStats, TokenTiming};
pub use report::ProfilerReport;
pub use system::{ResourceDiff, ResourceSnapshot, SystemMonitor};

#[cfg(feature = "profile")]
pub fn init_chrome_tracing(path: &str) -> tracing_chrome::FlushGuard {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .file(path)
        .include_args(true)
        .build();

    let _ = tracing_subscriber::registry().with(chrome_layer).try_init();
    guard
}

