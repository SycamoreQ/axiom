use std::time::Instant;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceSnapshot {
    pub rss_bytes: usize,
    pub peak_rss_bytes: usize,
    pub user_cpu_time_us: u64,
    pub system_cpu_time_us: u64,
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceDiff {
    pub wall_time_ms: f64,
    pub delta_rss_mb: f64,
    pub peak_rss_mb: f64,
    pub current_rss_mb: f64,
    pub user_cpu_ms: f64,
    pub system_cpu_ms: f64,
    pub total_cpu_ms: f64,
    pub cpu_utilization_pct: f64,
}

pub struct SystemMonitor;

impl SystemMonitor {
    pub fn snapshot() -> ResourceSnapshot {
        let (user_cpu_us, sys_cpu_us, ru_maxrss) = Self::get_rusage();
        let current_rss = Self::get_current_rss().unwrap_or(ru_maxrss);

        ResourceSnapshot {
            rss_bytes: current_rss,
            peak_rss_bytes: ru_maxrss,
            user_cpu_time_us: user_cpu_us,
            system_cpu_time_us: sys_cpu_us,
            timestamp: Instant::now(),
        }
    }

    pub fn diff(start: &ResourceSnapshot, end: &ResourceSnapshot) -> ResourceDiff {
        let wall_time_sec = end.timestamp.duration_since(start.timestamp).as_secs_f64();
        let wall_time_ms = wall_time_sec * 1000.0;

        let delta_rss_bytes = (end.rss_bytes as i64) - (start.rss_bytes as i64);
        let delta_rss_mb = delta_rss_bytes as f64 / (1024.0 * 1024.0);
        let peak_rss_mb = end.peak_rss_bytes as f64 / (1024.0 * 1024.0);
        let current_rss_mb = end.rss_bytes as f64 / (1024.0 * 1024.0);

        let user_cpu_ms =
            (end.user_cpu_time_us.saturating_sub(start.user_cpu_time_us)) as f64 / 1000.0;
        let system_cpu_ms = (end
            .system_cpu_time_us
            .saturating_sub(start.system_cpu_time_us)) as f64
            / 1000.0;
        let total_cpu_ms = user_cpu_ms + system_cpu_ms;

        let cpu_utilization_pct = if wall_time_ms > 0.0 {
            (total_cpu_ms / wall_time_ms) * 100.0
        } else {
            0.0
        };

        ResourceDiff {
            wall_time_ms,
            delta_rss_mb,
            peak_rss_mb,
            current_rss_mb,
            user_cpu_ms,
            system_cpu_ms,
            total_cpu_ms,
            cpu_utilization_pct,
        }
    }

    fn get_rusage() -> (u64, u64, usize) {
        unsafe {
            let mut usage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
                let user_us =
                    (usage.ru_utime.tv_sec as u64) * 1_000_000 + (usage.ru_utime.tv_usec as u64);
                let sys_us =
                    (usage.ru_stime.tv_sec as u64) * 1_000_000 + (usage.ru_stime.tv_usec as u64);

                // On macOS, ru_maxrss is in bytes; on Linux, it's in kilobytes.
                #[cfg(target_os = "macos")]
                let max_rss = usage.ru_maxrss as usize;
                #[cfg(not(target_os = "macos"))]
                let max_rss = (usage.ru_maxrss as usize) * 1024;

                (user_us, sys_us, max_rss)
            } else {
                (0, 0, 0)
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn get_current_rss() -> Option<usize> {
        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: libc::time_value_t,
            system_time: libc::time_value_t,
            policy: i32,
            suspend_count: i32,
        }

        const MACH_TASK_BASIC_INFO: libc::c_uint = 20;
        const MACH_TASK_BASIC_INFO_COUNT: libc::mach_msg_type_number_t =
            (std::mem::size_of::<MachTaskBasicInfo>() / std::mem::size_of::<libc::natural_t>())
                as libc::mach_msg_type_number_t;

        extern "C" {
            fn mach_task_self() -> libc::mach_port_t;
            fn task_info(
                target_task: libc::mach_port_t,
                flavor: libc::c_uint,
                task_info_out: *mut libc::c_void,
                task_info_outCnt: *mut libc::mach_msg_type_number_t,
            ) -> libc::kern_return_t;
        }

        unsafe {
            let mut info: MachTaskBasicInfo = std::mem::zeroed();
            let mut count = MACH_TASK_BASIC_INFO_COUNT;
            let kr = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut libc::c_void,
                &mut count,
            );
            if kr == 0 {
                Some(info.resident_size as usize)
            } else {
                None
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn get_current_rss() -> Option<usize> {
        // Read /proc/self/statm on Linux (second number is resident pages)
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        Some(resident_pages * page_size)
    }
}
