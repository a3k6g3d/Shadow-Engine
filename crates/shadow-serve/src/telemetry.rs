use crate::engine::Status;
use serde_json::{json, Value};
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(windows)]
mod win {
    #[repr(C)]
    pub struct MemoryStatusEx {
        pub length: u32,
        pub load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }

    #[repr(C)]
    pub struct ProcessMemoryCounters {
        pub cb: u32,
        pub page_fault_count: u32,
        pub peak_working_set_size: usize,
        pub working_set_size: usize,
        pub quota_peak_paged_pool_usage: usize,
        pub quota_paged_pool_usage: usize,
        pub quota_peak_non_paged_pool_usage: usize,
        pub quota_non_paged_pool_usage: usize,
        pub pagefile_usage: usize,
        pub peak_pagefile_usage: usize,
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct FileTime {
        pub low: u32,
        pub high: u32,
    }

    impl FileTime {
        pub fn as_u64(&self) -> u64 {
            ((self.high as u64) << 32) | (self.low as u64)
        }
    }

    extern "system" {
        pub fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
        pub fn GetCurrentProcess() -> isize;
        pub fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
        pub fn GetSystemTimes(idle: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
        pub fn GetProcessTimes(process: isize, creation: *mut FileTime, exit: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
    }
}

pub struct SysMonitor {
    last_sample: Instant,
    last_sys_idle: u64,
    last_sys_kernel: u64,
    last_sys_user: u64,
    last_proc_kernel: u64,
    last_proc_user: u64,
    proc_cpu_pct: f64,
    sys_cpu_pct: f64,
}

impl SysMonitor {
    pub fn new() -> Self {
        let mut m = Self {
            last_sample: Instant::now(),
            last_sys_idle: 0,
            last_sys_kernel: 0,
            last_sys_user: 0,
            last_proc_kernel: 0,
            last_proc_user: 0,
            proc_cpu_pct: 0.0,
            sys_cpu_pct: 0.0,
        };
        m.sample();
        m
    }

    pub fn sample(&mut self) {
        #[cfg(windows)]
        {
            let mut idle = win::FileTime::default();
            let mut sys_k = win::FileTime::default();
            let mut sys_u = win::FileTime::default();
            let mut proc_c = win::FileTime::default();
            let mut proc_e = win::FileTime::default();
            let mut proc_k = win::FileTime::default();
            let mut proc_u = win::FileTime::default();

            unsafe {
                win::GetSystemTimes(&mut idle, &mut sys_k, &mut sys_u);
                let proc = win::GetCurrentProcess();
                win::GetProcessTimes(proc, &mut proc_c, &mut proc_e, &mut proc_k, &mut proc_u);
            }

            let cur_sys_idle = idle.as_u64();
            let cur_sys_kernel = sys_k.as_u64();
            let cur_sys_user = sys_u.as_u64();
            let cur_proc_kernel = proc_k.as_u64();
            let cur_proc_user = proc_u.as_u64();

            let sys_total = (cur_sys_kernel.saturating_sub(self.last_sys_kernel))
                + (cur_sys_user.saturating_sub(self.last_sys_user));
            let sys_idle = cur_sys_idle.saturating_sub(self.last_sys_idle);
            let proc_total = (cur_proc_kernel.saturating_sub(self.last_proc_kernel))
                + (cur_proc_user.saturating_sub(self.last_proc_user));

            if sys_total > 0 && self.last_sys_kernel > 0 {
                let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f64;
                self.sys_cpu_pct = (((sys_total.saturating_sub(sys_idle)) as f64 / sys_total as f64) * 100.0).clamp(0.0, 100.0);
                self.proc_cpu_pct = ((proc_total as f64 / sys_total as f64) * 100.0 * cores).clamp(0.0, 100.0);
            }

            self.last_sys_idle = cur_sys_idle;
            self.last_sys_kernel = cur_sys_kernel;
            self.last_sys_user = cur_sys_user;
            self.last_proc_kernel = cur_proc_kernel;
            self.last_proc_user = cur_proc_user;
            self.last_sample = Instant::now();
        }
    }
}

pub struct TelemetryService {
    monitor: Mutex<SysMonitor>,
}

impl TelemetryService {
    pub fn new() -> Self {
        Self {
            monitor: Mutex::new(SysMonitor::new()),
        }
    }

    pub fn snapshot(&self, s: &Status, uptime_s: f64, queued: u64) -> Value {
        let (proc_cpu, sys_cpu) = {
            let mut m = self.monitor.lock().unwrap();
            if m.last_sample.elapsed() >= Duration::from_millis(400) {
                m.sample();
            }
            (m.proc_cpu_pct, m.sys_cpu_pct)
        };

        let mut total_ram = 0u64;
        let mut avail_ram = 0u64;
        let mut mem_load = 0u32;
        let mut working_set = 0usize;
        let mut commit_size = 0usize;

        #[cfg(windows)]
        unsafe {
            let mut mem_stat = win::MemoryStatusEx {
                length: std::mem::size_of::<win::MemoryStatusEx>() as u32,
                load: 0,
                total_phys: 0,
                avail_phys: 0,
                total_page_file: 0,
                avail_page_file: 0,
                total_virtual: 0,
                avail_virtual: 0,
                avail_extended_virtual: 0,
            };
            if win::GlobalMemoryStatusEx(&mut mem_stat) != 0 {
                total_ram = mem_stat.total_phys;
                avail_ram = mem_stat.avail_phys;
                mem_load = mem_stat.load;
            }

            let mut proc_mem = std::mem::zeroed::<win::ProcessMemoryCounters>();
            proc_mem.cb = std::mem::size_of::<win::ProcessMemoryCounters>() as u32;
            let proc = win::GetCurrentProcess();
            if win::K32GetProcessMemoryInfo(proc, &mut proc_mem, proc_mem.cb) != 0 {
                working_set = proc_mem.working_set_size;
                commit_size = proc_mem.pagefile_usage;
            }
        }

        let used_ram = total_ram.saturating_sub(avail_ram);
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let vram_used = s.vram_total.saturating_sub(s.vram_free);
        let hit_rate = if s.vram_hits + s.vram_misses > 0 {
            (s.vram_hits as f64 / (s.vram_hits + s.vram_misses) as f64) * 100.0
        } else {
            0.0
        };

        let tok_s = if s.decode_secs > 0.0 {
            s.completion_tokens as f64 / s.decode_secs
        } else {
            0.0
        };

        json!({
            "cpu": {
                "process_percent": (proc_cpu * 10.0).round() / 10.0,
                "system_percent": (sys_cpu * 10.0).round() / 10.0,
                "logical_cores": cores
            },
            "ram": {
                "system_total_bytes": total_ram,
                "system_available_bytes": avail_ram,
                "system_used_bytes": used_ram,
                "system_load_percent": mem_load,
                "process_working_set_bytes": working_set,
                "process_commit_bytes": commit_size,
                "pinned_tier_bytes": s.ram_tier_bytes,
                "pinned_tier_experts": s.ram_tier_experts,
                "ram_served_hits": s.ram_served,
                "file_served_misses": s.file_served
            },
            "gpu": {
                "name": s.gpu_name,
                "vram_total_bytes": s.vram_total,
                "vram_free_bytes": s.vram_free,
                "vram_used_bytes": vram_used,
                "expert_cache_slots": s.vram_slots,
                "expert_cache_hits": s.vram_hits,
                "expert_cache_misses": s.vram_misses,
                "expert_cache_hit_rate": (hit_rate * 10.0).round() / 10.0
            },
            "process": {
                "pid": std::process::id(),
                "uptime_s": (uptime_s * 10.0).round() / 10.0,
                "model": s.model,
                "state": s.state,
                "busy": s.busy,
                "n_ctx": s.n_ctx,
                "queued": queued,
                "tok_per_sec": (tok_s * 100.0).round() / 100.0,
                "prompt_tokens": s.prompt_tokens,
                "cached_tokens": s.cached_tokens,
                "completion_tokens": s.completion_tokens,
                "requests": s.requests
            }
        })
    }
}
