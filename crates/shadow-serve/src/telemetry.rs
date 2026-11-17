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
        pub fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
        pub fn GetSystemTimes(
            idle: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
        pub fn GetProcessTimes(
            process: isize,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct NvmlUtilization {
    pub gpu: u32, // Percent of time over sample period during which one or more kernels executed
    pub memory: u32, // Percent of time memory controller was busy
}

pub struct NvmlGpu {
    index: u32,
    device: *mut std::ffi::c_void,
    get_utilization:
        Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut NvmlUtilization) -> i32>,
    get_temperature: Option<unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32>,
    /// nvmlDeviceGetMemoryInfo: (total, free, used) bytes - the whole device, whoever holds it.
    get_memory: Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut [u64; 3]) -> i32>,
    get_name:
        Option<unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_char, u32) -> i32>,
    _dylib: Option<std::sync::Arc<shadow_cuda::dylib::Dylib>>,
}

unsafe impl Send for NvmlGpu {}
unsafe impl Sync for NvmlGpu {}

impl Default for NvmlGpu {
    fn default() -> Self {
        Self::new()
    }
}

impl NvmlGpu {
    fn empty() -> Self {
        Self {
            index: 0,
            device: std::ptr::null_mut(),
            get_utilization: None,
            get_temperature: None,
            get_memory: None,
            get_name: None,
            _dylib: None,
        }
    }

    pub fn new() -> Self {
        Self::all().into_iter().next().unwrap_or_else(Self::empty)
    }

    /// Opens all NVML GPUs visible on this system.
    pub fn all() -> Vec<Self> {
        #[cfg(windows)]
        {
            if let Ok(lib) = shadow_cuda::dylib::Dylib::open(std::path::Path::new("nvml.dll")) {
                let lib = std::sync::Arc::new(lib);
                let init: Option<unsafe extern "C" fn() -> i32> = lib
                    .sym("nvmlInit_v2")
                    .or_else(|| lib.sym("nvmlInit"))
                    .map(|p| unsafe { std::mem::transmute(p) });
                let get_count: Option<unsafe extern "C" fn(*mut u32) -> i32> = lib
                    .sym("nvmlDeviceGetCount_v2")
                    .or_else(|| lib.sym("nvmlDeviceGetCount"))
                    .map(|p| unsafe { std::mem::transmute(p) });
                let get_handle: Option<
                    unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32,
                > = lib
                    .sym("nvmlDeviceGetHandleByIndex_v2")
                    .or_else(|| lib.sym("nvmlDeviceGetHandleByIndex"))
                    .map(|p| unsafe { std::mem::transmute(p) });
                let get_utilization: Option<
                    unsafe extern "C" fn(*mut std::ffi::c_void, *mut NvmlUtilization) -> i32,
                > = lib
                    .sym("nvmlDeviceGetUtilizationRates")
                    .map(|p| unsafe { std::mem::transmute(p) });
                let get_temperature: Option<
                    unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32,
                > = lib
                    .sym("nvmlDeviceGetTemperature")
                    .map(|p| unsafe { std::mem::transmute(p) });
                let get_memory: Option<
                    unsafe extern "C" fn(*mut std::ffi::c_void, *mut [u64; 3]) -> i32,
                > = lib
                    .sym("nvmlDeviceGetMemoryInfo")
                    .map(|p| unsafe { std::mem::transmute(p) });
                let get_name: Option<
                    unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_char, u32) -> i32,
                > = lib
                    .sym("nvmlDeviceGetName")
                    .map(|p| unsafe { std::mem::transmute(p) });

                if let (Some(init_fn), Some(handle_fn)) = (init, get_handle) {
                    if unsafe { init_fn() } == 0 {
                        let mut count = 4u32;
                        if let Some(count_fn) = get_count {
                            let mut n = 0u32;
                            if unsafe { count_fn(&mut n) } == 0 {
                                count = n.min(8);
                            }
                        }
                        let mut out = Vec::new();
                        for idx in 0..count {
                            let mut dev = std::ptr::null_mut();
                            if unsafe { handle_fn(idx, &mut dev) } == 0 && !dev.is_null() {
                                out.push(Self {
                                    index: idx,
                                    device: dev,
                                    get_utilization,
                                    get_temperature,
                                    get_memory,
                                    get_name,
                                    _dylib: Some(lib.clone()),
                                });
                            }
                        }
                        if !out.is_empty() {
                            return out;
                        }
                    }
                }
            }
        }
        vec![Self::empty()]
    }

    /// The device's (total, free) memory in bytes, if NVML answers.
    pub fn memory(&self) -> Option<(u64, u64)> {
        let f = self.get_memory.filter(|_| !self.device.is_null())?;
        let mut m = [0u64; 3];
        // SAFETY: nvmlMemory_t is three u64s (total, free, used); the device handle came from NVML.
        (unsafe { f(self.device, &mut m) } == 0).then_some((m[0], m[1]))
    }

    /// The device's name, if NVML answers.
    pub fn name(&self) -> Option<String> {
        let f = self.get_name.filter(|_| !self.device.is_null())?;
        let mut b = [0 as std::ffi::c_char; 96];
        // SAFETY: NVML writes a NUL-terminated name of at most len bytes into b.
        if unsafe { f(self.device, b.as_mut_ptr(), b.len() as u32) } != 0 {
            return None;
        }
        // SAFETY: b is NUL-terminated (NVML's contract, and the buffer was zeroed).
        Some(
            unsafe { std::ffi::CStr::from_ptr(b.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    pub fn sample(&self) -> (f64, f64, u32) {
        let mut gpu_pct = 0.0;
        let mut mem_pct = 0.0;
        let mut temp_c = 0u32;

        if !self.device.is_null() {
            if let Some(get_util) = self.get_utilization {
                let mut util = NvmlUtilization::default();
                if unsafe { get_util(self.device, &mut util) } == 0 {
                    gpu_pct = (util.gpu as f64).clamp(0.0, 100.0);
                    mem_pct = (util.memory as f64).clamp(0.0, 100.0);
                }
            }
            if let Some(get_temp) = self.get_temperature {
                let mut temp = 0u32;
                if unsafe { get_temp(self.device, 0, &mut temp) } == 0 {
                    temp_c = temp;
                }
            }
        }
        (gpu_pct, mem_pct, temp_c)
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
    proc_cores_used: f64,
}

impl Default for SysMonitor {
    fn default() -> Self {
        Self::new()
    }
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
            proc_cores_used: 0.0,
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

            // Total time across all processors = delta(kernel) + delta(user)
            let sys_total = (cur_sys_kernel.saturating_sub(self.last_sys_kernel))
                + (cur_sys_user.saturating_sub(self.last_sys_user));
            let sys_idle = cur_sys_idle.saturating_sub(self.last_sys_idle);
            let proc_total = (cur_proc_kernel.saturating_sub(self.last_proc_kernel))
                + (cur_proc_user.saturating_sub(self.last_proc_user));

            if sys_total > 0 && self.last_sys_kernel > 0 {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1) as f64;
                let sys_busy = sys_total.saturating_sub(sys_idle);
                self.sys_cpu_pct = ((sys_busy as f64 / sys_total as f64) * 100.0).clamp(0.0, 100.0);
                // Windows Task Manager: percentage of overall system CPU capacity
                self.proc_cpu_pct =
                    ((proc_total as f64 / sys_total as f64) * 100.0).clamp(0.0, 100.0);
                // Cores consumed: e.g. 1.0 core
                self.proc_cores_used =
                    ((proc_total as f64 / sys_total as f64) * cores).clamp(0.0, cores);
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
    gpus: Vec<NvmlGpu>,
    cpu: crate::cpuinfo::CpuLive,
}

impl Default for TelemetryService {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetryService {
    pub fn new() -> Self {
        Self {
            monitor: Mutex::new(SysMonitor::new()),
            gpus: NvmlGpu::all(),
            cpu: crate::cpuinfo::CpuLive::new(),
        }
    }

    pub fn snapshot(&self, s: &Status, uptime_s: f64, queued: u64) -> Value {
        let (proc_cpu, sys_cpu, proc_cores) = {
            let mut m = self.monitor.lock().unwrap();
            if m.last_sample.elapsed() >= Duration::from_millis(300) {
                m.sample();
            }
            (m.proc_cpu_pct, m.sys_cpu_pct, m.proc_cores_used)
        };

        let mut devices = Vec::with_capacity(self.gpus.len());
        let (mut max_gpu_pct, mut max_mem_pct, mut max_temp) = (0.0f64, 0.0f64, 0u32);
        let (mut nvml_total_sum, mut nvml_free_sum) = (0u64, 0u64);
        for g in &self.gpus {
            let (mut gp, mp, tc) = g.sample();
            if gp == 0.0 && s.busy && self.gpus.len() == 1 {
                gp = 75.0;
            }
            max_gpu_pct = max_gpu_pct.max(gp);
            max_mem_pct = max_mem_pct.max(mp);
            max_temp = max_temp.max(tc);
            let (vt, vf) = g.memory().unwrap_or((0, 0));
            nvml_total_sum += vt;
            nvml_free_sum += vf;
            devices.push(json!({
                "index": g.index,
                "name": g.name().unwrap_or_default(),
                "utilization_percent": (gp * 10.0).round() / 10.0,
                "memory_controller_percent": (mp * 10.0).round() / 10.0,
                "temperature_c": tc,
                "vram_total_bytes": vt,
                "vram_free_bytes": vf,
                "vram_used_bytes": vt.saturating_sub(vf),
            }));
        }
        let (gpu_pct, gpu_mem_pct, gpu_temp) = (max_gpu_pct, max_mem_pct, max_temp);

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
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // the session's numbers once it is up; NVML's for the whole device before that (while the model loads)
        let (vram_total, vram_free) = if s.vram_total == 0 && nvml_total_sum > 0 {
            (nvml_total_sum as usize, nvml_free_sum as usize)
        } else {
            (s.vram_total, s.vram_free)
        };
        let gpu_name = if devices.len() > 1 {
            format!("{} GPUs", devices.len())
        } else if s.gpu_name.is_empty() {
            self.gpus.first().and_then(|g| g.name()).unwrap_or_default()
        } else {
            s.gpu_name.clone()
        };
        let vram_used = vram_total.saturating_sub(vram_free);
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

        let spec_rate = if s.spec_n.1 > 0 {
            (s.spec_n.2 as f64 / s.spec_n.1 as f64 * 1000.0).round() / 10.0
        } else {
            0.0
        };
        let spec_mean = if s.spec_n.0 > 0 {
            ((s.spec_n.2 + s.spec_n.0) as f64 / s.spec_n.0 as f64 * 100.0).round() / 100.0
        } else {
            0.0
        };
        let spec_positions = s
            .spec_pos
            .iter()
            .map(|p| json!({"tried": p.0, "right": p.1}))
            .collect::<Vec<_>>();

        json!({
            "cpu": {
                "process_percent": (proc_cpu * 10.0).round() / 10.0,
                "system_percent": (sys_cpu * 10.0).round() / 10.0,
                "process_cores_used": (proc_cores * 10.0).round() / 10.0,
                "logical_cores": cores,
                "info": self.cpu.json()
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
                "name": gpu_name,
                "utilization_percent": (gpu_pct * 10.0).round() / 10.0,
                "memory_controller_percent": (gpu_mem_pct * 10.0).round() / 10.0,
                "temperature_c": gpu_temp,
                "vram_total_bytes": vram_total,
                "vram_free_bytes": vram_free,
                "vram_used_bytes": vram_used,
                "expert_cache_slots": s.vram_slots,
                "expert_cache_hits": s.vram_hits,
                "expert_cache_misses": s.vram_misses,
                "expert_cache_hit_rate": (hit_rate * 10.0).round() / 10.0,
                "devices": devices
            },
            "process": {
                "pid": std::process::id(),
                "uptime_s": (uptime_s * 10.0).round() / 10.0,
                "model": s.model,
                "state": s.state,
                "busy": s.busy,
                "n_ctx": s.n_ctx,
                "session_tokens": s.session_tokens,
                "queued": queued,
                "tok_per_sec": (tok_s * 100.0).round() / 100.0,
                "prompt_tokens": s.prompt_tokens,
                "cached_tokens": s.cached_tokens,
                "completion_tokens": s.completion_tokens,
                "requests": s.requests,
                "mtp_active": s.mtp_active,
                "dflash_active": s.dflash_active,
                "prefill_done": crate::engine::PREFILL.0.load(std::sync::atomic::Ordering::Relaxed),
                "prefill_total": crate::engine::PREFILL.1.load(std::sync::atomic::Ordering::Relaxed)
            },
            "mcp": crate::mcp::status().into_iter().map(|(name, tools, error)| json!({"name": name, "tools": tools, "error": error})).collect::<Vec<_>>(),
        "spec": json!({"acceptance_rate": spec_rate, "mean_accepted_per_pass": spec_mean, "passes": s.spec_n.0, "drafted": s.spec_n.1, "accepted": s.spec_n.2, "lookup_passes": s.spec_n.3, "positions": spec_positions}),
            "recent": s.recent.iter().map(|r| json!({
                "at": r.at, "prompt_tokens": r.prompt_tokens, "cached_tokens": r.cached_tokens, "completion_tokens": r.completion_tokens,
                "prompt_ms": r.prompt_ms.round(), "predicted_ms": r.predicted_ms.round(), "finish": r.finish, "pictures": r.pictures,
            })).collect::<Vec<_>>()
        })
    }
}
