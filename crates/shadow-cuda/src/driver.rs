//! The CUDA driver API, bound by hand: the ~30 entry points stage 0 to 2 need, resolved from `nvcuda.dll` at run
//! time.  Versioned names (`_v2`) are asked for first, because the unversioned ones are the old ABI.
//!
//! Ownership: `Context` retains the device's primary context and releases it on drop; `DeviceBuffer`, `Module`,
//! `Stream` and `Event` free what they own on drop.  Every call makes the context current on the calling thread first,
//! so a value may be used from any thread one at a time.  None of them is `Sync`.

use crate::dylib::Dylib;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::OnceLock;

pub type CuResult = c_int;
pub type CuDevice = c_int;
pub type CuDevicePtr = u64;
type H = *mut c_void;

pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError {
    pub code: CuResult,
    pub call: &'static str,
    pub name: String,
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} failed: {} ({})", self.call, self.name, self.code)
    }
}

impl std::error::Error for CudaError {}

macro_rules! table {
    ($($field:ident : fn($($a:ty),*) = [$($name:literal),+];)*) => {
        #[allow(non_snake_case)]
        pub struct Api {
            pub lib: Dylib,
            $(pub $field: unsafe extern "system" fn($($a),*) -> CuResult,)*
            /// `cuMemcpyBatchAsync_v2` (CUDA 12.8+ drivers): many copies in one call; None on an older driver (the callers copy one by one)
            pub cuMemcpyBatchAsync: Option<unsafe extern "system" fn(*mut u64, *mut u64, *mut usize, usize, *mut c_void, *mut usize, usize, H) -> CuResult>,
        }
        impl Api {
            fn load(lib: Dylib) -> Result<Api, String> {
                Ok(Api {
                    // SAFETY: each symbol is the driver's own export of that name, and the type written here is its
                    // documented signature for the versioned name asked for first.
                    $($field: unsafe { std::mem::transmute::<*mut c_void, unsafe extern "system" fn($($a),*) -> CuResult>(lib.sym_any(&[$($name),+])?) },)*
                    // SAFETY: the documented signature of `cuMemcpyBatchAsync_v2`.
                    cuMemcpyBatchAsync: lib.sym_any(&["cuMemcpyBatchAsync_v2"]).ok().map(|p| unsafe {
                        std::mem::transmute::<*mut c_void, unsafe extern "system" fn(*mut u64, *mut u64, *mut usize, usize, *mut c_void, *mut usize, usize, H) -> CuResult>(p)
                    }),
                    lib,
                })
            }
        }
    };
}

table! {
    cuInit: fn(c_uint) = ["cuInit"];
    cuDriverGetVersion: fn(*mut c_int) = ["cuDriverGetVersion"];
    cuGetErrorName: fn(CuResult, *mut *const c_char) = ["cuGetErrorName"];
    cuDeviceGetCount: fn(*mut c_int) = ["cuDeviceGetCount"];
    cuDeviceGet: fn(*mut CuDevice, c_int) = ["cuDeviceGet"];
    cuDeviceGetName: fn(*mut c_char, c_int, CuDevice) = ["cuDeviceGetName"];
    cuDeviceGetAttribute: fn(*mut c_int, c_int, CuDevice) = ["cuDeviceGetAttribute"];
    cuDeviceTotalMem: fn(*mut usize, CuDevice) = ["cuDeviceTotalMem_v2"];
    cuDevicePrimaryCtxRetain: fn(*mut H, CuDevice) = ["cuDevicePrimaryCtxRetain"];
    cuDevicePrimaryCtxSetFlags: fn(CuDevice, c_uint) = ["cuDevicePrimaryCtxSetFlags_v2", "cuDevicePrimaryCtxSetFlags"];
    cuDevicePrimaryCtxRelease: fn(CuDevice) = ["cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease"];
    cuCtxSetCurrent: fn(H) = ["cuCtxSetCurrent"];
    cuCtxSynchronize: fn() = ["cuCtxSynchronize"];
    cuMemGetInfo: fn(*mut usize, *mut usize) = ["cuMemGetInfo_v2"];
    cuMemAlloc: fn(*mut CuDevicePtr, usize) = ["cuMemAlloc_v2"];
    cuMemFree: fn(CuDevicePtr) = ["cuMemFree_v2"];
    cuMemcpyHtoD: fn(CuDevicePtr, *const c_void, usize) = ["cuMemcpyHtoD_v2"];
    cuMemcpyDtoH: fn(*mut c_void, CuDevicePtr, usize) = ["cuMemcpyDtoH_v2"];
    cuMemcpyHtoDAsync: fn(CuDevicePtr, *const c_void, usize, H) = ["cuMemcpyHtoDAsync_v2"];
    cuMemcpyDtoD: fn(CuDevicePtr, CuDevicePtr, usize) = ["cuMemcpyDtoD_v2"];
    cuMemcpyDtoHAsync: fn(*mut c_void, CuDevicePtr, usize, H) = ["cuMemcpyDtoHAsync_v2"];
    cuMemsetD8: fn(CuDevicePtr, u8, usize) = ["cuMemsetD8_v2"];
    // the virtual memory API (physical chunks mapped into a reserved address range), for lending the unfilled part of the KV window
    cuMemGetAllocationGranularity: fn(*mut usize, *const c_void, c_uint) = ["cuMemGetAllocationGranularity"];
    cuMemCreate: fn(*mut u64, usize, *const c_void, u64) = ["cuMemCreate"];
    cuMemAddressReserve: fn(*mut CuDevicePtr, usize, usize, CuDevicePtr, u64) = ["cuMemAddressReserve"];
    cuMemAddressFree: fn(CuDevicePtr, usize) = ["cuMemAddressFree"];
    cuMemMap: fn(CuDevicePtr, usize, usize, u64, u64) = ["cuMemMap"];
    cuMemUnmap: fn(CuDevicePtr, usize) = ["cuMemUnmap"];
    cuMemSetAccess: fn(CuDevicePtr, usize, *const c_void, usize) = ["cuMemSetAccess"];
    cuMemRelease: fn(u64) = ["cuMemRelease"];
    cuMemHostRegister: fn(*mut c_void, usize, c_uint) = ["cuMemHostRegister_v2"];
    cuMemHostUnregister: fn(*mut c_void) = ["cuMemHostUnregister"];
    cuMemHostGetDevicePointer: fn(*mut CuDevicePtr, *mut c_void, c_uint) = ["cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer"];
    cuModuleLoadData: fn(*mut H, *const c_void) = ["cuModuleLoadData"];
    cuModuleUnload: fn(H) = ["cuModuleUnload"];
    cuModuleGetFunction: fn(*mut H, H, *const c_char) = ["cuModuleGetFunction"];
    cuModuleGetGlobal: fn(*mut CuDevicePtr, *mut usize, H, *const c_char) = ["cuModuleGetGlobal_v2", "cuModuleGetGlobal"];
    cuLaunchKernel: fn(H, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, H, *mut *mut c_void, *mut *mut c_void) = ["cuLaunchKernel"];
    cuStreamCreate: fn(*mut H, c_uint) = ["cuStreamCreate"];
    cuStreamDestroy: fn(H) = ["cuStreamDestroy_v2"];
    cuStreamSynchronize: fn(H) = ["cuStreamSynchronize"];
    cuStreamWaitEvent: fn(H, H, c_uint) = ["cuStreamWaitEvent"];
    cuEventCreate: fn(*mut H, c_uint) = ["cuEventCreate"];
    cuEventDestroy: fn(H) = ["cuEventDestroy_v2"];
    cuEventRecord: fn(H, H) = ["cuEventRecord"];
    cuEventSynchronize: fn(H) = ["cuEventSynchronize"];
    cuEventElapsedTime: fn(*mut f32, H, H) = ["cuEventElapsedTime_v2", "cuEventElapsedTime"];
    cuStreamBeginCapture: fn(H, c_uint) = ["cuStreamBeginCapture_v2", "cuStreamBeginCapture"];
    cuStreamEndCapture: fn(H, *mut H) = ["cuStreamEndCapture"];
    cuGraphInstantiateWithFlags: fn(*mut H, H, u64) = ["cuGraphInstantiateWithFlags"];
    cuGraphLaunch: fn(H, H) = ["cuGraphLaunch"];
    cuGraphExecDestroy: fn(H) = ["cuGraphExecDestroy"];
    cuGraphDestroy: fn(H) = ["cuGraphDestroy"];
}

static API: OnceLock<Result<Api, String>> = OnceLock::new();

/// The driver, loaded and initialised once.  `Err` on a machine without an NVIDIA driver: that is the skip path, not
/// a failure, and callers say so out loud.
pub fn api() -> Result<&'static Api, String> {
    API.get_or_init(|| {
        let lib = Dylib::open(std::path::Path::new("nvcuda.dll"))?;
        let api = Api::load(lib)?;
        // SAFETY: flags must be 0.
        let r = unsafe { (api.cuInit)(0) };
        if r != 0 {
            return Err(format!("cuInit failed ({r})"));
        }
        Ok(api)
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn check(api: &Api, r: CuResult, call: &'static str) -> Result<(), CudaError> {
    if r == 0 {
        return Ok(());
    }
    let mut p: *const c_char = std::ptr::null();
    // SAFETY: an out-pointer; the driver returns a static string or leaves it null.
    let name = if unsafe { (api.cuGetErrorName)(r, &mut p) } == 0 && !p.is_null() {
        // SAFETY: the driver's static, NUL-terminated name.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    } else {
        "unknown".to_string()
    };
    Err(CudaError { code: r, call, name })
}

macro_rules! call {
    ($api:expr, $f:ident ( $($arg:expr),* )) => {
        // SAFETY: each call site passes the arguments the documented signature asks for; see the call sites.
        check($api, unsafe { ($api.$f)($($arg),*) }, stringify!($f))
    };
}

/// One GPU, through its primary context.
pub struct Context {
    pub api: &'static Api,
    pub device: CuDevice,
    ctx: H,
    pub name: String,
    pub cc: (i32, i32),
    pub sms: i32,
    pub total_mem: usize,
}

impl Context {
    pub fn new(ordinal: i32) -> Result<Context, Box<dyn std::error::Error>> {
        let api = api()?;
        let mut dev = 0;
        call!(api, cuDeviceGet(&mut dev, ordinal))?;
        let mut ctx: H = std::ptr::null_mut();
        // how the host waits for the GPU (every layer's router read): spin (the default here), yield, or the driver's
        // choice - `SHADOW_SYNC=spin|yield|auto`.  A spinning wait wakes up sooner than a yielding or blocking one.
        let flags = match std::env::var("SHADOW_SYNC").as_deref() {
            Ok("auto") => 0,
            Ok("yield") => 2,
            _ => 1,
        };
        let _ = call!(api, cuDevicePrimaryCtxSetFlags(dev, flags)); // refused when the context is already active
        call!(api, cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
        call!(api, cuCtxSetCurrent(ctx))?;
        let mut name = [0 as c_char; 256];
        call!(api, cuDeviceGetName(name.as_mut_ptr(), 256, dev))?;
        // SAFETY: the driver NUL-terminates within the 256 bytes it was given.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let attr = |a: c_int| -> Result<i32, CudaError> {
            let mut v = 0;
            call!(api, cuDeviceGetAttribute(&mut v, a, dev))?;
            Ok(v)
        };
        let cc = (attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?, attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?);
        let sms = attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let mut total_mem = 0usize;
        call!(api, cuDeviceTotalMem(&mut total_mem, dev))?;
        Ok(Context { api, device: dev, ctx, name, cc, sms, total_mem })
    }

    /// Makes this context current on the calling thread.  Every method below calls it first.
    pub fn bind(&self) -> Result<(), CudaError> {
        call!(self.api, cuCtxSetCurrent(self.ctx))
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.bind()?;
        call!(self.api, cuCtxSynchronize())
    }

    /// (free, total) device memory, in bytes.
    pub fn mem_info(&self) -> Result<(usize, usize), CudaError> {
        self.bind()?;
        let (mut free, mut total) = (0usize, 0usize);
        call!(self.api, cuMemGetInfo(&mut free, &mut total))?;
        Ok((free, total))
    }

    pub fn alloc(&self, bytes: usize) -> Result<DeviceBuffer<'_>, CudaError> {
        self.bind()?;
        let mut p: CuDevicePtr = 0;
        call!(self.api, cuMemAlloc(&mut p, bytes.max(1)))?;
        Ok(DeviceBuffer { ctx: self, ptr: p, bytes, owned: true })
    }

    /// A module from a CUBIN (or PTX) image.
    pub fn load_module(&self, image: &[u8]) -> Result<Module<'_>, CudaError> {
        self.bind()?;
        // cuModuleLoadData reads a PTX image as a NUL-terminated string; a CUBIN is self-describing.  Copying into a
        // NUL-terminated buffer covers both.
        let mut img = image.to_vec();
        img.push(0);
        let mut m: H = std::ptr::null_mut();
        call!(self.api, cuModuleLoadData(&mut m, img.as_ptr() as *const c_void))?;
        Ok(Module { ctx: self, module: m })
    }

    pub fn stream(&self) -> Result<Stream<'_>, CudaError> {
        self.bind()?;
        let mut s: H = std::ptr::null_mut();
        call!(self.api, cuStreamCreate(&mut s, 1))?; // CU_STREAM_NON_BLOCKING
        Ok(Stream { ctx: self, stream: s })
    }

    /// A stream that orders with the legacy default stream both ways (work on it waits for the default stream's earlier
    /// work, and the default stream for its), unlike `stream`; what a CUDA graph is captured and launched on, so the
    /// graph's kernels fall in line with everything launched on the default stream.
    pub fn blocking_stream(&self) -> Result<Stream<'_>, CudaError> {
        self.bind()?;
        let mut s: H = std::ptr::null_mut();
        call!(self.api, cuStreamCreate(&mut s, 0))?; // CU_STREAM_DEFAULT
        Ok(Stream { ctx: self, stream: s })
    }

    pub fn event(&self) -> Result<Event<'_>, CudaError> {
        self.bind()?;
        let mut e: H = std::ptr::null_mut();
        call!(self.api, cuEventCreate(&mut e, 0))?;
        Ok(Event { ctx: self, event: e })
    }

    /// Makes `stream` (None: the default stream) wait for `event` before running anything queued after this.
    pub fn stream_wait_event(&self, stream: Option<&Stream>, event: &Event) -> Result<(), CudaError> {
        self.bind()?;
        let s = stream.map(|s| s.stream).unwrap_or(std::ptr::null_mut());
        call!(self.api, cuStreamWaitEvent(s, event.event, 0))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: balances the retain in `new`.
        unsafe { (self.api.cuDevicePrimaryCtxRelease)(self.device) };
    }
}

/// Device memory, freed on drop.  Byte-sized: the typed view is the kernel's business, and its signature is checked
/// at the launch.
pub struct DeviceBuffer<'c> {
    ctx: &'c Context,
    pub ptr: CuDevicePtr,
    pub bytes: usize,
    /// false for a `view` of memory another buffer owns: dropping it frees nothing
    owned: bool,
}

impl<'c> DeviceBuffer<'c> {
    /// A buffer over `bytes` at `ptr`, inside memory that another buffer owns and that outlives this one.
    ///
    /// # Safety
    /// `[ptr, ptr + bytes)` must stay allocated, and not be used through another buffer, while this value (or anything made from it) is used.
    pub unsafe fn view(ctx: &'c Context, ptr: CuDevicePtr, bytes: usize) -> DeviceBuffer<'c> {
        DeviceBuffer { ctx, ptr, bytes, owned: false }
    }
}

impl DeviceBuffer<'_> {
    pub fn upload(&self, src: &[u8]) -> Result<(), CudaError> {
        assert!(src.len() <= self.bytes, "upload of {} bytes into a {}-byte buffer", src.len(), self.bytes);
        self.ctx.bind()?;
        call!(self.ctx.api, cuMemcpyHtoD(self.ptr, src.as_ptr() as *const c_void, src.len()))
    }

    pub fn download(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        assert!(dst.len() <= self.bytes, "download of {} bytes from a {}-byte buffer", dst.len(), self.bytes);
        self.ctx.bind()?;
        call!(self.ctx.api, cuMemcpyDtoH(dst.as_mut_ptr() as *mut c_void, self.ptr, dst.len()))
    }

    /// The first `bytes` of `src` into the first `bytes` of this buffer, on the device (ordered on the default stream).
    pub fn copy_prefix_from(&self, src: &DeviceBuffer, bytes: usize) -> Result<(), CudaError> {
        assert!(bytes <= self.bytes && bytes <= src.bytes, "copy of {bytes} bytes between {} and {}", src.bytes, self.bytes);
        if bytes == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        call!(self.ctx.api, cuMemcpyDtoD(self.ptr, src.ptr, bytes))
    }

    pub fn zero(&mut self) -> Result<(), CudaError> {
        self.ctx.bind()?;
        call!(self.ctx.api, cuMemsetD8(self.ptr, 0, self.bytes))
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        if self.owned && self.ctx.bind().is_ok() {
            // SAFETY: the allocation cuMemAlloc made for this value, freed once.
            unsafe { (self.ctx.api.cuMemFree)(self.ptr) };
        }
    }
}

pub struct Module<'c> {
    ctx: &'c Context,
    module: H,
}

impl<'c> Module<'c> {
    pub fn ctx(&self) -> &'c Context {
        self.ctx
    }

    /// Sets a `__device__` pointer variable of the module (`name`, an `extern "C"` global of 8 bytes) to `value` (a device address; 0 clears it).
    pub fn set_global_ptr(&self, name: &str, value: u64) -> Result<(), CudaError> {
        self.ctx.bind()?;
        let c = std::ffi::CString::new(name).expect("global names have no NUL");
        let (mut p, mut size): (CuDevicePtr, usize) = (0, 0);
        call!(self.ctx.api, cuModuleGetGlobal(&mut p, &mut size, self.module, c.as_ptr()))?;
        assert_eq!(size, 8, "{name}: not a pointer");
        call!(self.ctx.api, cuMemcpyHtoD(p, &value as *const u64 as *const c_void, 8))
    }

    /// An entry point by its unmangled (`extern "C"`) name.
    pub fn function(&self, name: &str) -> Result<RawFunction<'_>, CudaError> {
        self.ctx.bind()?;
        let c = std::ffi::CString::new(name).expect("kernel names have no NUL");
        let mut f: H = std::ptr::null_mut();
        call!(self.ctx.api, cuModuleGetFunction(&mut f, self.module, c.as_ptr()))?;
        Ok(RawFunction { ctx: self.ctx, f })
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        if self.ctx.bind().is_ok() {
            // SAFETY: the module cuModuleLoadData made for this value; its functions borrow it, so none outlives it.
            unsafe { (self.ctx.api.cuModuleUnload)(self.module) };
        }
    }
}

/// An entry point, borrowed from its module.  Launching it is `unsafe`: the argument list is the kernel's contract,
/// and `crate::kernels` is where it is checked.
pub struct RawFunction<'m> {
    ctx: &'m Context,
    f: H,
}

impl RawFunction<'_> {
    /// # Safety
    /// `params` must point to one value per kernel parameter, each the parameter's exact type and size, and every
    /// device pointer among them must be valid for what the kernel does with it.
    pub unsafe fn launch(&self, grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32, stream: Option<&Stream>,
                         params: &mut [*mut c_void]) -> Result<(), CudaError> {
        self.ctx.bind()?;
        let s = stream.map(|s| s.stream).unwrap_or(std::ptr::null_mut());
        call!(self.ctx.api, cuLaunchKernel(self.f, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared, s,
                                           params.as_mut_ptr(), std::ptr::null_mut()))
    }
}

pub struct Stream<'c> {
    ctx: &'c Context,
    stream: H,
}

impl Stream<'_> {
    pub fn raw(&self) -> *mut c_void {
        self.stream
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.bind()?;
        call!(self.ctx.api, cuStreamSynchronize(self.stream))
    }

    pub fn wait_event(&self, event: &Event) -> Result<(), CudaError> {
        self.ctx.stream_wait_event(Some(self), event)
    }

    /// Starts recording this stream's launches into a graph instead of running them (relaxed mode: other threads may
    /// use the driver meanwhile).  Nothing may touch the default stream until `end_capture`.
    pub fn begin_capture(&self) -> Result<(), CudaError> {
        self.ctx.bind()?;
        call!(self.ctx.api, cuStreamBeginCapture(self.stream, 2)) // CU_STREAM_CAPTURE_MODE_RELAXED
    }

}

impl<'c> Stream<'c> {
    /// The launches recorded since `begin_capture`, instantiated.
    pub fn end_capture(&self) -> Result<Graph<'c>, CudaError> {
        self.ctx.bind()?;
        let mut g: H = std::ptr::null_mut();
        call!(self.ctx.api, cuStreamEndCapture(self.stream, &mut g))?;
        let mut exec: H = std::ptr::null_mut();
        if let Err(e) = call!(self.ctx.api, cuGraphInstantiateWithFlags(&mut exec, g, 0)) {
            // SAFETY: the graph cuStreamEndCapture made.
            unsafe { (self.ctx.api.cuGraphDestroy)(g) };
            return Err(e);
        }
        Ok(Graph { ctx: self.ctx, graph: g, exec })
    }
}

/// A captured, instantiated sequence of launches (`Stream::end_capture`), replayed with one call.
pub struct Graph<'c> {
    ctx: &'c Context,
    graph: H,
    exec: H,
}

impl Graph<'_> {
    pub fn launch(&self, stream: &Stream) -> Result<(), CudaError> {
        self.ctx.bind()?;
        call!(self.ctx.api, cuGraphLaunch(self.exec, stream.stream))
    }
}

impl Drop for Graph<'_> {
    fn drop(&mut self) {
        if self.ctx.bind().is_ok() {
            // SAFETY: the executable and graph `end_capture` made for this value.
            unsafe {
                (self.ctx.api.cuGraphExecDestroy)(self.exec);
                (self.ctx.api.cuGraphDestroy)(self.graph);
            }
        }
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if self.ctx.bind().is_ok() {
            // SAFETY: the stream cuStreamCreate made for this value.
            unsafe { (self.ctx.api.cuStreamDestroy)(self.stream) };
        }
    }
}

pub struct Event<'c> {
    ctx: &'c Context,
    event: H,
}

impl Event<'_> {
    pub fn record(&self, stream: Option<&Stream>) -> Result<(), CudaError> {
        self.ctx.bind()?;
        let s = stream.map(|s| s.stream).unwrap_or(std::ptr::null_mut());
        call!(self.ctx.api, cuEventRecord(self.event, s))
    }

    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.bind()?;
        call!(self.ctx.api, cuEventSynchronize(self.event))
    }

    /// Milliseconds from `start` to this event; both must have completed.
    pub fn since(&self, start: &Event) -> Result<f32, CudaError> {
        self.ctx.bind()?;
        let mut ms = 0f32;
        call!(self.ctx.api, cuEventElapsedTime(&mut ms, start.event, self.event))?;
        Ok(ms)
    }
}

impl Drop for Event<'_> {
    fn drop(&mut self) {
        if self.ctx.bind().is_ok() {
            // SAFETY: the event cuEventCreate made for this value.
            unsafe { (self.ctx.api.cuEventDestroy)(self.event) };
        }
    }
}
