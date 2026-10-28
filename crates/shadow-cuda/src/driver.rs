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
        }
        impl Api {
            fn load(lib: Dylib) -> Result<Api, String> {
                Ok(Api {
                    // SAFETY: each symbol is the driver's own export of that name, and the type written here is its
                    // documented signature for the versioned name asked for first.
                    $($field: unsafe { std::mem::transmute::<*mut c_void, unsafe extern "system" fn($($a),*) -> CuResult>(lib.sym_any(&[$($name),+])?) },)*
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
    cuMemHostRegister: fn(*mut c_void, usize, c_uint) = ["cuMemHostRegister_v2"];
    cuMemHostUnregister: fn(*mut c_void) = ["cuMemHostUnregister"];
    cuMemHostGetDevicePointer: fn(*mut CuDevicePtr, *mut c_void, c_uint) = ["cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer"];
    cuModuleLoadData: fn(*mut H, *const c_void) = ["cuModuleLoadData"];
    cuModuleUnload: fn(H) = ["cuModuleUnload"];
    cuModuleGetFunction: fn(*mut H, H, *const c_char) = ["cuModuleGetFunction"];
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
        Ok(DeviceBuffer { ctx: self, ptr: p, bytes })
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
        if self.ctx.bind().is_ok() {
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
