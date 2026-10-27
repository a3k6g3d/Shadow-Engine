//! The forward pass's matrix-vector products on the GPU (shadow_model::forward::Backend).  Dense tensors - the ones
//! with a `key` - are uploaded once and stay resident in VRAM up to a budget; one-off matrices (an expert's slice) go
//! through a reused scratch buffer.  The kernels are matvec.cu's reference ones: correct, held to the CPU within
//! 1e-4 of the sum's size, not yet fast.

use crate::driver::{Context, DeviceBuffer};
use crate::kernels::{file, Arg, Program, QUANT_KERNELS};
use shadow_model::forward::Backend;
use shadow_model::gguf::TensorInfo;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

pub struct GpuBackend<'c> {
    ctx: &'c Context,
    prog: Program<'c>,
    resident: RefCell<HashMap<String, DeviceBuffer<'c>>>,
    scratch_w: RefCell<Option<DeviceBuffer<'c>>>,
    scratch_x: RefCell<Option<DeviceBuffer<'c>>>,
    scratch_y: RefCell<Option<DeviceBuffer<'c>>>,
    budget: usize,
    pub resident_bytes: Cell<usize>,
    pub uploads: Cell<usize>,
}

impl<'c> GpuBackend<'c> {
    /// `budget`: the most VRAM the resident weights may take, in bytes.
    pub fn new(ctx: &'c Context, budget: usize) -> Result<GpuBackend<'c>, String> {
        let prog = Program::load(ctx, file("matvec.cu").ok_or("matvec.cu is not registered")?)?;
        Ok(GpuBackend {
            ctx,
            prog,
            resident: RefCell::new(HashMap::new()),
            scratch_w: RefCell::new(None),
            scratch_x: RefCell::new(None),
            scratch_y: RefCell::new(None),
            budget,
            resident_bytes: Cell::new(0),
            uploads: Cell::new(0),
        })
    }

    /// A scratch buffer of at least `bytes`, grown (never shrunk) on demand.
    fn scratch<'s>(&self, slot: &'s RefCell<Option<DeviceBuffer<'c>>>, bytes: usize)
                   -> std::cell::RefMut<'s, Option<DeviceBuffer<'c>>> {
        let mut s = slot.borrow_mut();
        if s.as_ref().is_none_or(|b| b.bytes < bytes) {
            *s = None; // free the old one first
            *s = Some(self.ctx.alloc(bytes).expect("scratch allocation"));
        }
        s
    }
}

impl Backend for GpuBackend<'_> {
    fn matvec(&self, t: &TensorInfo, bytes: &[u8], key: Option<&str>, x: &[f32]) -> Vec<f32> {
        let ty = t.ty.name();
        let &(_, qk, bb, _, entry) = QUANT_KERNELS.iter().find(|q| q.0 == ty)
            .unwrap_or_else(|| panic!("{}: no matvec kernel for {ty}", t.name));
        let cols = t.dims[0] as usize;
        let row_bytes = cols / qk as usize * bb as usize;
        let rows = bytes.len() / row_bytes;
        assert_eq!(x.len(), cols, "{}: input {} for {} columns", t.name, x.len(), cols);

        let mut resident = self.resident.borrow_mut();
        let mut scratch_w = None;
        let w_ptr = match key {
            Some(k) if resident.contains_key(k) => resident[k].ptr,
            Some(k) if self.resident_bytes.get() + bytes.len() <= self.budget => {
                let b = self.ctx.alloc(bytes.len()).expect("weight allocation");
                b.upload(bytes).expect("weight upload");
                self.resident_bytes.set(self.resident_bytes.get() + bytes.len());
                self.uploads.set(self.uploads.get() + 1);
                let p = b.ptr;
                resident.insert(k.to_string(), b);
                p
            }
            _ => {
                let mut s = self.scratch(&self.scratch_w, bytes.len());
                s.as_mut().unwrap().upload(bytes).expect("weight upload");
                self.uploads.set(self.uploads.get() + 1);
                let p = s.as_ref().unwrap().ptr;
                scratch_w = Some(s);
                p
            }
        };
        let mut sx = self.scratch(&self.scratch_x, cols * 4);
        sx.as_mut().unwrap().upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).expect("x upload");
        let sy = self.scratch(&self.scratch_y, rows * 4);
        // SAFETY: `rows` whole rows of `cols` weights at w_ptr (resident or scratch, both at least that big), cols f32 at
        // x, rows f32 at y; the kernel bounds-checks rows.
        unsafe {
            self.prog.launch(entry, ((rows as u32).div_ceil(8), 1, 1), (256, 1, 1), 0, None,
                             &[Arg::Ptr(w_ptr), Arg::Ptr(sx.as_ref().unwrap().ptr), Arg::Ptr(sy.as_ref().unwrap().ptr),
                               Arg::U32(rows as u32), Arg::U32(cols as u32)])
                .expect("matvec launch");
        }
        let mut out = vec![0u8; rows * 4];
        sy.as_ref().unwrap().download(&mut out).expect("y download"); // synchronous: waits for the kernel
        drop(scratch_w);
        out.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}
