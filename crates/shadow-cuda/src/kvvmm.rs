//! A device buffer made of 2 MiB physical chunks mapped into a reserved address range (the CUDA virtual memory API), so that the unfilled tail of the KV window can be
//! handed back to the driver's pool for a while (`release_tail`) - a long prompt borrows it for its batch buffers - and mapped again afterwards (`remap_tail`).
//! The kernels keep the same base pointer throughout.  tests/vmm_probe.rs measures the calls (create 768 chunks 24 ms, release 768 MiB 30 ms on this card).

use crate::driver::{Context, CuDevicePtr};

pub struct VmmRegion<'c> {
    ctx: &'c Context,
    pub base: CuDevicePtr,
    chunk: usize,
    handles: Vec<u64>,
    mapped: usize,
    total: usize,
    /// bytes one position of the window takes in this buffer
    pub per_pos: usize,
}

fn prop() -> [u8; 32] {
    // CUmemAllocationProp: type PINNED (1), no handle types, location {DEVICE (1), id 0}, no win32 metadata, flags 0
    let mut p = [0u8; 32];
    p[0..4].copy_from_slice(&1u32.to_le_bytes());
    p[8..12].copy_from_slice(&1u32.to_le_bytes());
    p
}

fn access() -> [u8; 12] {
    // CUmemAccessDesc: location {DEVICE, 0}, flags READWRITE (3)
    let mut a = [0u8; 12];
    a[0..4].copy_from_slice(&1u32.to_le_bytes());
    a[8..12].copy_from_slice(&3u32.to_le_bytes());
    a
}

impl<'c> VmmRegion<'c> {
    /// A zeroed buffer for `cap` positions of `per_pos` bytes, all of it mapped.
    pub fn new(ctx: &'c Context, per_pos: usize, cap: usize) -> Result<VmmRegion<'c>, String> {
        let api = ctx.api;
        ctx.bind().map_err(|e| e.to_string())?;
        let p = prop();
        let mut gran = 0usize;
        // SAFETY: out-pointer and a 32-byte property block as the driver wants.
        let r = unsafe { (api.cuMemGetAllocationGranularity)(&mut gran, p.as_ptr() as *const _, 1) };
        if r != 0 {
            return Err(format!("cuMemGetAllocationGranularity failed ({r})"));
        }
        let chunk = gran.max(2 << 20);
        let n = (per_pos * cap).div_ceil(chunk).max(1);
        let mut base = 0u64;
        // SAFETY: out-pointer; reserving address space only.
        let r = unsafe { (api.cuMemAddressReserve)(&mut base, n * chunk, 0, 0, 0) };
        if r != 0 {
            return Err(format!("cuMemAddressReserve failed ({r})"));
        }
        let mut reg = VmmRegion { ctx, base, chunk, handles: Vec::with_capacity(n), mapped: 0, total: n, per_pos };
        reg.map_to(n)?;
        Ok(reg)
    }

    pub fn bytes(&self) -> usize {
        self.mapped * self.chunk
    }

    pub fn is_full(&self) -> bool {
        self.mapped == self.total
    }

    fn total_chunks(&self) -> usize {
        self.total
    }

    /// An address range for `bytes` with nothing mapped yet (the pointer stays the same for every `map_bytes` / `release_tail(0)` round).
    pub fn new_empty(ctx: &'c Context, bytes: usize) -> Result<VmmRegion<'c>, String> {
        let api = ctx.api;
        ctx.bind().map_err(|e| e.to_string())?;
        let p = prop();
        let mut gran = 0usize;
        // SAFETY: as `new`.
        let r = unsafe { (api.cuMemGetAllocationGranularity)(&mut gran, p.as_ptr() as *const _, 1) };
        if r != 0 {
            return Err(format!("cuMemGetAllocationGranularity failed ({r})"));
        }
        let chunk = gran.max(2 << 20);
        let n = bytes.div_ceil(chunk).max(1);
        let mut base = 0u64;
        let r = unsafe { (api.cuMemAddressReserve)(&mut base, n * chunk, 0, 0, 0) };
        if r != 0 {
            return Err(format!("cuMemAddressReserve failed ({r})"));
        }
        Ok(VmmRegion { ctx, base, chunk, handles: Vec::with_capacity(n), mapped: 0, total: n, per_pos: 1 })
    }

    /// Map (zeroed) chunks until `bytes` are backed.
    pub fn map_bytes(&mut self, bytes: usize) -> Result<(), String> {
        let mut empty = Vec::new();
        self.map_bytes_from(bytes, &mut empty, true)
    }

    /// Map chunks until `bytes` are backed, drawing existing physical handles from `pool` before creating new ones, and optionally zeroing the newly mapped range.
    pub fn map_bytes_from(&mut self, bytes: usize, pool: &mut Vec<u64>, zero: bool) -> Result<(), String> {
        let n = bytes.div_ceil(self.chunk).min(self.total);
        self.map_to_from(n, pool, zero)
    }

    /// Map chunks until `n` are in place (zeroed).
    fn map_to(&mut self, n: usize) -> Result<(), String> {
        let mut empty = Vec::new();
        self.map_to_from(n, &mut empty, true)
    }

    /// Map chunks until `n` are in place, drawing physical handles from `pool` first so `lend_kv` / `unlend_kv` avoid ~2,600 `cuMemCreate` / `cuMemRelease` driver calls per prompt.
    fn map_to_from(&mut self, n: usize, pool: &mut Vec<u64>, zero: bool) -> Result<(), String> {
        let api = self.ctx.api;
        self.ctx.bind().map_err(|e| e.to_string())?;
        let (p, acc) = (prop(), access());
        let from = self.mapped;
        while self.mapped < n {
            let h = match pool.pop() {
                Some(h) => h,
                None => {
                    let mut h = 0u64;
                    // SAFETY: a chunk of physical device memory; mapped at its own slot of the reserved range.
                    let r = unsafe { (api.cuMemCreate)(&mut h, self.chunk, p.as_ptr() as *const _, 0) };
                    if r != 0 {
                        return Err(format!("cuMemCreate failed ({r}): no VRAM to map the KV window's tail back"));
                    }
                    h
                }
            };
            let at = self.base + (self.mapped * self.chunk) as u64;
            let r = unsafe { (api.cuMemMap)(at, self.chunk, 0, h, 0) };
            if r != 0 {
                unsafe { (api.cuMemRelease)(h) };
                return Err(format!("cuMemMap failed ({r})"));
            }
            self.handles.push(h);
            self.mapped += 1;
        }
        if self.mapped > from {
            let at = self.base + (from * self.chunk) as u64;
            let len = (self.mapped - from) * self.chunk;
            // SAFETY: the range was just mapped.
            let r = unsafe { (api.cuMemSetAccess)(at, len, acc.as_ptr() as *const _, 1) };
            if r != 0 {
                return Err(format!("cuMemSetAccess failed ({r})"));
            }
            if zero {
                let r = unsafe { (api.cuMemsetD8)(at, 0, len) };
                if r != 0 {
                    return Err(format!("cuMemsetD8 failed ({r})"));
                }
            }
        }
        Ok(())
    }

    /// Unmap and release every chunk past the ones that hold `keep_pos` positions; the bytes handed back.  The caller has seen the device idle.
    pub fn release_tail(&mut self, keep_pos: usize) -> usize {
        let api = self.ctx.api;
        let keep = (keep_pos * self.per_pos).div_ceil(self.chunk).min(self.mapped);
        if keep >= self.mapped || self.ctx.bind().is_err() {
            return 0;
        }
        let at = self.base + (keep * self.chunk) as u64;
        // SAFETY: the chunks past `keep` are mapped, nothing on the device is using them.
        let r = unsafe { (api.cuMemUnmap)(at, (self.mapped - keep) * self.chunk) };
        if r != 0 {
            return 0;
        }
        for h in self.handles.drain(keep..) {
            unsafe { (api.cuMemRelease)(h) };
        }
        let freed = (self.mapped - keep) * self.chunk;
        self.mapped = keep;
        freed
    }

    /// Unmap every chunk past `keep_pos` positions and move their physical handles into `pool` without releasing them to the driver.
    pub fn release_tail_to(&mut self, keep_pos: usize, pool: &mut Vec<u64>) -> usize {
        let api = self.ctx.api;
        let keep = (keep_pos * self.per_pos).div_ceil(self.chunk).min(self.mapped);
        if keep >= self.mapped || self.ctx.bind().is_err() {
            return 0;
        }
        let at = self.base + (keep * self.chunk) as u64;
        // SAFETY: the chunks past `keep` are mapped, nothing on the device is using them.
        let r = unsafe { (api.cuMemUnmap)(at, (self.mapped - keep) * self.chunk) };
        if r != 0 {
            return 0;
        }
        pool.extend(self.handles.drain(keep..));
        let freed = (self.mapped - keep) * self.chunk;
        self.mapped = keep;
        freed
    }

    /// Back to the whole window mapped (zeroed tail).
    pub fn remap_tail(&mut self) -> Result<(), String> {
        let n = self.total_chunks();
        self.map_to(n)
    }

    /// Back to the whole window mapped (zeroed tail), reusing physical handles from `pool`.
    pub fn remap_tail_from(&mut self, pool: &mut Vec<u64>) -> Result<(), String> {
        let n = self.total_chunks();
        self.map_to_from(n, pool, true)
    }

    /// Release any leftover physical handles in `pool`.
    pub fn release_pool(ctx: &Context, pool: &mut Vec<u64>) {
        if pool.is_empty() || ctx.bind().is_err() {
            return;
        }
        let api = ctx.api;
        for h in pool.drain(..) {
            unsafe { (api.cuMemRelease)(h) };
        }
    }
}

impl Drop for VmmRegion<'_> {
    fn drop(&mut self) {
        let api = self.ctx.api;
        if self.ctx.bind().is_err() {
            return;
        }
        // SAFETY: unmapping what this region mapped, releasing its handles, freeing its address range.
        unsafe {
            if self.mapped > 0 {
                (api.cuMemUnmap)(self.base, self.mapped * self.chunk);
            }
            for &h in &self.handles {
                (api.cuMemRelease)(h);
            }
            (api.cuMemAddressFree)(self.base, self.total * self.chunk);
        }
    }
}
