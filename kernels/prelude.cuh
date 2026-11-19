// Prepended to every kernel program before NVRTC sees it.  Our own small replacement for the toolkit headers, which
// are not on a user's PC: fixed-width integer names, and f16/bf16 <-> f32 through inline PTX `cvt`, so a kernel never
// needs cuda_fp16.h.  Keep it small; anything a single kernel family needs goes in that family's file.

typedef unsigned char      u8;
typedef signed char        i8;
typedef unsigned short     u16;
typedef short              i16;
typedef unsigned int       u32;
typedef int                i32;
typedef unsigned long long u64;
typedef long long          i64;

static __device__ __forceinline__ float f16_to_f32(u16 h) {
    float f;
    asm("{ .reg .b16 t; mov.b16 t, %1; cvt.f32.f16 %0, t; }" : "=f"(f) : "h"(h));
    return f;
}

static __device__ __forceinline__ u16 f32_to_f16(float f) {
    u16 h;
    asm("{ .reg .b16 t; cvt.rn.f16.f32 t, %1; mov.b16 %0, t; }" : "=h"(h) : "f"(f));
    return h;
}

// bf16 is the top half of an f32, so widening is a shift; narrowing rounds to nearest even.
static __device__ __forceinline__ float bf16_to_f32(u16 b) {
    return __uint_as_float(((u32) b) << 16);
}

static __device__ __forceinline__ u16 f32_to_bf16(float f) {
    u16 h;
    asm("{ .reg .b16 t; cvt.rn.bf16.f32 t, %1; mov.b16 %0, t; }" : "=h"(h) : "f"(f));
    return h;
}

// Programmatic Dependent Launch (PDL, SM 9.0+ / Blackwell SM 12.0): lets a dependent kernel start its CTA prologue
// as soon as the primary kernel finishes its work, and wait for the primary kernel's global stores before reading them.
// By default (`griddepcontrol.wait;` at kernel start + `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` on launch),
// the hardware automatically signals dependents at primary kernel completion while overlapping dependent CTA setup.
// Explicit `griddepcontrol.launch_dependents;` is opt-in (`PDL_EARLY_SIGNAL`) for single-wave primary kernels.
static __device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(PDL_NO_GRIDDEP)
    asm volatile("griddepcontrol.wait;");
#endif
}

static __device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(PDL_NO_GRIDDEP) && defined(PDL_EARLY_SIGNAL)
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}
