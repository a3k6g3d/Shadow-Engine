/* strata C ABI - the only interface allowed between the Rust host and the C++ device layer.
 *
 * Hand-written on purpose: no bindgen, no cxx, no generated code in the build. The header IS the contract, and
 * the Rust crate's test suite parses this file and asserts the version constant matches its own. That guard is
 * the whole reason a hand-written header is worth its upkeep: the failure mode of a cross-language seam is a
 * field added on one side and a silent mismatch on the other, and a generated header cannot catch that.
 *
 * Rules:
 *   - C ABI only. No C++ type crosses this line. Buffers are (ptr, len), errors are this enum.
 *   - Nothing returned by the facade is owned by it. Anything Rust hands down must outlive the call.
 *   - Adding a field to a struct below means bumping STRATA_ABI_VERSION, and the Rust mirror, in the same commit.
 */
#ifndef STRATA_C_ABI_H
#define STRATA_C_ABI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Keep in step with strata_abi::ABI_VERSION in rust-host/crates/strata-abi/src/lib.rs. */
#define STRATA_ABI_VERSION 1

/* Stable across releases: the Rust side matches on these by value, never by string. */
typedef enum strata_status {
    STRATA_OK = 0,
    STRATA_ERR_WOULD_BLOCK = 1,   /* the ring is full; the caller owns the retry, the entry was NOT consumed */
    STRATA_ERR_BUFFER_TOO_SHORT = 2,
    STRATA_ERR_BAD_SLOT = 3,
    STRATA_ERR_DEVICE_UNAVAILABLE = 4,
    STRATA_ERR_ABI_VERSION = 5    /* the two sides disagree about this header */
} strata_status;

/* One expert the host has decided the device may read. The host owns these bytes for exactly as long as the
 * device holds the borrow - which is the rule the C++ side states as a comment today (expert_source.hpp:200:
 * "stays valid for the layer it was asked in and the next one or two"). Here it is a borrow, not a comment. */
typedef struct strata_expert_ref {
    const unsigned char *bytes;
    unsigned long long len;
    int slot;
    int source_tier;   /* 0 = pinned RAM, 1 = NVMe, 2 = SSD, 3 = model file */
} strata_expert_ref;

/* Milestone 1 needed only the handshake.  Milestone 2 is the DLL, so the calls below are the ones that make it a
 * link test rather than a compile test: each reaches into an existing static library and returns what that code
 * decided.  Without such a reference the linker drops the library and the test passes without ever having linked
 * it - which is exactly the silent pass this whole seam is built to prevent. */
#ifdef _WIN32
  #ifdef STRATA_FACADE_BUILD
    #define STRATA_API __declspec(dllexport)
  #else
    #define STRATA_API __declspec(dllimport)
  #endif
#else
  #define STRATA_API __attribute__((visibility("default")))
#endif

STRATA_API int strata_abi_version(void);
STRATA_API const char *strata_abi_last_error(void);

/* The seam's first real call: the policy's own decision, pulled out of strata_spec (MSVC-built, no CUDA).
 * `lookup_out` and `t_out` are the caller's; the facade writes them and owns nothing it writes.  On a policy with
 * no measurements yet, choose(3, 4, 6) is decided by the priors alone, which makes it a fixed, checkable answer
 * on both sides of the seam: lookup=1, t=4. */
STRATA_API int strata_facade_probe_policy(int t_mtp, int lookup_k, int match,
                                          int *lookup_out, int *t_out);

/* The toolchain question this milestone exists to answer: can cudart be linked into something Rust calls?
 * Built with CUDA, this returns STRATA_OK and the runtime version; built without it, STRATA_ERR_DEVICE_UNAVAILABLE
 * - an honest answer, not a skipped test. */
STRATA_API int strata_facade_cudart_version(int *runtime_version_out);

/* Milestone 3: the first call that carries a struct across the seam, and the first where the bytes being
 * described were read by the caller - by the Rust ring, through its own IOCP, on this machine.  `ref` is the
 * struct above, and it is the rules made concrete: the host owns those bytes for exactly as long as this call
 * runs, and the facade does not copy them, cache them, or free them.
 *
 * `x` is the H-wide activation and `out` is the H-wide result; both are the caller's buffers, passed as
 * (ptr, len), and the facade owns nothing it writes into them.  The facade runs the existing kernel on the
 * bytes it is given - the same s2_expert_vnni the C++ path runs, no Rust kernel code anywhere - and refuses
 * with STRATA_ERR_DEVICE_UNAVAILABLE on a CPU it has not been given permission to run it on.
 *
 * The version stays at 1.  An added function cannot cause the silent mismatch the version exists to catch:
 * a caller built against v1 never calls it, and a caller that does fails at GetProcAddress, loudly. */
STRATA_API int strata_facade_expert_apply(const strata_expert_ref *ref,
                                          const float *x, unsigned long long x_len,
                                          float *out, unsigned long long out_cap);

#ifdef __cplusplus
}
#endif

#endif /* STRATA_C_ABI_H */
