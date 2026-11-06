# Verification plan after the 2026-11-06 prompt-path changes

Written 2026-11-06, after STATUS entries 44-52. Nothing here changes behaviour; it decides whether today's defaults stay, and records the net result honestly.

## What changed by default today (and what it could break)

| Change | STATUS | Default | Risk to check |
|---|---|---|---|
| Sub-chunk size fitted to the prompt so a long prompt runs in one pass | 45 | on (`SHADOW_ONE_PASS=0` off) | prompts of 12k-16k rows only; the lent-room carving, the conversation snapshots after such a prompt, the first decode after it |
| DeltaNet recurrence on 512 threads, 4 per state column | 49 | on for the float path (`SHADOW_DELTA_F4=0` off) | bit-identical to the old kernel in fast mode; `--exact` never uses it |
| Sparse attention scores on the tensor cores (fp16 inputs) | 50 | on (`SHADOW_ATT_TC=0` off) | fidelity on long prompts: wording changes, retrieval must not degrade |
| int8 MMQ for the dense projections (earlier, entry 43) | 43 | on when `native/mmq/shadow_mmq.dll` exists | already gated; re-run with the final binary |

Opt-in and measured as ties or losses (not defaults): `SHADOW_MMQ_MOE=1`, `SHADOW_GU_SWIGLU=1`, `SHADOW_MMA_GROUP=48/64`, `SHADOW_ATT_TC=2`, `SHADOW_ATT_TCD=1`.

## Gaps in what was verified today

1. The short-prompt suites (`quality_test`, `task_check`) were not re-run on the final binary. The changes above act on long prompts, so they should not move, but nothing has shown it.
2. `--exact` was hashed after the DeltaNet change only (162a8e07); not on the final binary.
3. Long-prompt fidelity was tested at 9.6k tokens (needle 8/8, multi-needle 12/12, both saturated). Not at 15k or at the 24k window, and not with a second turn that reuses a conversation snapshot.
4. The net speed of today's defaults against the start of the day was never measured as a pair: the wall-clock A/Bs were inside the machine's +-8% noise, so only kernel times (Nsight Systems) back the gains. Decode speed and the 3k prompt were not re-measured at all.
5. `cargo test` and `cargo clippy` were not run.
6. The new Start Menu entry: the server I started by it did not stop when I killed the launcher with `Stop-Process` (a hard kill skips its clean shutdown, so that is not proof of a bug); the icon was not looked at; a graceful Quit from the tray was not tried.

## The plan (in this order; each step names its gate)

A. **Static checks (5 min).** `cargo build --release` with no new warnings from today's code; `cargo test --release`; `cargo clippy --release` if the repo's CI uses it. Gate: all pass; anything failing is fixed or recorded.

B. **Exact path (6 min).** Server with `--exact`: `hash9k.py` must give 162a8e07; `tools/quality_test.py` 12/12. Gate: both unchanged.

C. **Fast-mode short suites (10 min).** `quality_test.py` 36-item agreement (was 0.410) and `task_check.py` (was 43/48). Gate: task_check >= 42/48; agreement within the metric's known swing (0.17-0.44) and not below the earlier default by more than 0.05 in an A/B against `SHADOW_ATT_TC=0 SHADOW_DELTA_F4=0 SHADOW_ONE_PASS=0`.

D. **Long-context correctness (25 min).**
   1. `tools/needle_test.py` at 15k tokens (about 45,000 chars) and at about 20k tokens (about 60,000 chars; `--ctx 24576`), single and `multi`, default build and `SHADOW_ATT_TC=0`. Gate: default finds at least as many as `SHADOW_ATT_TC=0` and at least 11 of 12.
   2. Snapshot reuse: a 15k prompt, then the same prompt plus a follow-up question (the server reuses the stored snapshot); compare that reply with the same follow-up on a cold server (`SHADOW_NO_PREFIX=1`). Gate: both answer the question correctly; byte-identical in `--exact`.
   3. First decode after a one-pass 15k prompt: 64 tokens, coherent, decode tok/s within 10% of the same decode after a 9.5k prompt.

E. **Speed against the start of the day (40 min).** Build commit 1dc8a90 (the dense-MMQ commit) in a separate worktree and target directory (no branch switch in this checkout; `SHADOW_MMQ_DLL` names the DLL), then A-B-B-A pairs, three servers an arm, one server per run, fresh each time: 3k, 9.5k and 15k prompts (first prompt of a server only for 15k, and never after a shorter prompt that shares its start: the cached prefix hides the work) and a 256-token decode. Gate: HEAD is not slower at any length (a tie within the spread is reported as a tie); the 15k pair is expected to show about +13% and 9.5k a few percent. The numbers go to STATUS (entry 53) whatever they are.

F. **Launcher (10 min, needs the screen).** Open Shadow from the Start menu search; check the icon in the menu, the window and the tray; quit through the tray; confirm no `shadow.exe` is left and RAM is released; open it again. Gate: clean shutdown. If the server survives a graceful Quit, that is a bug in `crates/shadow-ui` to fix (`stop_on_quit`), as its own change.

G. **Housekeeping (10 min).** Update the README's speed table, `CHANGELOG.md` and the head of `docs/STATUS.md` ("state after today"); save the process lessons to memory (nsys export does not overwrite; free RAM decides the tier; no Strata or other large process during measurements). Commit.

## Rules while it runs

- Check free RAM (about 49 GiB for the 36 GiB tier) before every server start; do not start one while another large process is running, and do not close the user's own processes.
- Every model load under `tools/run_capped.ps1`, floor 3 GiB.
- A failed gate is reported with its numbers and the change goes back to opt-in; nothing is hidden by re-running until it passes.

## Time

About two hours of machine time, most of it D and E. A to C can run first and are the cheap ones.
