# Restore tests for the DFlash drafter (STATUS 98-99), run on 2026-11-11. `tools/df_restore_test.py PORT OUT.json`: one 34-turn conversation (varied36 prompts)
# against a running `shadow serve ... --port PORT --ctx 8192`; at turn 24 the previous assistant reply is altered so the session's tokens no longer start the
# prompt: the server resets and restores a conversation snapshot.  Read the per-request `speculation:` lines of the server log (drafting seconds, accept) before and after.
# `tools/df_same_replies_test.py PORT OUT.json`: 3 turns, `cache_prompt: false`, to compare replies between builds.  Files here:
#   dfr_k1c / dfr_k0c : the 34-turn test, SHADOW_DF_KEEP=1 (the fix) / 0 (old behaviour); dfr_k1b / dfr_k0b : the same on the first version of the fix (rollback bug, kept)
#   one_old / one_new1 / one_new0 : the 3-turn test on the old exe / the fix / the fix with SHADOW_DF_KEEP=0 (identical replies)
