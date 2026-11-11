tools/pair_e2e_test.py PORT OUT.json (STATUS 102): a 10-request conversation pair against a running `shadow serve ... --port PORT`: conversation A (a 418-token prompt, then 4 follow-ups), B (a 370-token prompt, a follow-up), back to A, B (498 tokens of history), A.  Replies are greedy.
single / pair : `--exact`, one card against `--gpus 0,0 --split 24` (both stages on the one 5070, a small expert cache each): 10 of 10 replies identical, same prompt_n / cache_n
fsingle / fpair : the default fast mode (tensor-core prompts, tail skip, verify rank 2, prompt-lookup speculation, a RAM tier), one card against the pair: coherent, not identical (inexact mode)
nocard : `--gpus 0,1` on a machine with one card: a warning and one card; 10 of 10 replies identical to `single`
