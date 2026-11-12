STATUS 104: the hop between two stages (one 5070, both stages on it, --gpus 0,0 --split 24, 0.6 GiB caches each, fast mode with the drafter).
old_hop / pinned_hop = tools/pair_e2e_test.py conversation with SHADOW_HOP_TIME=1; the last "hop:" line of each .log.err is the cumulative timing.
prompt_*.client.txt = the abandoned prompt-speed A-B-B-A (39 tok/s with these caches: the experts stream from disk, the hop is under 1% of it).
