# ARCH.md: the model, as math, from the file and the reference

Written 2026-10-26 (stage 0 of [SHADOW_PLAN.md](SHADOW_PLAN.md)). This is the spec the CPU reference forward pass
(stage 1) is written from, and the one every kernel is tested against. It has three kinds of statement, and each line
says which:

- **[file]** read from the GGUF itself: `shadow inspect` / `shadow kv` / `shadow stats` (`docs/model-inspect.txt`).
- **[ref]** from the published reference implementation: Hugging Face transformers, `models/qwen4_exp` (modular file),
  which inherits from `qwen3_5` / `qwen3_next`, Apache-2.0; and the base model's `config.json`
  (huggingface.co/Qwen/Qwen3.8-Flash-Next). Read 2026-10-26.
- **[open]** not settled. Each one is collected again at the end with how to settle it.

Nothing here comes from Strata.

**This answers the plan's oracle question.** Option 1, llama.cpp: the only build on this PC is Strata's fork and it
failed three loads (`J:\Strata-work\RUN.md`), so it is out. Mainline llama.cpp now has a `qwen4exp` converter and
knows type 42 (Q2_0), so a mainline build is a possible second oracle, but it fuses tensors this file keeps separate
and may not load this file as it is. **Option 2 holds:** the architecture has a public reference implementation, and
the base model's weights are public (BF16 safetensors, 131 shards, 360 GB). A fixture can be made from them without
downloading all of it (see the end).

## Shapes

Notation: `H = 2560` hidden, `C = 4` hyper-connection streams, `CH = 10240`, `T` = tokens. ggml shape order is
`[in, out]` for a matrix (the first dimension is the row length).

| | value | source |
|---|---|---|
| layers | 48; layer `l` is full attention (K) iff `l % 4 == 3`, else linear (J) | [file] tensors + `full_attention_interval 4`; [ref] `layer_types` |
| vocab | 248,320; `token_embd` IQ4_XS `[2560, 248320]`; `output` Q6_K `[2560, 248320]`, not tied | [file]; [ref] `tie_word_embeddings false` |
| norm eps | 1e-6 everywhere | [file] `layer_norm_rms_epsilon`; [ref] `rms_norm_eps` |
| MoE | 512 experts, top 10, intermediate 640; shared expert 640 | [file] [ref] |
| J (GatedDeltaNet) | 16 key heads x 128, 48 value heads x 128, conv kernel 4 | [file] `ssm.*`; [ref] `linear_*` |
| K (attention) | 24 query heads, 2 KV heads, head dim 256, partial RoPE on 64 dims (factor 0.25) | [file] [ref] |
| RoPE | theta 1e7, NeoX halves, mrope sections [11, 11, 10] interleaved | [file] `rope.*`; [ref] `rope_parameters` |
| indexer (K layers) | 4 heads x 128, 1 key head, budget 2048 tokens, compress ratio 4 | [file] [ref] |
| PLE | at layer 1 only (zero-based; `ple_layer_ids [2]` is one-indexed), n-gram 3, 8 heads per n, 16 heads x 160 | [file] `blk.1.ple_*`, `ple.*`; [ref] |
| MTP | the base model has 1 MTP layer; **this GGUF has no MTP tensors** | [ref] `mtp_num_hidden_layers 1`; [file] |

## Weight conventions in this file

- **Norm weights are stored with the +1 already folded in.** Qwen3-Next's RMSNorm is zero-centred (`x * (1 + w)`,
  `w` initialised to zeros) [ref]; llama.cpp's converter adds 1 to every `*norm.weight` except the DeltaNet's gated
  norm (`linear_attn.norm`, which is not zero-centred) [converter], and every RMSNorm weight in the file sits around 1
  (`attn_q_norm` mean 1.28, `hc_attn_norm` 0.94, `ssm_norm` 0.97, `indexer.*_norm` 0.96) [file]. So the forward pass
  multiplies by `w` as stored.
- **`ssm_a` is stored as `A = -exp(A_log)`**: every value is negative (−158.0 to −0.028) [file].
- **`blk.1.ple_norm_key` and `blk.1.ple_norm_query` are identical**, value for value [file]. Either the checkpoint ties
  them or the conversion copied one into the other. [open]
- **Type 42 is upstream ggml's `GGML_TYPE_Q2_0`** (ggml.h, with NVFP4 = 40 and Q1_0 = 41, all newer than the first
  type table here): 64 weights per block, an f16 scale `d` and 16 bytes of 2-bit codes, 18 bytes = 2.25 bits per
  weight, which is exactly what the offsets gave. 9 of the 48 `ffn_down_exps` use it (blk 1, 8, 10, 12, 13, 20, 21,
  30, 36) [file]. Code `q` is `(q - 1) * d` (levels -d, 0, d, 2d), four codes per byte, lowest bits first (upstream
  `dequantize_row_q2_0`). Implemented in `quant.rs`; no independent fixture exists for it (the gguf package lacks
  Q2_0), so it is checked by hand and by its spread against the neighbouring IQ4_NL layer (RMS 0.0115 vs 0.0128).
- **Every other type is checked bit for bit** against ggml's reference dequantizer on real blocks of this file
  (`eval/fixtures/dequant.txt`, 84 blocks, 11 types; `crates/shadow-model/tests/dequant.rs`).
- **The routed experts are mixed precision, per layer** [file]: `ffn_gate_exps` / `ffn_up_exps` are IQ3_XXS in 17
  layers, IQ2_S in 20, IQ3_S in 10, IQ4_XS in 1; `ffn_down_exps` are IQ4_NL in 39 and Q2_0 in 9. So stage 1 needs
  dequant for IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL and Q2_0 on the expert path alone, plus Q4_K, Q5_K, Q6_K, BF16,
  F16 and F32 elsewhere.
- **This file's tensor layout predates upstream's converter.** Upstream llama.cpp now has `conversion/qwen4exp.py`,
  which fuses `attn_qkv`+`attn_gate`, `ssm_beta`+`ssm_alpha` and each `hc_*_down`+`hc_*_inject` into single
  tensors; this file keeps all of them separate. The same converter adds +1 to the PLE and indexer norms ("Gemma
  zero-centred gammas") and takes `ple.norm_key` and `ple.norm_query` from **separate** source tensors.

## The forward pass, one token at a time

### Embedding and the streams
```
x = dequant(token_embd)[:, id]                      # [H]                                    [file]
R = concat(x, x, x, x)                              # [CH]: the residual is C copies of x    [ref]
```

### A hyper-connection read (before every sublayer) — `hc_attn_*` before the mixer, `hc_ffn_*` before the MoE
`Qwen4ExpTextGatedResidual` [ref]; tensors `hc_*_norm [10240]`, `hc_*_down [10240, 320]`, `hc_*_up [320, 10240]`,
`hc_*_inject [10240, 4]` [file].
```
Rn   = rmsnorm_per_stream(R) * hc_norm              # each of the C slices of H normalised on its own, eps 1e-6
mix  = sigmoid( up( silu( down(Rn) / C ) ) )        # [CH]
u    = mean over the C streams of (mix * Rn)        # [H]  -> the sublayer's input
g    = 2 * sigmoid( inject(Rn) / C )                # [C]  per-stream gain in (0, 2)
```
### A hyper-connection write (after the sublayer, output `y [H]`)
```
R[c] = R[c] + g[c] * y        for c in 0..C        # R is the raw stream from before the read
```

### A block
```
if l == 1:  R = R + PLE(R, token history)          # on the full CH stream, before attention       [ref]
u, g = read(hc_attn);  y = mixer_l(u);  write(R, g, y)          # mixer = GatedDeltaNet (J) or attention (K)
u, g = read(hc_ffn);   y = moe_l(u);    write(R, g, y)
```
There is no separate pre-norm inside the sublayers beyond what the read does. [ref; open: confirm no
`input_layernorm` survives in `Qwen4ExpTextDecoderLayer` - the file has no `attn_norm` / `ffn_norm` tensors, which
agrees]

### J: GatedDeltaNet (36 layers)
Tensors: `attn_qkv [2560, 10240]` Q6_K, `attn_gate [2560, 6144]` Q4_K (`z`), `ssm_beta [2560, 48]`, `ssm_alpha [2560, 48]`
BF16, `ssm_conv1d [4, 10240]` F32, `ssm_a [48]`, `ssm_dt.bias [48]`, `ssm_norm [128]`, `ssm_out [6144, 2560]` IQ4_XS [file].
```
qkv  = attn_qkv · u                                 # [10240] = q 2048 | k 2048 | v 6144
z    = attn_gate · u                                # [6144]
b    = ssm_beta · u;  a = ssm_alpha · u             # [48] each (beta from in_proj_b, alpha from in_proj_a)
qkv  = silu( causal_conv1d(qkv, ssm_conv1d, kernel 4, depthwise, state = the previous 3 inputs) )
q, k = l2norm per head (eps 1e-6), 16 heads x 128;  q = q / sqrt(128)
value head h uses key head  h % 16                # NOT h / 3: see "value heads are tiled" below
beta = sigmoid(b);  gdecay = exp( A * softplus(a + dt_bias) )        # A = ssm_a, already negative
for each value head h (state S_h is 128 x 128, k-index by v-index):
    S_h = S_h * gdecay_h
    d   = (v_h - S_h^T k_h) * beta_h                # the delta rule
    S_h = S_h + k_h d^T
    o_h = S_h^T q_h
o    = rmsnorm(o_h, per head 128) * ssm_norm * sigmoid(z_h)          # output_gate_type = sigmoid  [ref config]
y    = ssm_out · o                                  # [H]
```
**The value heads are stored tiled, not grouped** [file, by the converter's convention; found the hard way]. The
reference groups them (`repeat_interleave`: value head `k*3 + v` belongs to key head `k`); llama.cpp's converter
(`_LinearAttentionVReorderBase`) permutes every per-value-head tensor - the v part of `attn_qkv` and of the conv, `z`,
`ssm_beta`, `ssm_alpha`, `ssm_a`, `ssm_dt.bias` and the columns of `ssm_out` - to tiled order (`v*16 + k`), because
ggml's broadcast tiles. So in this file value head `h` pairs with key head `h % 16`. With `h / 3` the model still says
"Paris" after "The capital of France is" and then falls apart within ~30 tokens; that is how it was found.

State per J layer: 48 x 128 x 128 f32 = **3 MiB**, plus the conv state 3 x 10240. Fixed, whatever the context.

### K: gated attention with the indexer (12 layers)
Tensors: `attn_q [2560, 12288]` IQ4_XS, `attn_k [2560, 512]`, `attn_v [2560, 512]` Q6_K, `attn_q_norm [256]`,
`attn_k_norm [256]`, `attn_output [6144, 2560]` IQ4_XS; indexer `q_proj [2560, 512]`, `k_proj [2560, 128]` BF16,
`q_norm [128]`, `k_norm [128]` [file].
```
qg   = attn_q · u  viewed as 24 heads x 512;  q, gate = the two halves of each head   # [ref] view(-1, 2*hd).chunk(2)
k    = attn_k · u (2 heads x 256);  v = attn_v · u (2 heads x 256)
q, k = rmsnorm per head * q_norm / k_norm           # before RoPE
q, k = rope on the first 64 of 256 dims, NeoX halves, theta 1e7, text position p (for text, mrope reduces to this)
attn = softmax( q k^T / sqrt(256) + mask ) v        # GQA 12:1
o    = attn * sigmoid(gate);  y = attn_output · o
```
**The indexer only matters past 2,048 tokens.** It keeps `budget / ratio = 512` blocks of 4 keys, plus the
incomplete tail block, so with 2,048 or fewer visible keys every key is kept and the mask is plain causal [ref]. Stage 1
can therefore be exact without it on prompts under 2,048 tokens. Above that:
```
iq   = indexer.q_proj · u (4 x 128), q_norm, partial rope;  ik = indexer.k_proj · u (128), cached per position
for complete blocks of 4 visible keys: kb = k_norm( mean(ik over the block) ), rope at the block's first position
score(block) = sum over the 4 heads of relu(iq_h · kb) / sqrt(128)
keep the top 512 blocks + the tail; mask everything else
```
[open: whether `k_norm` is a LayerNorm or an RMSNorm here (`k_layernorm` in the reference); the file has a weight but
no bias]

### MoE (48 layers)
Tensors: `ffn_gate_inp [2560, 512]` F32 (router), `ffn_gate_exps` / `ffn_up_exps [2560, 640, 512]`,
`ffn_down_exps [640, 2560, 512]` IQ4_NL or Q2_0 (gate/up: IQ2_S, IQ3_XXS, IQ3_S or IQ4_XS by layer), `ffn_gate_shexp [2560, 640]`, `ffn_up_shexp [2560, 640]`,
`ffn_down_shexp [640, 2560]`, `ffn_gate_inp_shexp [2560]` F32 [file].
```
p    = softmax( ffn_gate_inp · u )                  # over all 512, in f32                       [ref]
top  = the 10 largest;  w = p[top] / sum(p[top])    # norm_topk_prob (the qwen4_exp default)     [ref]
y    = sum_e w_e * down_e( silu(gate_e · u) * (up_e · u) )
     + sigmoid( ffn_gate_inp_shexp · u ) * down_sh( silu(gate_sh · u) * (up_sh · u) )
```

### PLE: the hashed n-gram memory (layer 1 only)
Tensors: `per_layer_token_embd [160, 320001536]` IQ4_NL (26.82 GiB), `blk.1.ple_key [2560, 10240]`,
`ple_value [2560, 2560]` BF16, `ple_norm_key / _query / _conv [10240]` F32, `ple_conv1d [4, 10240]` F16 [file];
`ple.head_vocab_sizes` (16), `ple.head_offsets` (16), `ple.layer_multipliers` (3), `ple.ngram_size 3`,
`ple.heads_per_ngram 8`, `ple.eos_token_id 248044`, `ple.conv_kernel 4` [file].
```
prev1, prev2 = the two previous token ids; EOS (248044) before the start and across an EOS boundary   [ref]
for n in (2, 3), for each of its 8 heads j (head index i = (n-2)*8 + j):
    mixed = XOR over the n tokens ending at this one of ( token_p * multiplier_p )   (int64, wrapping)
    row   = mixed mod head_vocab_sizes[i] + head_offsets[i]
E    = concat over the 16 heads of dequant(per_layer_token_embd)[:, row]      # [16 x 160 = 2560]
key  = ple_norm_key( ple_key · E )      viewed as C x H
qry  = ple_norm_query( R )              viewed as C x H                # R is the full residual stream
val  = ple_value · E                    # [H], shared by the streams
gate = sum over H of (key * qry) / sqrt(H), per stream;  gate = sign(gate) * sqrt(max(|gate|, 1e-6))
gv   = sigmoid(gate) * val              # [C x H] -> [CH]
out  = gv + silu( causal_depthwise_conv1d( ple_norm_conv(gv), ple_conv1d, kernel 4, dilation 3 ) )
```
[open: the exact multiplier per token position and per head (three multipliers in the file, `p` = position in the
n-gram); whether the norms here are per stream or over all CH; the mod of a negative int64 (Python `remainder` is
non-negative)]. The conv state is the last 9 `gv` inputs (dilation 3 x (kernel 4 − 1)).

### The head
```
u      = read(output_hc_norm, output_hc_down, output_hc_up) without the injection   # [H]           [ref]
logits = output · u                                  # no further norm: the file has no output_norm   [file] [ref]
```

## Open questions, and how each gets settled

| # | question | how |
|---|---|---|
| 1 | ~~Q2_0's code-to-value map~~ settled: `(q - 1) * d`, from upstream; a fixture would still be better | a mainline ggml build, or a gguf package that has Q2_0 |
| 2 | ~~norm weights~~ settled: +1 folded in (converter convention, values near 1) | - |
| 3 | ~~`attn_qkv` order, alpha vs beta~~ settled: q, k, v grouped; value heads tiled; beta = b, alpha = a (converter) | - |
| 4 | PLE hashing details (multiplier per position, per head, negative mod) | the reference's `Qwen4ExpTextNGramEmbedding`, read line by line before stage 1 writes it |
| 5 | `ple_norm_key == ple_norm_query` here, though upstream's converter reads two source tensors | the checkpoint: identical there too, or a conversion slip in this file |
| 6 | indexer `k_norm`: LayerNorm or RMSNorm | the reference |
| 7 | MTP for speculative decoding | the base model's MTP layer is in the checkpoint, not in this GGUF; quantize it ourselves, or use prompt-lookup drafting first |

## The fixture, without 360 GB

The oracle is the reference implementation run on the BF16 weights. Stage 1 does not need all of it to start:
safetensors files have a JSON header that says where each tensor is, so one layer's weights can be fetched by HTTP
range requests. The plan for the first fixture: fetch the embedding, layers 0 to 3 (three J layers and the first K,
plus layer 1's PLE rows for the prompt's n-grams) and the head; run the reference on CPU in BF16 for one short prompt;
store each layer's output stream `R` as the fixture. Stage 1's reference must match it within the quantization error
of the IQ3/IQ4 weights, which the same run can measure by also running the reference on our dequantized weights.

That needs Python with torch for one run, on this PC, under the same hard working-set cap as `J:\Strata-work\run_capped.ps1`.
