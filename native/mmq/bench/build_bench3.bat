@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set G=J:/Strata-work/build-0145/_deps/strata_llamacpp-src/ggml
cd /d J:\Shadow\bench\mmq
set C=-arch=sm_120 -std=c++17 -O2 --expt-relaxed-constexpr -DGGML_CUDA_NO_VMM -I%G%/include -I%G%/src -I%G%/src/ggml-cuda
nvcc %C% -DBENCH_TYPE=GGML_TYPE_Q6_K -DBENCH_QK=256 -DBENCH_BLOCK=block_q6_K -DBENCH_NAME=\"Q6_K\" -DBENCH_IS_IQ4NL=0 -o mmq_q6k.exe mmq_bench.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/template-instances/mmq-instance-q6_k.cu 2>&1 | findstr /i "error LNK fatal"
nvcc %C% -DBENCH_TYPE=GGML_TYPE_IQ4_XS -DBENCH_QK=256 -DBENCH_BLOCK=block_iq4_xs -DBENCH_NAME=\"IQ4_XS\" -DBENCH_IS_IQ4NL=0 -o mmq_iq4xs.exe mmq_bench.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu 2>&1 | findstr /i "error LNK fatal"
echo built
