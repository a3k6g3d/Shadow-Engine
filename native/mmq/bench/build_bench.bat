@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set G=J:/Strata-work/build-0145/_deps/strata_llamacpp-src/ggml
cd /d J:\Shadow\bench\mmq
nvcc -arch=sm_120 -std=c++17 -O2 --expt-relaxed-constexpr -DGGML_CUDA_NO_VMM -I%G%/include -I%G%/src -I%G%/src/ggml-cuda -o mmq_bench.exe mmq_bench.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu 2>&1
