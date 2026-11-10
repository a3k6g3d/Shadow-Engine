@echo off
rem Builds Strata 0.1.44's fused int8 MoE kernels (MIT, J:\Strata-work\Strata-0144) beside a harness that feeds them Shadow-shaped experts: usage build.bat [harness.cu] [out.exe]
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d J:\Shadow\bench\stratafused
set S=J:/Strata-work/Strata-0144
set G=J:/Strata-work/build-0145/_deps/strata_llamacpp-src/ggml
set H=%1
if "%H%"=="" set H=strata_fused_bench.cu
set O=%2
if "%O%"=="" set O=strata_fused_bench.exe
nvcc -O3 -arch=sm_120 -std=c++17 --expt-relaxed-constexpr -DGGML_CUDA_NO_VMM -I%S%/include -I%S%/src/prefill -I%G%/include -I%G%/src -I%G%/src/ggml-cuda -o %O% %H% %S%/src/prefill/moe_fused.cu %S%/src/prefill/moe_fused_iq.cu 2>&1 | findstr /i "error fatal warning.*C4"
if exist %O% (echo built %O%) else (echo BUILD FAILED)
