@echo off
rem Builds shadow_fused.dll (Strata's fused int8 MoE kernels behind a C interface; see shadow_fused.cu).  Needs nvcc and MSVC (VS 18 Community here).
rem GGML_DIR: a llama.cpp checkout's ggml folder (MIT), for ggml.h / ggml-common.h.  Default: the copy Strata fetched on this PC (as native/mmq/build.bat).
if "%GGML_DIR%"=="" set GGML_DIR=J:/Strata-work/build-0145/_deps/strata_llamacpp-src/ggml
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
set G=%GGML_DIR%
nvcc -shared -arch=sm_120 -std=c++17 -O3 --expt-relaxed-constexpr -DGGML_CUDA_NO_VMM -Iinclude -I%G%/include -I%G%/src -I%G%/src/ggml-cuda -o shadow_fused.dll shadow_fused.cu src/moe_fused.cu src/moe_fused_iq.cu 2>&1 | findstr /i "error LNK fatal"
if exist shadow_fused.dll (echo built shadow_fused.dll) else (echo BUILD FAILED)
