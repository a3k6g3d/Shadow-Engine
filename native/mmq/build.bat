@echo off
rem Builds shadowzero_mmq.dll (llama.cpp's MMQ kernels behind a C interface; see shadowzero_mmq.cu).  Needs nvcc and MSVC (VS 18 Community here).
rem GGML_DIR: a llama.cpp checkout's ggml folder (MIT).  Default: the copy Strata fetched on this PC.
if "%GGML_DIR%"=="" set GGML_DIR=J:/Strata-work/build-0145/_deps/strata_llamacpp-src/ggml
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
set G=%GGML_DIR%
set T=%G%/src/ggml-cuda/template-instances
nvcc -shared -arch=sm_120 -std=c++17 -O2 --expt-relaxed-constexpr -DGGML_CUDA_NO_VMM -I%G%/include -I%G%/src -I%G%/src/ggml-cuda -o shadowzero_mmq.dll shadowzero_mmq.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/mmvq.cu %T%/mmq-instance-q4_k.cu %T%/mmq-instance-q5_k.cu %T%/mmq-instance-q6_k.cu %T%/mmq-instance-iq4_xs.cu 2>&1 | findstr /i "error LNK fatal"
if exist shadowzero_mmq.dll (echo built shadowzero_mmq.dll) else (echo BUILD FAILED)
