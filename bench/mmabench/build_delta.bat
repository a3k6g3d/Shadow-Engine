@echo off
rem usage: build_delta.bat OUT      the DeltaNet kernels' standalone check and timing (bench/delta_bench.cu; delta_stub.cpp gives forward.cu's g_rope_tab a host symbol to link)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d J:\Shadow\bench\mmabench
nvcc -O3 -arch=sm_120 -std=c++17 -I..\..\kernels -I. -o %1 ..\delta_bench.cu ..\delta_stub.cpp 2>&1 | findstr /i "error fatal"
echo built %1
