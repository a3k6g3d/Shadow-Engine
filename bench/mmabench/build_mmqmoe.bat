@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d J:\Shadow\bench\mmabench
nvcc -O3 -arch=sm_120 -std=c++17 -o mmq_moe_bench.exe ..\mmq_moe_bench.cu 2>&1 | findstr /i "error fatal"
echo built
