@echo off
rem usage: build.bat KERNEL_FILE TYPE ENTRY CH OUT      e.g. build.bat matvec_new.cu IQ3XXS mma_grouped_iq3_xxs 32 mma_iq3xxs_new.exe
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d J:\Shadow\bench\mmabench
nvcc -O3 -arch=sm_120 -std=c++17 -I..\..\kernels -I. -DKERNEL_FILE=\"%1\" -DQT=%2 -DENTRY=%3 -DENTRY_NAME=\"%3\" -DCH=%4 -o %5 ..\mma_bench.cu 2>&1 | findstr /i "error fatal"
echo built %5
