@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
for %%P in (llama ik) do (
  echo === %%P
  cmake -S J:\Shadow\target-src-%%P -B J:\Shadow\target-build-%%P -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe" ^
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON || echo CONFIGURE FAILED %%P
  cmake --build J:\Shadow\target-build-%%P --target llama-server -j 14 && echo BUILD OK %%P || echo BUILD FAILED %%P
)
