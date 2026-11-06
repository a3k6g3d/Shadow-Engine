@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set G=J:/Strata-work/build-0145/_deps/strata_llamacpp-src/ggml
cd /d J:\Shadow\bench\mmq
set C=-arch=sm_120 -std=c++17 -O2 --expt-relaxed-constexpr -DGGML_CUDA_NO_VMM -I%G%/include -I%G%/src -I%G%/src/ggml-cuda
nvcc %C% -DBENCH_TYPE=GGML_TYPE_IQ4_NL -DBENCH_QK=32 -DBENCH_BLOCK=block_iq4_nl -DBENCH_NAME=\"IQ4_NL\" -DBENCH_IS_IQ4NL=1 -o mmq_iq4nl.exe mmq_bench.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu 2>&1 | findstr /i "error LNK fatal"
nvcc %C% -DBENCH_TYPE=GGML_TYPE_IQ2_S -DBENCH_QK=256 -DBENCH_BLOCK=block_iq2_s -DBENCH_NAME=\"IQ2_S\" -DBENCH_IS_IQ4NL=0 -o mmq_iq2s.exe mmq_bench.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/template-instances/mmq-instance-iq2_s.cu 2>&1 | findstr /i "error LNK fatal"
nvcc %C% -DBENCH_TYPE=GGML_TYPE_IQ3_XXS -DBENCH_QK=256 -DBENCH_BLOCK=block_iq3_xxs -DBENCH_NAME=\"IQ3_XXS\" -DBENCH_IS_IQ4NL=0 -o mmq_iq3xxs.exe mmq_bench.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu 2>&1 | findstr /i "error LNK fatal"
nvcc %C% -DBENCH_TYPE=GGML_TYPE_IQ3_S -DBENCH_QK=256 -DBENCH_BLOCK=block_iq3_s -DBENCH_NAME=\"IQ3_S\" -DBENCH_IS_IQ4NL=0 -o mmq_iq3s.exe mmq_bench.cu %G%/src/ggml-cuda/quantize.cu %G%/src/ggml-cuda/template-instances/mmq-instance-iq3_s.cu 2>&1 | findstr /i "error LNK fatal"
echo built
