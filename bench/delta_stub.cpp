// forward.cu declares `extern "C" __device__ const int* g_rope_tab` for the engine's module loader; a standalone nvcc build of it wants a host symbol of that name to link
extern "C" const int* g_rope_tab = nullptr;
