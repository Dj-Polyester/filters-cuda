#if !defined(CUDADEBUG)
#define CUDADEBUG

#include "debug.h"

#ifdef LITTLEBUG
#define INITCUDADBG() cudaError_t cudaErr
#define CUDACHECK()                                                                          \
    {                                                                                        \
        cudaError_t errSync = cudaGetLastError();                                            \
        cudaError_t errAsync = cudaDeviceSynchronize();                                      \
        if (errSync != cudaSuccess)                                                          \
            std::cerr << "Sync kernel error:" << cudaGetErrorString(errSync) << std::endl;   \
        if (errAsync != cudaSuccess)                                                         \
            std::cerr << "Async kernel error:" << cudaGetErrorString(errAsync) << std::endl; \
    }
#define CUDADBG(cudaerr, cleanup)                        \
    {                                                    \
        cudaErr = cudaerr;                               \
        if (cudaErr != cudaSuccess)                      \
            ERROR(cudaGetErrorString(cudaErr), cleanup); \
    }
#else
#define INITCUDADBG()
#define CUDACHECK()
#define CUDADBG(cudaErr, cleanup) cudaErr
#endif //LITTLEBUG
#endif // CUDADEBUG
