#if !defined(CUDABENCH)
#define CUDABENCH

#include "cudadebug.h"

#define MS(ms) ms
#define US(ms) ms / 1000
#define NS(ms) ms / 1000000

#ifdef LITTLEBENCH
#define INITCUDABENCH()                 \
    cudaEvent_t start, stop;            \
    CUDADBG(cudaEventCreate(&start), ); \
    CUDADBG(cudaEventCreate(&stop), );
#define STARTCUDABENCH() CUDADBG(cudaEventRecord(start), )
#define STOPCUDABENCH() CUDADBG(cudaEventRecord(stop), )
#define PRINTCUDABENCH(time)                           \
    CUDADBG(cudaEventSynchronize(stop), );             \
    float ms = 0;                                      \
    CUDADBG(cudaEventElapsedTime(&ms, start, stop), ); \
    std::cerr << "\033[1;37m" << __FILE__ << ":" << __LINE__ << ": \033[1;36m" #time ":\033[0m " << time(ms) << std::endl
#define PRINTCUDABENCH2(time)                          \
    CUDADBG(cudaEventSynchronize(stop), );             \
    float ms = 0;                                      \
    CUDADBG(cudaEventElapsedTime(&ms, start, stop), ); \
    std::cout << time(ms)

#else

#define INITCUDABENCH()
#define STARTCUDABENCH()
#define STOPCUDABENCH()
#define PRINTCUDABENCH(time)
#define PRINTCUDABENCH2(time)

#endif // LITTLEBENCH
#endif // CUDABENCH
