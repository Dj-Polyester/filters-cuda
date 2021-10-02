#if !defined(DEBUG_H)
#define DEBUG_H

#ifdef LITTLEBUG
#define SINGLE(var) var

#define DEBUG(label, var, debug) std::cerr << "\033[1;37m" << __FILE__ << ":" << __LINE__ << ": \033[1;36m" #label ":\033[0m " << debug(var) << std::endl
#define DEBUG2(var, debug) DEBUG(var, var, debug)
#define DEBUG3(var, debug) DEBUG(debug, var, debug)
#define DEBUG4(label, var) DEBUG(label, var, SINGLE)
#define DEBUG5(var) DEBUG(var, var, SINGLE)

#else
#define SINGLE(var)

#define DEBUG(label, var, debug)
#define DEBUG2(var, debug)
#define DEBUG3(var, debug)
#define DEBUG4(label, var)
#define DEBUG5(var)
#endif

#define ERROR(str, cleanup)                                                                                           \
    {                                                                                                                 \
        std::cerr << "\033[1;37m" << __FILE__ << ":" << __LINE__ << ": \033[1;31merror:\033[0m " << str << std::endl; \
        cleanup;                                                                                                      \
        exit(1);                                                                                                      \
    }

#define PRINT(str) std::cerr << str << std::endl

#endif // DEBUG_H
