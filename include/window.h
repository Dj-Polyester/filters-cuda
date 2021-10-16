#if !defined(WINDOW_H)
#define WINDOW_H

#include "global.h"

typedef float winType;

enum WindowType
{
    mean,
    gaussian,
};
#define GAUSSIAN(disti, doublevar) 1 / (sqrt(M_PI * doublevar) * pow(M_E, (disti * disti) / doublevar))
#define GAUSSIAN2D(disti, distj, doublevar) 1 / (M_PI * doublevar * pow(M_E, (disti * disti + distj * distj) / doublevar))

#endif // WINDOW_H
