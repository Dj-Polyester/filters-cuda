import sys
import numpy as np
filterfunc = sys.argv[1]

with open(f"logs/{filterfunc}.log") as logfile:
    imgname = logfile.readline()[:-1]
    xStr = logfile.readline()[:-1]
    yStr = logfile.readline()[:-1]

    print(imgname)
    print(xStr)
    print(yStr)
