import sys
import numpy as np
# import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

filterfunc = sys.argv[1]

fig = plt.figure()
ax = fig.add_subplot(111)
colors = ["r", "g", "b"]

with open(f"logs/{filterfunc}.log") as logfile:
    times = int(logfile.readline()[:-1])
    for i in range(times):
        imgname = logfile.readline()[:-1]
        xStr = logfile.readline()[:-1]
        yStr = logfile.readline()[:-1]

        x = np.array(list(map(int, xStr.split())))
        y = np.array(list(map(float, yStr.split())))

        ax.plot(x, y, c=colors[i], label=imgname,
                marker="o")
        for i, j in zip(x, y):
            ax.annotate(j, xy=(i, j))
        ax.vlines(x, 0, y, linestyle="dashed")
        plt.xticks(x)


plt.legend()
plt.xlim(0, None)
plt.ylim(0, None)
plt.xlabel('blocksize')
plt.ylabel('time (ms)')
plt.show()
