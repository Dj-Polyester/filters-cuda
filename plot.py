import random
import sys

import numpy as np
# import tkinter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

matplotlib.use('TkAgg')


def annotatePlot(x, y):
    anns = []
    for i, j in zip(x, y):
        anns.append(ax.annotate(j, xy=(i, j)))
    return anns


def rmAnnotations(anns):
    for ann in anns:
        ann.remove()


filterfuncs = sys.argv[1:]

fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(111)

plots = []
labels = []
visibility = []

xmax = float("-inf")
ymax = float("-inf")
totalimgs = 0
for filterfunc in filterfuncs:
    with open(f"logs/{filterfunc}.log") as logfile:
        times = int(logfile.readline()[:-1])
        for time in range(times):
            imgfiles = int(logfile.readline()[:-1])
            totalimgs += imgfiles
            for i in range(imgfiles):
                imgname = logfile.readline()[:-1]
                xStr = logfile.readline()[:-1]
                yStr = logfile.readline()[:-1]

                x = np.array(list(map(int, xStr.split())))
                y = np.array(list(map(float, yStr.split())))

                xmaxTmp = x.max()
                ymaxTmp = y.max()

                if xmaxTmp > xmax:
                    xmax = xmaxTmp
                if ymaxTmp > ymax:
                    ymax = ymaxTmp

                lbl = f"{filterfunc}_{imgname.split('.')[0]}_{time}"

                pl = ax.plot(x, y, c=(random.random(), random.random(), random.random()), label=lbl,
                             marker="o")

                dic = {
                    "plot": pl,
                    "annotations": annotatePlot(
                        x, y),
                    "x": x,
                    "y": y, }

                plots.append(dic)
                labels.append(lbl)
                visibility.append(True)

                ax.vlines(x, 0, ymax, linestyle="dashed")
                plt.xticks(x)

# Make checkbuttons with all plotted lines with correct visibility
rax = plt.axes([0.01, 0.02, 0.105, 0.02*totalimgs])
check = CheckButtons(rax, labels, visibility)


def func(label):
    index = labels.index(label)
    dic = plots[index]
    lines = dic["plot"]
    vis = None
    for line in lines:
        vis = not line.get_visible()
        line.set_visible(vis)
    if vis:
        dic["annotations"] = annotatePlot(dic["x"], dic["y"])
    else:
        rmAnnotations(dic["annotations"])

    plt.draw()


check.on_clicked(func)


plt.legend()
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.xlabel('blocksize')
plt.ylabel('time (ms)')
plt.show()
