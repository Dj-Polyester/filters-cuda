import pathlib
import random
import sys

import numpy as np
# import tkinter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import os
VISIBILITY = False


def annotatePlot(x, y):
    anns = []
    for i, j in zip(x, y):
        anns.append(ax.annotate(j, xy=(i, j)))
    return anns


def rmAnnotations(anns):
    for ann in anns:
        ann.remove()


def setVisibility(dic, vis):
    lines = dic["plot"]
    for line in lines:
        line.set_visible(vis)
    if vis:
        dic["annotations"] = annotatePlot(dic["x"], dic["y"])
    elif "annotations" in dic.keys():
        rmAnnotations(dic["annotations"])


def setCheckbox(index, vis): visibility[index] = vis


def func(label):
    index = labels.index(label)
    dic = plots[index]
    lines = dic["plot"]
    line = lines[0]
    vis = not line.get_visible()
    setVisibility(dic, vis)
    setCheckbox(index, vis)
    plt.draw()


matplotlib.use('TkAgg')

logfilepaths = sys.argv[1:]

fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(111)

plots = []
labels = []
visibility = []

xmax = float("-inf")
ymax = float("-inf")
totalimgs = 0
for logfilepath in logfilepaths:
    with open(logfilepath) as logfile:
        filterfunc = os.path.basename(logfilepath)
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
                    "x": x,
                    "y": y, }

                plots.append(dic)
                labels.append(lbl)
                visibility.append(VISIBILITY)

                setVisibility(dic, VISIBILITY)

                ax.vlines(x, 0, ymax, linestyle="dashed")
                plt.xticks(x)

# Make checkbuttons with all plotted lines with correct visibility
rax = plt.axes([0.01, 0.02, 0.105, 0.02*totalimgs])
check = CheckButtons(rax, labels, visibility)

check.on_clicked(func)


plt.legend()
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.xlabel('blocksize')
plt.ylabel('time (ms)')
plt.show()
