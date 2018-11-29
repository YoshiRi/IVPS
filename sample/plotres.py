import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

pos3d = np.loadtxt("pos3d.txt")


fig = plt.figure()
xdata, ydata = [], []

ims = []

for p in pos3d:
    xdata.append(p[0])
    ydata.append(p[1])
    line = plt.plot(xdata,ydata,'r-',p[0],p[1],'rx')
    ims.append(line)

ani = ArtistAnimation(fig, ims, interval=20, blit=True)
ani.save()