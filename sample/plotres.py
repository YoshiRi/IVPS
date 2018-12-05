import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.animation import FuncAnimation

pos3d = np.loadtxt("data/pos3d.txt")
anime = 0

fig, ax = plt.subplots()
line, = ax.plot([], [], 'r',label='trajectory')
line2, = ax.plot([], [], 'kx',label='current position')
ax.grid()
xdata, ydata = [], []
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid ='on' 



#def data_gen():
#    pos3d = np.loadtxt("pos3d.txt")
#    for pos in pos3d:
#        yield pos


def init():
    ax.set_ylim(-0.2, 1.2)
    ax.set_xlim(-1.2, 0.2)
    line.set_data(xdata, ydata)
    #line2.set_data(xdata, ydata)
    return line,

def run(i):
    # update the data
    x, y = pos3d[8*i]
    if not i:
        xt, yt = pos3d[8*i]
        xdata.append(xt)
        ydata.append(yt)

    else:
        for it in range(8*i-7,8*i+1):
            xt, yt = pos3d[it]
            xdata.append(xt)
            ydata.append(yt)
    

    line.set_data(xdata, ydata)
    line2.set_data(x,y)
    plt.legend(loc='upper left')



if anime:
    fig = plt.figure()
    xdata, ydata = [], []
    plt.grid = 'on'
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['tracking route','current position'],loc='upper left')

    ims = []

    ss = 0

    for p in pos3d:
        xdata.append(p[0])
        ydata.append(p[1])
        line = plt.plot(xdata,ydata,'r-',p[0],p[1],'kx')
        if ss > 10:
            ims.append(line)
            ss = 0
        ss = ss + 1
        #plt.clf()

    ani = ArtistAnimation(fig, ims, interval=30, blit=True)
    ani.save("plot.gif",writer='imagemagick')
    plt.show()

else:

    ani = FuncAnimation(fig, run,  blit=False, interval=10, frames = int(len(pos3d)/8)-1,
                            repeat=False, init_func=init)

    ani.save("plot2.gif",writer='imagemagick')
    plt.show()