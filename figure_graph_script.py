import matplotlib.pyplot as plt
import scipy.stats as norms
import numpy as np
from cycler import cycler

Ymean = .20 * 255
Yneighbors = [255 * x for x in [0.10, 0.90, 0.75, 0.50]]
stdev = 100
nStdev = 4
xTick = 51

x1 = np.arange(-nStdev * stdev + Ymean, -1, 1)
x2 = np.arange(0, 255, 1)
x3 = np.arange(256, stdev * nStdev + Ymean, 1)
xf = np.arange(-nStdev * stdev + Ymean, stdev * nStdev + Ymean, 1)

# color = plt.cm.viridis(np.linspace(0, 1, x2.shape[0]))
color = plt.cm.Greys(np.linspace(1, 0, x2.shape[0]))
custom_cycler = (cycler(color=color))

fig, ax = plt.subplots()

ax.plot(xf, norms.norm.pdf(xf, Ymean, stdev), linewidth=15, zorder=-1)

for i, j in zip(x1, norms.norm.pdf(x1, Ymean, stdev)):
    ax.scatter(i, j, color="black")

ax.set_prop_cycle(custom_cycler)
for i, j in zip(x2, norms.norm.pdf(x2, Ymean, stdev)):
    ax.scatter(i, j)

ax.set_prop_cycle(None)
for i, j in zip(x3, norms.norm.pdf(x3, Ymean, stdev)):
    ax.scatter(i, j, color="white")


plt.xlim((-2.2 * xTick, 6.2 * xTick))
plt.xticks(np.arange(0 , 6 * xTick, xTick))

ax.set_yticklabels([])
plt.axvline(Ymean,color="black", lw=2, linestyle='--')
plt.axvline(Yneighbors[0], color="green", lw=4)
plt.axvline(Yneighbors[1], color="red", lw=4)
plt.axvline(Yneighbors[2], color="blue", lw=4)
plt.axvline(Yneighbors[3], color="purple", lw=4)

plt.show()