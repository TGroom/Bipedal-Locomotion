import matplotlib.pyplot as plt
import numpy as np


# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = []

for i in x:
    y += [1 + np.sin(i)*0.1 + np.sin(i*2)*0.1 + np.cos(i*3)*0.15]

xc = y*np.cos(x)
yc = y*np.sin(x)

xcirc = np.cos(x)
ycirc = np.sin(x)

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(xc, yc)
#ax.plot(xcirc, ycirc)
ax1.plot([-1.3, 1.3], [0.1, -0.1])
ax1.plot([-0.1, 0.1], [-1.3, 1.3])

ax1.set_title('PCA on Unaltered Data')

ax2.plot(x, y)
ax2.plot([0,2 * np.pi], [1,1])
ax2.plot([np.pi, np.pi], [0.8, 1.2])
ax2.set_title('PCA on Polar Coordinates')
ax2.set_ylim([0, 1.5])

plt.show()

fig, ax1 = plt.subplots(1,1, subplot_kw=dict(projection='polar'))
ax1.plot(x, y)

