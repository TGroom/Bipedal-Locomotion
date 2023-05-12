# Code source: Gaël Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.colors as colors


from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import scale

import pandas as pd

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

np.random.seed(5)

import numpy as np
from scipy.signal import savgol_filter

from numpy import genfromtxt

"""
# Spectorgram Of Neuron Activation

import cv2
import numpy as np
img = np.zeros((500,500,3), np.uint8)
scalex = 10
scaley = 1

input_file = "C:/Users/Thomas Groom/Desktop/Blind-Bipedal-Locomotion/NeuronLog/test.csv"

activationData = genfromtxt(input_file, delimiter=',')
"""
"""
for y in range(len(activationData[0])):
    for x in range(len(activationData)):
        for i in range(scalex):
            for j in range(scaley):
                img[x*scalex+i][y*scaley+j] = [(1 - activationData[x][y])*255, (1 - activationData[x][y])*255, (1 - activationData[x][y])*255]#[, 0, (1 - activationData[y][x])*255]

#img = cv2.cvtColor(hsvimage, cv2.COLOR_HSV2RGB)
img_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)

img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV).astype("float32")
(h, s, v) = cv2.split(img_color)
s = np.clip(s*0.8,0,255)
img_color = cv2.merge([h,s,v])
img_color = cv2.cvtColor(img_color.astype("uint8"), cv2.COLOR_HSV2BGR)

cv2.imshow('image',img_color)]
cv2.waitKey(0)

"""
"""
plt.imshow(np.transpose(activationData), extent=[0,4.2,0,48000], cmap='jet',
           vmin=0, vmax=1, origin='lower', aspect='auto')
plt.colorbar()
plt.show()

exit()
"""


# PCA

input_file = "C:/Users/Thomas Groom/Desktop/Blind-Bipedal-Locomotion/PCALog/WalkingFinal.csv"
#input_file = "C:/Users/Thomas Groom/Desktop/Blind-Bipedal-Locomotion/PCALog/StairTraversalWalking.csv"
#input_file = "C:/Users/Thomas Groom/Desktop/Blind-Bipedal-Locomotion/PCALog/StairTraversalSteppingNew.csv"
Xt = []

my_data = genfromtxt(input_file, delimiter=',')
rawX = my_data[500:]


X = scale(rawX)
#X = scale(my_data[:])

print(rawX[:][1])
for i in range(6):
    Xt.append(savgol_filter([j[i] for j in X], 50, 3)) # window size 51, polynomial order 3

Xt = np.array(Xt)
X = Xt.T



fig = plt.figure(1, figsize=plt.figaspect(0.5))
plt.clf()

ax1 = fig.add_subplot(1, 2, 1, projection="3d", elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=6)
pca.fit(X)
X = pca.transform(X)

loadings = pd.DataFrame(pca.components_[0:3].T, columns=['PC1', 'PC2', 'PC3'])

def printPCA(pca):
    concat = ""
    for i in range(6):
        concat += str(pca[i]) + ", "
        #concat += "%.2f" % pca[i] + ", "
    return concat

print("PCA1 = [" + printPCA(loadings.PC1)+"]")
print("PCA2 = [" + printPCA(loadings.PC2)+"]")
print("PCA3 = [" + printPCA(loadings.PC3)+"]")


explainedVariance = np.multiply(pca.explained_variance_ratio_, 100)

print(explainedVariance)

ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(np.arange(1, len(explainedVariance)+1), explainedVariance, color='grey')  #, label=bar_labels, color=bar_colors)
ax2.plot(np.arange(1, len(explainedVariance)+1), np.cumsum(explainedVariance), c='black')
ax2.scatter(np.arange(1, len(explainedVariance)+1), np.cumsum(explainedVariance), c='black')
ax2.set_xlabel("PC")
ax2.set_ylabel("Variance Explained (%)")
ax2.plot([0, len(explainedVariance)], [np.cumsum(explainedVariance)[2], np.cumsum(explainedVariance)[2]], '--', c='grey')
ax2.set_xlim([0, 7])
ax2.text(-0.8, np.cumsum(explainedVariance)[2] - 2, int(np.cumsum(explainedVariance)[2]))
ax2.set_xticks([1,2,3,4,5,6])

ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

c = []
c2 = []

def average(data, cycle, num):
    dataAveraged = data[0:cycle].copy()
    for i in range(cycle):
        for j in range(num):
            dataAveraged[i] += data[i + cycle*j]
        dataAveraged[i] = dataAveraged[i] / num
    return dataAveraged

X = np.concatenate((X, rawX[:,6:10]), axis=1)

AvRange = 200
lenToPlot = AvRange#len(X[:, 1]) #200
XAveraged = average(X, AvRange, 22)
XAveraged1 = average(X[0:(1800-500)], AvRange, 6)
XAveraged2 = average(X[(1800-500):(2600-500)], AvRange, 4)
XAveraged3 = average(X[(3450-500):(3850-500)], AvRange, 2)

cTemp = []
c3 = []
for i in range(4500):
    """
    dx = XAveraged[i, 0] - XAveraged[i-1, 0]
    dy = XAveraged[i, 1] - XAveraged[i-1, 1]
    angle = np.arctan(dy/dx)
    c2.append(colors.hsv_to_rgb([(angle+(np.pi/2))/np.pi, 1, 0.9]))

    left = np.multiply(XAveraged[i, 6], [0,0.5,1])
    right = np.multiply(XAveraged[i, 7], [1,0.3,0])
    diff = abs(XAveraged[i, 6] + XAveraged[i, 7] - 1)
    both = np.multiply(diff, [1,0,1])
    c1 = np.clip(np.abs(np.subtract(np.add(left, right), both)), 0, 1)

    c.append(c1)#colors.hsv_to_rgb([(angle+(np.pi/2))/np.pi, 1, 0.9]))
        
    """
    if(i<(1800-500)): #walking ground
        cTemp.append(colors.hsv_to_rgb([0.05, 1, 1]))
    elif(i<(2600-500)):#step up
        cTemp.append(colors.hsv_to_rgb([0.3, 1, 1]))
    elif(i > (3450-500) and i < (3850-500)): #step down
        cTemp.append(colors.hsv_to_rgb([0.6, 1, 1]))
    elif((3850-500)<i):#walking ground
        cTemp.append(colors.hsv_to_rgb([0.1, 1, 1]))
    else:#walking landing
        cTemp.append(colors.hsv_to_rgb([0.1, 1, 1]))

for i in range(3):
    c.append(savgol_filter([j[i] for j in cTemp], 100, 2)) # window size 51, polynomial order 3

c = np.array(c)
c = c.T
c = np.clip(c, 0, 1)
#ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c = c)#, cmap=plt.cm.nipy_spectral, edgecolor="k")
#ax1.plot(X[:, 0], X[:, 1], X[:, 2], color=[0,0,0], linewidth='1', alpha=0.2)#, color=plt.cm.jet(255*i/len(X[:,0])))

for i in range(lenToPlot):
    #c = colors.hsv_to_rgb([X[i-1, 8]-0.5,1,1])
    ax1.plot(XAveraged1[i:i+2, 0], XAveraged1[i:i+2, 1], XAveraged1[i:i+2, 2], color=colors.hsv_to_rgb([0.05, 1, 1]), linewidth='2', alpha=1)
    ax1.plot(XAveraged2[i:i+2, 0], XAveraged2[i:i+2, 1], XAveraged2[i:i+2, 2], color=colors.hsv_to_rgb([0.3, 1, 1]), linewidth='2', alpha=1)
    ax1.plot(XAveraged3[i:i+2, 0], XAveraged3[i:i+2, 1], XAveraged3[i:i+2, 2], color=colors.hsv_to_rgb([0.6, 1, 1]), linewidth='2', alpha=1)
    
fig.tight_layout()

fig.show()

fig2 = plt.figure(2)
ax3 = fig2.add_subplot(1, 1, 1)
ax3.plot(X[:, 1], X[:, 2], color=[0,0,0], linewidth='1', alpha=0.2)

for i in range(lenToPlot):
    #if(i < 50):
    #    color = [1,0,0]
    #elif(i < 100):
    #    color = [0,1,0]
    #else:
    #    color = [0,0,1]
    ax3.plot(XAveraged1[i:i+2, 1], XAveraged1[i:i+2, 2], color=colors.hsv_to_rgb([0.05, 1, 1]), linewidth='2', alpha=1)
    ax3.plot(XAveraged2[i:i+2, 1], XAveraged2[i:i+2, 2], color=colors.hsv_to_rgb([0.3, 1, 1]), linewidth='2', alpha=1)
    ax3.plot(XAveraged3[i:i+2, 1], XAveraged3[i:i+2, 2], color=colors.hsv_to_rgb([0.6, 1, 1]), linewidth='2', alpha=1)

ax3.set_xlabel('PC2')
ax3.set_ylabel('PC3')
#plt.xlim(-4,4)
#plt.ylim(-3,3)


plt.show()






