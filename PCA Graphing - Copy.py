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



# PCA

input_file = "C:/Users/Thomas Groom/Desktop/Blind-Bipedal-Locomotion/PCALog/WalkingFinal.csv"
#input_file = "C:/Users/Thomas Groom/Desktop/Blind-Bipedal-Locomotion/PCALog/StairTraversalWalking.csv"
#input_file = "C:/Users/Thomas Groom/Desktop/Blind-Bipedal-Locomotion/PCALog/StairTraversalSteppingNew.csv"
Xt = []
my_data = genfromtxt(input_file, delimiter=',')
rawX = my_data[500:]

import math

def cartesian_to_hyperspherical(xs):
    x1, x2, x3, x4, x5, x6 = xs
    r = math.sqrt(x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2)

    """for i in range(6):
        for j in range(6-i):
    
    if x[k] >= 0:
        theta[k] = 0
    else:
        theta[k] = math.pi
    """
    theta1 = math.acos(x1 / r)
    theta2 = math.acos(x2 / math.sqrt(x6**2 + x5**2 + x4**2 + x3**2 + x2**2))
    theta3 = math.acos(x3 / math.sqrt(x6**2 + x5**2 + x4**2 + x3**2))
    theta4 = math.acos(x4 / math.sqrt(x6**2 + x5**2 + x4**2))
    theta5 = math.acos(x5 / math.sqrt(x6**2 + x5**2))
    if(x6 < 0):
        theta5 = (2*math.pi) - theta5
    
    return [r, theta1, theta2, theta3, theta4, theta5]

def hyperspherical_to_cartesian(polar):
    r, theta1, theta2, theta3, theta4, theta5 = polar
    x1 = r * math.cos(theta1)
    x2 = r * math.sin(theta1) * math.cos(theta2)
    x3 = r * math.sin(theta1) * math.sin(theta2) * math.cos(theta3)
    x4 = r * math.sin(theta1) * math.sin(theta2) * math.sin(theta3) * math.cos(theta4)
    x5 = r * math.sin(theta1) * math.sin(theta2) * math.sin(theta3) * math.sin(theta4) * math.cos(theta5)
    x6 = r * math.sin(theta1) * math.sin(theta2) * math.sin(theta3) * math.sin(theta4) * math.sin(theta5)
    return [x1,x2,x3,x4,x5,x6]




polarData = []
for x in rawX:
    polarData.append(cartesian_to_hyperspherical(x[0:6]))

X = (polarData)
#X = scale(my_data[:])

print(X[:][1])
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

print("PCA1 = [" + printPCA((loadings.PC1))+"]")
print("PCA2 = [" + printPCA((loadings.PC2))+"]")
print("PCA3 = [" + printPCA((loadings.PC3))+"]")


import sys
sys.exit()
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
#XAveraged1 = average(X[0:(1800-500)], AvRange, 6)
#XAveraged2 = average(X[(1800-500):(2600-500)], AvRange, 4)
#XAveraged3 = average(X[(3450-500):(3850-500)], AvRange, 2)

cTemp = []
c3 = []
for i in range(lenToPlot):
    
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
        """

#for i in range(3):
#    c.append(savgol_filter([j[i] for j in cTemp], 100, 2)) # window size 51, polynomial order 3

#c = np.array(c)
#c = c.T
#c = np.clip(c, 0, 1)
#ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c = c)#, cmap=plt.cm.nipy_spectral, edgecolor="k")
#ax1.plot(X[:, 0], X[:, 1], X[:, 2], color=[0,0,0], linewidth='1', alpha=0.2)#, color=plt.cm.jet(255*i/len(X[:,0])))

for i in range(lenToPlot):
    #c = colors.hsv_to_rgb([X[i-1, 8]-0.5,1,1])
    ax1.plot(XAveraged[i:i+2, 0], XAveraged[i:i+2, 1], XAveraged[i:i+2, 2], color=c[i], linewidth='2', alpha=1)
   # ax1.plot(XAveraged1[i:i+2, 0], XAveraged1[i:i+2, 1], XAveraged1[i:i+2, 2], color=colors.hsv_to_rgb([0.05, 1, 1]), linewidth='2', alpha=1)
   # ax1.plot(XAveraged2[i:i+2, 0], XAveraged2[i:i+2, 1], XAveraged2[i:i+2, 2], color=colors.hsv_to_rgb([0.3, 1, 1]), linewidth='2', alpha=1)
   # ax1.plot(XAveraged3[i:i+2, 0], XAveraged3[i:i+2, 1], XAveraged3[i:i+2, 2], color=colors.hsv_to_rgb([0.6, 1, 1]), linewidth='2', alpha=1)
    
fig.tight_layout()

fig.show()

fig2 = plt.figure(2)
ax3 = fig2.add_subplot(1, 1, 1)
ax3.plot(X[:, 1], X[:, 2], color=[0,0,0], linewidth='1', alpha=0.2)

#for i in range(lenToPlot):
    #if(i < 50):
    #    color = [1,0,0]
    #elif(i < 100):
    #    color = [0,1,0]
    #else:
    #    color = [0,0,1]
    #ax3.plot(XAveraged1[i:i+2, 1], XAveraged1[i:i+2, 2], color=colors.hsv_to_rgb([0.05, 1, 1]), linewidth='2', alpha=1)
    #ax3.plot(XAveraged2[i:i+2, 1], XAveraged2[i:i+2, 2], color=colors.hsv_to_rgb([0.3, 1, 1]), linewidth='2', alpha=1)
    #ax3.plot(XAveraged3[i:i+2, 1], XAveraged3[i:i+2, 2], color=colors.hsv_to_rgb([0.6, 1, 1]), linewidth='2', alpha=1)

ax3.set_xlabel('PC2')
ax3.set_ylabel('PC3')
#plt.xlim(-4,4)
#plt.ylim(-3,3)


plt.show()






