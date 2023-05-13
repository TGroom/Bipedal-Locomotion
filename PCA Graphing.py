import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn import decomposition
from sklearn.preprocessing import scale
import pandas as pd
import mpl_toolkits.mplot3d
from scipy.signal import savgol_filter
from numpy import genfromtxt
import os

np.random.seed(5)

def printPCA(pca):
    concat = ""
    for i in range(6):
        concat += "%.3f" % pca[i]
        if i < 5:
            concat += ", "
    return concat


def average(data, cycle, num):
    dataAveraged = data[0 : cycle].copy()  # Initialise dataAveraged with the first 'cycle' of values
    for i in range(cycle):
        for j in range(num):
            dataAveraged[i] += data[i + cycle * j]
        dataAveraged[i] = dataAveraged[i] / num  # Calculate the average
    return dataAveraged

""" PCA Stair Traversal """

# Input file path may need to be changed depending on where this file is run from
input_file = "Desktop/Bipedal-Locomotion/DataForPCA/StairTraversalSteppingFINAL.csv"
input_file = os.path.abspath(input_file)

my_data = genfromtxt(input_file, delimiter = ',')
rawX = my_data[500:]

# Pre-process the data
X = scale(rawX)
Xtemp = []
for i in range(6):
    # Smoothing data using a window size 50 and polynomial order 3
    Xtemp.append(savgol_filter([j[i] for j in X], 50, 3))

X = np.array(Xtemp).T

fig = plt.figure("PC1, PC2 and PC3 for Stair Traversal", figsize=plt.figaspect(0.5))
ax1 = fig.add_subplot(1, 2, 1, projection = "3d", elev = 48, azim = 134)

pca = decomposition.PCA(n_components = 6)
pca.fit(X)
X = pca.transform(X)
loadings = pd.DataFrame(pca.components_[0:3].T, columns = ['PC1', 'PC2', 'PC3'])
explainedVariance = np.multiply(pca.explained_variance_ratio_, 100)

print("\n=========== Stair Traversal PCA Results ===========")
print("PCA1 = [" + printPCA(loadings.PC1) + "]" + '\n'
    + "PCA2 = [" + printPCA(loadings.PC2) + "]" + '\n'
    + "PCA3 = [" + printPCA(loadings.PC3) + "]" + '\n'
    + "Explained Variance = [" + printPCA(explainedVariance) + "]")

ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(np.arange(1, len(explainedVariance) + 1), explainedVariance, color = 'grey')
ax2.plot(np.arange(1, len(explainedVariance) + 1), np.cumsum(explainedVariance), c = 'black')
ax2.scatter(np.arange(1, len(explainedVariance) + 1), np.cumsum(explainedVariance), c = 'black')
ax2.set_xlabel("PC")
ax2.set_ylabel("Variance Explained (%)")
ax2.plot([0, len(explainedVariance)], [np.cumsum(explainedVariance)[2], np.cumsum(explainedVariance)[2]], '--', c = 'grey')
ax2.set_xlim([0, 7])
ax2.text(-0.8, np.cumsum(explainedVariance)[2] - 2, int(np.cumsum(explainedVariance)[2]))
ax2.set_xticks([1, 2, 3, 4, 5, 6])
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

c, c2 = [], []

X = np.concatenate((X, rawX[:, 6:10]), axis = 1)

AvRange = 200
lenToPlot = AvRange  # len(X[:, 1])
XAveraged = average(X, AvRange, 22)
XAveraged1 = average(X[0:(1800-500)], AvRange, 6)
XAveraged2 = average(X[(1800-500):(2600-500)], AvRange, 4)
XAveraged3 = average(X[(3450-500):(3850-500)], AvRange, 2)

c3, cTemp = [], []
for i in range(4500):
    # Walking on Ground
    if (i < (1800 - 500)):  
        cTemp.append(colors.hsv_to_rgb([0.05, 1, 1]))

    # Stepping up
    elif (i < (2600 - 500)): 
        cTemp.append(colors.hsv_to_rgb([0.3, 1, 1]))

    # Steppping down
    elif (i > (3450 - 500) and i < (3850 - 500)):  
        cTemp.append(colors.hsv_to_rgb([0.6, 1, 1]))

    # Walking on Ground
    elif ((3850 - 500) < i):
        cTemp.append(colors.hsv_to_rgb([0.1, 1, 1]))

    # Walking on landing
    else:  
        cTemp.append(colors.hsv_to_rgb([0.1, 1, 1]))

for i in range(3):
    # window size 100, polynomial order 2
    c.append(savgol_filter([j[i] for j in cTemp], 100, 2))

c = np.clip(np.array(c).T, 0, 1)

for i in range(lenToPlot):
    ax1.plot(XAveraged1[i:i + 2, 0], XAveraged1[i:i + 2, 1], XAveraged1[i:i + 2, 2],
             color = colors.hsv_to_rgb([0.05, 1, 1]), linewidth = '2', alpha = 1)
    ax1.plot(XAveraged2[i:i + 2, 0], XAveraged2[i:i + 2, 1], XAveraged2[i:i + 2, 2],
             color = colors.hsv_to_rgb([0.3, 1, 1]), linewidth = '2', alpha = 1)
    ax1.plot(XAveraged3[i:i + 2, 0], XAveraged3[i:i + 2, 1], XAveraged3[i:i + 2, 2],
             color = colors.hsv_to_rgb([0.6, 1, 1]), linewidth = '2', alpha = 1)

fig.tight_layout()
fig.show()

fig2 = plt.figure("PC3 Against PC2 for Stair Traversal")
ax3 = fig2.add_subplot(1, 1, 1)
ax3.plot(X[:, 1], X[:, 2], color = [0, 0, 0], linewidth = '1', alpha = 0.2)

for i in range(lenToPlot):
    ax3.plot(XAveraged1[i:i + 2, 1], XAveraged1[i:i + 2, 2],
             color = colors.hsv_to_rgb([0.05, 1, 1]), linewidth = '2', alpha = 1)
    ax3.plot(XAveraged2[i:i + 2, 1], XAveraged2[i:i + 2, 2],
             color = colors.hsv_to_rgb([0.3, 1, 1]), linewidth = '2', alpha = 1)
    ax3.plot(XAveraged3[i:i + 2, 1], XAveraged3[i:i + 2, 2],
             color = colors.hsv_to_rgb([0.6, 1, 1]), linewidth = '2', alpha = 1)

ax3.set_xlabel('PC2')
ax3.set_ylabel('PC3')

""" PCA Walking """

input_file = "Desktop/Bipedal-Locomotion/DataForPCA/WalkingFINAL.csv"
input_file = os.path.abspath(input_file)

my_data = genfromtxt(input_file, delimiter = ',')
rawX = my_data[500:]

# Pre-process the data
Xtemp = []
for i in range(6):
    # Smoothing data using a window size 50 and polynomial order 3
    Xtemp.append(savgol_filter([j[i] for j in rawX], 50, 3))

X = np.array(Xtemp).T

fig3 = plt.figure("PC1, PC2 and PC3 for Walking", figsize = plt.figaspect(0.5))
ax4 = fig3.add_subplot(1, 2, 1, projection = "3d", elev = 48, azim = 134)

pca = decomposition.PCA(n_components = 6)
pca.fit(X)
X = pca.transform(X)
loadings = pd.DataFrame(pca.components_[0:3].T, columns = ['PC1', 'PC2', 'PC3'])
explainedVariance = np.multiply(pca.explained_variance_ratio_, 100)

print("\n=========== Walking PCA Results ===========")
print("PCA1 = [" + printPCA(loadings.PC1) + "]" + '\n'
    + "PCA2 = [" + printPCA(loadings.PC2) + "]" + '\n'
    + "PCA3 = [" + printPCA(loadings.PC3) + "]" + '\n'
    + "Explained Variance = [" + printPCA(explainedVariance) + "]")

ax5 = fig3.add_subplot(1, 2, 2)
ax5.bar(np.arange(1, len(explainedVariance) + 1), explainedVariance, color = 'grey')  #, label=bar_labels, color=bar_colors)
ax5.plot(np.arange(1, len(explainedVariance) + 1), np.cumsum(explainedVariance), c = 'black')
ax5.scatter(np.arange(1, len(explainedVariance) + 1), np.cumsum(explainedVariance), c = 'black')
ax5.set_xlabel("PC")
ax5.set_ylabel("Variance Explained (%)")
ax5.plot([0, len(explainedVariance)], [np.cumsum(explainedVariance)[2], np.cumsum(explainedVariance)[2]], '--', c = 'grey')
ax5.set_xlim([0, 7])
ax5.text(-0.8, np.cumsum(explainedVariance)[2] - 2, int(np.cumsum(explainedVariance)[2]))
ax5.set_xticks([1, 2, 3, 4, 5, 6])
ax4.set_xlabel('PC1')
ax4.set_ylabel('PC2')
ax4.set_zlabel('PC3')

c, c2 = [], []
X = np.concatenate((X, rawX[:, 6:10]), axis = 1)

AvRange = 200
lenToPlot = AvRange  # len(X[:, 1])
XAveraged = average(X, AvRange, 22)
XAveraged1 = average(X[0:(1800 - 500)], AvRange, 6)
XAveraged2 = average(X[(1800 - 500):(2600 - 500)], AvRange, 4)
XAveraged3 = average(X[(3450 - 500):(3850 - 500)], AvRange, 2)

c3, cTemp = [], []
for i in range(lenToPlot):
    dx = XAveraged[i, 0] - XAveraged[i-1, 0]
    dy = XAveraged[i, 1] - XAveraged[i-1, 1]
    angle = np.arctan(dy / dx)
    c2.append(colors.hsv_to_rgb([(angle + (np.pi / 2)) / np.pi, 1, 0.9]))

    left = np.multiply(XAveraged[i, 6], [0, 0.5, 1])
    right = np.multiply(XAveraged[i, 7], [1, 0.3, 0])
    diff = abs(XAveraged[i, 6] + XAveraged[i, 7] - 1)
    both = np.multiply(diff, [1, 0, 1])
    c1 = np.clip(np.abs(np.subtract(np.add(left, right), both)), 0, 1)
    c.append(c1)  # colors.hsv_to_rgb([(angle+(np.pi/2))/np.pi, 1, 0.9]))

for i in range(lenToPlot):
    ax4.plot(XAveraged[i:i + 2, 0], XAveraged[i:i + 2, 1], XAveraged[i:i + 2, 2], color = c[i], linewidth = '2', alpha = 1)
 
fig3.tight_layout()
fig3.show()
fig4 = plt.figure("PC3 Against PC2 for Walking")
ax6 = fig4.add_subplot(1, 1, 1)
ax6.plot(X[:, 1], X[:, 2], color = [0, 0, 0], linewidth = '1', alpha = 0.2)

for i in range(lenToPlot):
    ax6.plot(XAveraged[i:i + 2, 1], XAveraged[i:i + 2, 2], color = c[i], linewidth = '2', alpha = 1)

ax6.set_xlabel('PC2')
ax6.set_ylabel('PC3')
plt.show()


"""
def cartesian_to_hyperspherical(xs):
    x1, x2, x3, x4, x5, x6 = xs
    r = math.sqrt(x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2)

    for i in range(6):
        for j in range(6-i):
    
    if x[k] >= 0:
        theta[k] = 0
    else:
        theta[k] = math.pi
    
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
X = scale(my_data[:])
"""

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
