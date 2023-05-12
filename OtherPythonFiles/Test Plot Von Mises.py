import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, iv
from scipy.stats import vonmises

"""
Plots a cumulative Von Mises distribution
"""

Amu = (0/100)*np.pi
Bmu = (75/100)*np.pi
kappa = 50

A = np.random.vonmises(Amu, kappa, size=None)
B = np.random.vonmises(Bmu, kappa, size=None)
a = []

for o in range(200):
    i = []
    for sample in range(1000):
        currentTimeinPhase = ((o % 100)/100)*np.pi

        A = np.random.vonmises(Amu, kappa, size=None)
        B = np.random.vonmises(Bmu, kappa, size=None)

        if currentTimeinPhase > A and currentTimeinPhase < B:
            i.append(1)
        else:
            i.append(0)
    a.append(np.average(i))

plt.figure()
plt.hist(np.random.vonmises(Amu, kappa, size=10000),
         100, range=[-np.pi, np.pi])

num = 100
accuracy = 200  # Increase accuracy by increasing this number
Sxs = np.linspace(-np.pi, np.pi, num=accuracy)
Ays = []
Bys = []
for x in Sxs:
    Ays.append(np.exp(kappa*np.cos(x-Amu))/(2*np.pi*i0(kappa)))

for x in Sxs:
    Bys.append(np.exp(kappa*np.cos(x-Bmu))/(2*np.pi*i0(kappa)))


def CDFVonMeses(x, flag):
    sample = int(((x+np.pi)/(2*np.pi))*accuracy)

    if flag:
        return np.trapz(Ays[0:sample], Sxs[0:sample])
    else:
        return np.trapz(Bys[0:sample], Sxs[0:sample])


def probIndecator(o, x):
    PAB = CDFVonMeses(o, 1) * (1 - CDFVonMeses(o, 0))

    if x == 1:
        return PAB
    else:
        return 1 - PAB


xs = np.linspace(0, 10, num=num)
inds = []
for o in xs:
    inds.append(probIndecator(o % np.pi, 0))

plt.figure()
plt.plot(xs, inds, linewidth=2, color='r')
plt.show()
