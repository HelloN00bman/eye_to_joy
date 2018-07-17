import matplotlib
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
import os
import os.path


ls = []
with open('traj_info.csv') as f:
	reader = csv.reader(f)
	for i in reader:
		ls+=[i]
header = ls[0]
data = np.array(ls[1:]).astype('float64')

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 1

inp_x = data[:,2]
inp_y = data[:,3]
xmax = np.max(inp_x)
ymax = np.max(inp_y)
ymin = np.min(inp_y)
xmin = np.min(inp_x)
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([inp_x,inp_y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
ax.contour(X,Y,F,[0])
plt.show()
