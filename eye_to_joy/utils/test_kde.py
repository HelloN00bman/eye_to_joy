import matplotlib
#matplotlib.use('Agg')
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
import os
import os.path

path = '/mnt/sdb1/bnewman1/harpdata/cleaned_data/'
save_path = '/mnt/sdb1/bnewman1/kdes/'

trials = []
types = []
with open('good_gaze.txt') as f:
	lines = f.readlines()
	for line in lines:
		tmp = line.strip().split('.')
		trials+=[os.path.join(tmp[0].split("'")[-1], tmp[1].split(' ')[0])]
		types+=[tmp[1].split(' ')[1][1]]


x_circ = np.linspace(-1,1,100)
y_circ = np.linspace(-1,1,100)
X_circ, Y_circ = np.meshgrid(x_circ,y_circ)
F = X_circ**2 + Y_circ**2 - 1

failures_0 = []
failures_other = []
all_x = []
all_y = []
for trial in trials:
	print('mkdir -p ' + save_path + trial.split('/')[0])
	os.system('mkdir -p ' + save_path + '/' + trial.split('/')[0])
	print(trial)
	ls = []
	with open(os.path.join(path, trial, 'traj_info.csv')) as f:
		reader = csv.reader(f)
		for i in reader:
			ls+=[i]
	header = ls[0]
	data = np.array(ls[1:]).astype('float64')
	print(data.shape)
	if data.shape[0] == 0:
		failures_0 = []
		continue

	inp_x = data[:,2]
	inp_y = data[:,3]
	xmax = np.max(inp_x)
	ymax = np.max(inp_y)
	ymin = np.min(inp_y)
	xmin = np.min(inp_x)
	#X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	X, Y = np.mgrid[-1:1:100j, -1:1:100j]
	positions = np.vstack([X.ravel(), Y.ravel()])
	values = np.vstack([inp_x,inp_y])
	try:
		kernel = stats.gaussian_kde(values)
		all_x = np.concatenate((all_x,inp_x),0)
		all_y = np.concatenate((all_y,inp_y),0)
	except:
		failures_other += [trial]
		continue
	Z = np.reshape(kernel(positions).T, X.shape)
	fig, ax = plt.subplots()
	ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[-1,1,-1,1])
	#ax.contour(X_circ,Y_circ,F,[0])
	ax.plot(inp_x, inp_y)
	plt.savefig(os.path.join(save_path, trial+'.png'))
	plt.close()

xmax = np.max(all_x)
ymax = np.max(all_y)
ymin = np.min(all_y)
xmin = np.min(all_x)
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([all_x,all_y])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin,xmax,ymin,ymax])
ax.contour(X,Y,F,[0])
plt.savefig(os.path.join(save_path, 'all.png'))
plt.close()

print(failures_0)
print(failures_other)
