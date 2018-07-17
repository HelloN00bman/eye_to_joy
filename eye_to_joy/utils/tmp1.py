import numpy as np
import matplotlib.pyplot as plt
import csv as reader

ls = []
with open('traj_info.csv') as f:
	with csv.reader(f) as r: 
		for i in reader:
			ls+=[i]

header = ls[0,:]
data = np.array(ls[1:,:]).astype('float64')
inp = [data[2,:], data[3,:]]
x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
X, Y = np.meshgrid(x,y)
F = X**2 + Y**2 - 1

	
