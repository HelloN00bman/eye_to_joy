import matplotlib
#matplotlib.use('Agg')
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
import os
import os.path

trials = []
types = []
with open('good_gaze.txt') as f:
	lines = f.readlines()
	for line in lines:
		tmp = line.strip().split('.')
		trials+=[os.path.join(tmp[0].split("'")[-1], tmp[1].split(' ')[0])]
		types+=[tmp[1].split(' ')[1][1]]

path = '/mnt/sdb1/bnewman1/harpdata/cleaned_data'
name = 'gaze_data.csv'
start = 0
end = 0
skip = 1

all_iterations = []
for trial, typ in zip(trials, types):
	if typ = 'a':
		continue
	p = os.path.join(path, trial)
	lines = []
	header = ''
	data = ''
	with open(p) as f:
		reader = csv.reader(f)
		for line in reader:
			lines += [line]
		header = np.array(lines[0])
		data = np.array(lines[1:])
	sz = data.shape[0]
	for i in range(start,len(sz)-end,skip):
		all_iterations+=[path+'-'+str(i)]
