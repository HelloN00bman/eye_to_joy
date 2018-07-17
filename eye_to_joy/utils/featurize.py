import featurizer

import numpy as np 

import os
import os.path 

import skimage
import skimage.io as skio
import skimage.transform as tfm

import sys

import time

import torch
import torch.autograd as ag
import torch.multiprocessing as mp

from exceptions import StopIteration

def is_locked(path):
	return os.path.isfile(path+'.lock') or \
			os.path.isdir(path+'.lock') or \
			path.split('.')[-1] == 'lock'

def lock(path):
	if not is_locked(path):
		os.system('mkdir -p ' + path + '.lock')

def unlock(path):
	if is_locked(path):
		os.system('rm -r ' + path + '.lock')

def get_frames(path):
	print(mp.current_process().name + ' is getting frames at ' + path )
	frames = dict()
	contents = os.listdir(path)
	for i in range(len(contents)):
		if not i % 50:
			print(mp.current_process().name + ' is ' + '{:.2f}'.format((i*1.0)/len(contents) * 100) + '% done')
		content = os.path.join(path, contents[i])
		frames[i] = torch.Tensor(tfm.resize(skio.imread(content), (3,224,224)))
	return frames

def get_batches(group, bsize=1):
	keys = group.keys()
	for i in range(0, len(keys), bsize):
		batch = torch.Tensor(np.zeros([bsize,3,224,224]))
		for j in range(bsize):
			if i+j < len(keys):
				key = keys[i+j]
				batch[j] = group[key]
			else:
				print(mp.current_process().name, i, j, i+j)
		yield batch

def save_batch_features(path, name, features):
	torch.save(features, os.path.join(path, name))

def traverse(l, path1, path2):
	f = featurizer.Featurizer('vgg13')
	worker = mp.current_process()
	print('Traversing with ' + worker.name)
	participants = os.listdir(path1)
	#participants = ['p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24']
	for participant in participants:
		part_path = os.path.join(path1, participant)
		trials = os.listdir(part_path)
		for trial in trials:
			trial_path = os.path.join(part_path, trial)
			l.acquire()
			if is_locked(trial_path):
				print(trial_path + ' was locked. Moving on.')
				l.release()
				continue
			else:
				print('Locking ' + trial_path)
				lock(trial_path)
				l.release()
				feat_path = os.path.join(path2, participant, trial)
				print('Working on ' + trial_path + ' with ' + worker.name)
				os.system('mkdir -p ' + feat_path)
				frames = get_frames(trial_path)
				batches = get_batches(frames)
				num = 1
				while True:
					try:
						batch = batches.next()
					except StopIteration:
						break
					print(worker.name + ' is doing a forward pass')
					t1 = time.time()
					batch_features = f.forward(ag.Variable(batch, volatile=True))
					save_batch_features(feat_path, '{:05d}.t7'.format(num), batch_features)
					print(worker.name + ' {:.2f}'.format(time.time()-t1) + ' seconds to complete forward pass')
					num+=1
				#unlock(trial_path)

def main():
	PROCESSES = int(sys.argv[1])
	print('Starting with ' + str(PROCESSES) + ' processes.')
	l = mp.Lock()
	path = '/file2/home/bnewman1/harplabstuff/harpdata/egoframes' #'/media/ben/HARPLab-2T1/eyegaze_videos'
	feature_path = '/file2/home/bnewman1/harplabstuff/harpdata/egofeatures'
	print('Creating Workers ............................')
	for i in range(PROCESSES):
		mp.Process(target=traverse, args=(l, path, feature_path,)).start()

if __name__ == '__main__':
	main()
