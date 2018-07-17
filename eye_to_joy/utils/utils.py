import csv
import numpy as np
import torch
import os.path
import pickle

def load_csv(path, header = True):
	header = []
	data = []
	with open(path, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for i, row in enumerate(reader):
			if header and not i:
				header = [row]
			else:
				data += [row]
	header = np.array(header)
	data = np.array(data)
	return header, data

def load_numpy(path, index):
	files = os.listdir(path)
	files = sort_file_list(files, ext='.dat.npy')
	file = files[index]
	return np.load(file)

def load_tensor(path, index):
	files = os.listdir(path)
	files = sort_file_list(files, ext='.pt7')
	file = files[index]
	return torch.load(os.path.join(path, file))

def sort_file_list(ls, ext='.txt'):
	nums = [int(l.split('_')[-1][:-len(ext)]) for l in ls]
	return np.array([file for _,file in sorted(zip(nums, ls))])

def test_train_split(dataset, train_pct, valid=False, **kwargs):
	dset = dataset(**kwargs)
	trainset = dataset(**kwargs)
	testset = dataset(**kwargs)
	validset = dataset(**kwargs)

	total = len(dset)
	inds = np.random.permutation(np.arange(total))
	train_end = int(train_pct * total)
	train_inds = inds[:train_end]
	test_inds = inds[len(train_inds):]
	if valid:
		valid_end = int(valid * len(train_inds))
		valid_inds = train_inds[:valid_end]
		train_inds = train_inds[len(valid_inds):]
		validset.set_inds(valid_inds)
	trainset.set_inds(train_inds)
	testset.set_inds(test_inds)	
	return dset, trainset, validset, testset

def get_trials(filename):
	trials = []
	types = []
	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			tmp = line.strip().split('.')
			trials+=[os.path.join(tmp[0].split("'")[-1], tmp[1].split(' ')[0])]
			types+=[tmp[1].split(' ')[1][1]]
	trials = np.array(trials)
	types = np.array(types)
	return trials, types

def csv_to_tensor_save(good_gaze, data_path, save_path, csv_name, header=True, inds=[0,1,4,5]):
	for trial in good_gaze:
		filename = os.path.join(data_path, trial, csv_name)
		save_dir = os.path.join(save_path, trial)
		os.system('mkdir -p ' + save_dir)

		csv_header = []
		csv_data = []
		with open(filename, 'rb') as f:
			reader = csv.reader(f)
			for i,line in enumerate(reader):
				if i == 0 and header:
					csv_header += [line]
				else:
					csv_data += [line]
			csv_header = np.array(csv_header)
			csv_data = np.array(csv_data)

			for i,data in enumerate(csv_data):
				tensor = torch.Tensor(1, len(inds))
				for j,ind	in enumerate(inds):
					tensor[0, j] = csv_data[i][ind].astype('float64')

				save_name = os.path.join(save_dir, '{:05d}.pt7'.format(i+1))
				torch.save(tensor, save_name)

def load_pickle(fname):
	f = open(fname, 'rb')
	data = pickle.load(f)
	f.close()
	return data

def split_inds(filename, train_pct, valid=False, prefix=None):
	trials, types = get_trials(filename)
	totals = {}

	train_trials = []
	train_types = []

	test_trials = []
	test_types = []

	valid_trials = []
	valid_types = []

	for t in set(types):
		totals[t] = sum(types == t)
		ind = int(np.floor(train_pct*sum(types == t)))
		
		seq_tr = np.random.permutation(trials[types==t])
		train_tr_perm = seq_tr[:ind]
		test_tr_perm = seq_tr[ind:]
		valid_tr_perm = []

		seq_ty = np.random.permutation(types[types==t])
		train_ty_perm = seq_ty[:ind]
		test_ty_perm = seq_ty[ind:]
		valid_ty_perm = []

		if valid:
			vind = int(np.ceil(valid*len(train_tr_perm)))
			
			seq_tr = train_tr_perm
			valid_tr_perm = seq_tr[:vind]
			train_tr_perm = seq_tr[vind:]

			seq_ty = train_ty_perm
			valid_ty_perm = seq_ty[:vind]
			train_ty_perm = seq_ty[vind:]

		train_trials.append(train_tr_perm)
		train_types.append(train_ty_perm)
		valid_trials.append(valid_tr_perm)
		valid_types.append(valid_ty_perm)
		test_trials.append(test_tr_perm)
		test_types.append(test_ty_perm)

	train_trials = np.concatenate(np.array(train_trials))
	train_types = np.concatenate(np.array(train_types))
	seq = np.random.permutation(np.arange(len(train_trials)))
	train_trials = np.array(prefix_to_list(prefix, train_trials[seq]))
	train_types = np.array(prefix_to_list(prefix, train_types[seq]))

	valid_trials = np.concatenate(np.array(valid_trials))
	valid_types = np.concatenate(np.array(valid_types))
	seq = np.random.permutation(np.arange(len(valid_trials)))
	valid_trials = np.array(prefix_to_list(prefix, valid_trials[seq]))
	valid_types = np.array(prefix_to_list(prefix, valid_types[seq]))

	test_trials = np.concatenate(np.array(test_trials))
	test_types = np.concatenate(np.array(test_types))
	seq = np.random.permutation(np.arange(len(test_trials)))
	test_trials = np.array(prefix_to_list(prefix, test_trials[seq]))
	test_types = np.array(prefix_to_list(prefix, test_types[seq]))

	return train_trials, train_types, valid_trials, valid_types, test_trials, test_types

def prefix_to_list(prefix, ls):
	return [os.path.join(prefix,el) for el in ls]

def save_inds(inds, filename):
	pass



