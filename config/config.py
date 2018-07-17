import numpy as np
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
trials = np.array(trials)
types = np.array(types)

non_auto = np.where(types != 'a')[0]
types = types[non_auto]
trials = trials[non_auto]

data_dict = dict()

data_dict['VID_NAME'] = 'world_{:05d}.png'
data_dict['FEAT_NAME'] = '{:05d}.png'
data_dict['JOY_NAME'] = 'traj_info.csv'
data_dict['GAZE_NAME'] = 'gaze_positions.csv'

data_dict['DATA_PATH'] = '/mnt/sdb1/bnewman1/harpdata'
data_dict['VID_PATH'] = os.path.join(data_dict['DATA_PATH'], 'eyegaze_videos')
data_dict['FEAT_PATH'] = os.path.join(data_dict['DATA_PATH'], 'eyegaze_features')
data_dict['JOY_PATH'] = os.path.join(data_dict['DATA_PATH'], 'cleaned_data')
data_dict['GAZE_PATH'] = os.path.join(data_dict['DATA_PATH'], 'cleaned_data')

data_dict['TRIALS'] = trials
data_dict['VID_TRIALS'] = np.array([os.path.join(data_dict['VID_PATH'], trial) for trial in trials])
data_dict['FEAT_TRIALS'] = np.array([os.path.join(data_dict['FEAT_PATH'], trial) for trial in trials])
data_dict['JOY_TRIALS'] = np.array([os.path.join(data_dict['JOY_PATH'], trial) for trial in trials])
data_dict['GAZE_TRIALS'] = np.array([os.path.join(data_dict['GAZE_PATH'], trial) for trial in trials])

data_dict['WINDOWS'] = [12, 24, 36]
data_dict['WINDOW_SIZE'] = max(data_dict['WINDOWS'])
data_dict['BATCH_SIZE'] = 25
data_dict['IM_HEIGHT'] = 224
data_dict['IM_WIDTH'] = 224

