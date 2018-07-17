import os
import os.path
import sys


import ffmpeg
import numpy as np 

def split(inp, out, name, rate=30):
	print('mkdir -p ' + out)
	os.system('mkdir -p ' + out)
	print('ffmpeg -i ' + inp + ' -r ' + str(rate) + ' ' + os.path.join(out, name))
	os.system('ffmpeg -i ' + inp + ' -r ' + str(rate) + ' ' + os.path.join(out, name))
	print('mogrify -verbose -resize 224x224! ' + out + '/*.png')
	os.system('mogrify -verbose -resize 224x224! ' + out + '/*.png')

def crawl(inp, out):
	print(inp)
	print(out)
	part_dirs = os.listdir(inp)
	for part in part_dirs:
		part_path = os.path.join(inp, part)
		print('part: ', part)
		if os.path.isdir(part_path):
			part_out = os.path.join(out, part)
			trial_dirs = os.listdir(part_path)
			for trial in trial_dirs:
				print('trial: ', trial)
				trial_path = os.path.join(part_path, trial, 'world.mp4')
				trial_out = os.path.join(part_out, trial)
				split(trial_path, trial_out, 'world_%05d.png')



inp = '/media/ben/HARPLab-2T/eyegaze_data_ada_eating_study/cleaned_data'
out = '/media/ben/HARPLab-2T/eyegaze_videos'


crawl(inp, out)