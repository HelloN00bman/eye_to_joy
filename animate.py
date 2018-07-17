import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)

csvdata = []
with open('test_results.csv', 'r') as csvfile:
   csvreader = csv.reader(csvfile)
   for row in csvreader:
      csvdata.append(row)

csvdata = np.array(csvdata)
inp = csvdata[:, 1]
total, inds = np.unique(inp, return_index=True)
inds = sorted(inds)
ordered_names = inp[inds]
ordered_sizes = csvdata[:,2][inds]
ordered_types = csvdata[:,3][inds]



for i in range(len(total)):
   print(ordered_names[i].replace('/','_')[1:]+'.mp4')
   
   fig = plt.figure()

   l1, = plt.plot([], [], 'k-o')
   l2, = plt.plot([], [], 'y-o')

   plt.gca().set_aspect('equal', 'box')
   plt.xlim(0, 1)
   plt.ylim(0, 1)

   x0, y0, x1, y1 = 0, 0, 0, 0
   plt.title(ordered_names[i]+' '+ordered_types[i])
   with writer.saving(fig, ordered_names[i].replace('/','_')[1:]+'.mp4', 100):
       ann = None
       for j,row in zip(range(int(ordered_sizes[i])), csvdata[:,4:]):
           if ann != None:
             ann.remove()
           x0 = float(row[0])
           y0 = float(row[1])
           x1 = float(row[2])
           y1 = float(row[3])
           dist = np.sqrt((x1-x0)**2 + (y1-y0)**2)
           if x1 >= 1 or x1 <=0 or y1>=1 or y1<=0:
            l1.set_data(-1, -1)
            l2.set_data(-1, -1)
            dist = 'NaN'
           else:
            l1.set_data(x0, y0)
            l2.set_data(x1, y1)
           ann = l1.axes.annotate(dist, xy=(0,0), xytext=(0,0))
           writer.grab_frame()
       plt.close()