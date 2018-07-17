import torch
import torch.nn as nn
import torchvision.models as models
from SpatSemLSTM import *
from torch.autograd import Variable
from visualize import make_dot
import torch.optim as optim
import gc


HIDDEN_DIM = 256
num_epochs = 5
num_seqs = 4
seq_len = 10

model = SpatSemLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs0 = torch.randn(num_seqs, seq_len, 4,224,224)
inputs1 = torch.randn(num_seqs, seq_len, 2,12)
inputs2 = torch.randn(num_seqs, seq_len, 2,24)
inputs3 = torch.randn(num_seqs, seq_len, 2,36)
lbls = torch.randn(num_seqs, seq_len, 2)

# EPOCHS
tensor_dicts = [dict() for _ in range(num_epochs+1)]
for k in range(num_epochs):
	count = 0
	tensor_dict = tensor_dicts[k]
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				count+=1
				if str(obj.size()) in tensor_dict.keys():
					tensor_dict[str(obj.size())] += 1
				else:
					tensor_dict[str(obj.size())] = 1
			elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
				count+=1
				if str(obj.size())  not in tensor_dict.keys():
					tensor_dict[str(obj.size())] += 1
				else:
					tensor_dict[str(obj.size())] = 1
			else:
				continue
		except Exception as e:
			pass
	print('num tensors:', count)

	# THROUGH EVERY SEQUENCE
	for i in range(num_seqs):
		loss = 0
		optimizer.zero_grad()

		sem_out = Variable(torch.randn(1,1, HIDDEN_DIM))
		sem_hidden = (Variable(torch.randn(1, 1, HIDDEN_DIM).float()), Variable(torch.randn(1, 1, HIDDEN_DIM).float()))
		spat_hidden = (Variable(torch.randn(1, 1, HIDDEN_DIM).float()), Variable(torch.randn(1, 1, HIDDEN_DIM).float()))

		# THROUGH THE SEQUENCE
		for j in range(seq_len):
			input0 = Variable(inputs0[i, j:j+1, :, :, :])
			input1 = Variable(inputs1[i, j:j+1, :, :])
			input2 = Variable(inputs2[i, j:j+1, :, :])
			input3 = Variable(inputs3[i, j:j+1, :, :])
			lbl = Variable(lbls[i, j, :])
			out, sem_out, sem_hidden, spat_hidden = model(j, input0, input1, input2, input3, sem_out, sem_hidden, spat_hidden)
			
			#dot = make_dot(out)
			#dot.render('test_gvs/SpatSemLSTM_viz_'+str(i)+'_'+str(j)+'.gv', view=False)

			loss += criterion(out, lbl)

		print('backward')
		loss.backward()
		print('optimize')
		optimizer.step()

gc.collect()
count=0		
for obj in gc.get_objects():
	try:
		if torch.is_tensor(obj):
			count+=1
			if str(obj.size()) in tensor_dict.keys():
				tensor_dict[str(obj.size())] += 1
			else:
				tensor_dict[str(obj.size())] = 1
		elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
			count+=1
			if str(obj.size())  not in tensor_dict.keys():
				tensor_dict[str(obj.size())] += 1
			else:
				tensor_dict[str(obj.size())] = 1
		else:
			continue
	except Exception as e:
		pass
print('num tensors:', count)
import IPython as ip; ip.embed()