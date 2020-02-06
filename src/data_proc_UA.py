'''
Feature extraction for the our 3D skeleton datasets
Read .mat file, extract feature and create feature array and label array, save to .pkl

Author: Yi Wei

'''

import os
import numpy as np
from tqdm import tqdm
import sys
from scipy.io import loadmat
from six.moves import cPickle as pickle

# Definitions of Kinect SDK joint order and names
HIP_CENTER = 0
SPINE = 1
SHOULDER_CENTER = 2
HEAD = 3
SHOULDER_LEFT = 4
ELBOW_LEFT = 5
WRIST_LEFT = 6
HAND_LEFT = 7
SHOULDER_RIGHT = 8
ELBOW_RIGHT = 9
WRIST_RIGHT = 10
HAND_RIGHT = 11
HIP_LEFT = 12
KNEE_LEFT = 13
ANKLE_LEFT = 14
FOOT_LEFT = 15
HIP_RIGHT = 16
KNEE_RIGHT = 17
ANKLE_RIGHT = 18
FOOT_RIGHT = 19

NUI_SKELETON_POSITION_COUNT = 20


nui_skeleton_names = {
	'HIP_CENTER', 'SPINE', 'SHOULDER_CENTER', 'HEAD',
	'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT', 'HAND_LEFT',
	'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT', 'HAND_RIGHT',
	'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT',
	'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT', 'FOOT_RIGHT' }

# skeleton connection matrix
nui_skeleton_conn = [
	[HIP_CENTER, SPINE],
	[SHOULDER_CENTER, SPINE],
	[HEAD, SHOULDER_CENTER],
	# Left arm
	[SHOULDER_LEFT, SHOULDER_CENTER],
	[ELBOW_LEFT, SHOULDER_LEFT],
	[WRIST_LEFT, ELBOW_LEFT],
	[HAND_LEFT, WRIST_LEFT],
	# Right arm
	[SHOULDER_RIGHT, SHOULDER_CENTER],
	[ELBOW_RIGHT, SHOULDER_RIGHT],
	[WRIST_RIGHT, ELBOW_RIGHT],
	[HAND_RIGHT, WRIST_RIGHT],
	# Left leg
	[HIP_LEFT, HIP_CENTER],
	[KNEE_LEFT, HIP_LEFT],
	[ANKLE_LEFT, KNEE_LEFT],
	#[ANKLE_LEFT, FOOT_LEFT],
	# Right leg
	[HIP_RIGHT, HIP_CENTER],
	[KNEE_RIGHT, HIP_RIGHT],
	[ANKLE_RIGHT, KNEE_RIGHT],
	#[ANKLE_RIGHT, FOOT_RIGHT]
]

# label classes
label_classes = [
		'Drink','Eat','Read_book','Write_on_paper','Write_on_blackboard','look_at_monitor',
		'Use_remote_controller','Play_read_phone','Make_a_call','Pick_up_phone','Turn_on_monitor',
		'Type_on_keyboard','Fetch_water','Pour_water','Throw_trash','Pick_up_trash','tear_up_paper',
		'wear_jacket','take_off_jacket','wear_shoes','take_off_shoes','wear_on_glasses','take_off_glasses',
		'touch_hair','Clapping','Wave_hand','nod_head','shake_head','wipe_face','Bend_down','Walk','sit',
		'stand_up','jump','cross_leg']


def save_dict(di_, filename_):
	with open(filename_, 'wb') as f:
		pickle.dump(di_, f)

def load_dict(filename_):
	with open(filename_, 'rb') as f:
		ret_di = pickle.load(f, encoding='latin1')
	return ret_di


def label_loader(file, frm_num):
	'''
	Reading label file, parsing the labels for a sequence to a binary matrix.

	input:
		file - label file contains 35 class with corresponding frame gaps
		frm_num_list - list contains all frame numbers of this sequence
	:return:
		[#frame, #class] numpy array. 1 represents action with label class occurs at current frame.
	'''
	label_mat = np.zeros((frm_num, len(label_classes)))
	with open(file, 'r') as fin:
		for line in fin:
			line = line.strip()
			if not line:
				# empty line
				continue
			if line in label_classes:
				# class index
				class_idx = label_classes.index(line)
			else:
				# assign label matrix for current class with (begin end) frame# pair
				start_idx, end_idx = line.split(' ')
				#print(start_idx, end_idx)
				try:
					label_mat[int(start_idx)-1:int(end_idx), class_idx] = 1
				except Exception as e:
					print('Errors in sequence file ', file)
					print(str(e))
				#for i in range(int(start_idx), int(end_idx)+1):
				#    try:
				#        label_mat[frm_num_list.index('%07d' % i)][class_idx] = 1
				#    except Exception as e:
				#        print('Errors in sequence file ', file)
				#        print(str(e))
	return label_mat

def data_loader(skelfile):
	'''
	Loading skeleton data of one sequence from .mat file

	Input:
		skelfile: skeleton .mat file, contains all frames skeleton data

	:return:
		[#frame,  60] Numpy array. 60 for 20 joints 3D coordinates
		Frame number of the sequence.
	'''
	skelstct = loadmat(skelfile)
	skels_3dpos = skelstct['body']['Position'] # get the 3d joints position
	_, nf = skels_3dpos.shape
	seq = np.zeros((nf, 60))     # 3D point array for sequence
	for i in range(nf):
		joints = skels_3dpos[:,i][0] # get a 3*25 joints position array at frame i
		seq[i] = np.reshape(joints[:,:20].T, [1, 60])
	return seq, nf

def normalize_skel(seq):
	'''
	Normalize skeleton joints with length of arm and legs.

	Input:
		seq - [#frame, 60] Numpy array along time sequence. Each frame 20*3 joints.
	Output:
		norm_seq - [#frame, 60] Normalized joints positions along time.
	'''
	length, dim = seq.shape
	norm_seq = np.zeros((length, dim))
	for idx in range(length):
		skel = np.reshape(seq[idx,:], [NUI_SKELETON_POSITION_COUNT, 3])
		center = skel[SPINE,:] # position of SPINE
		# get the length of torsos, get the maximum length for normalization
		spine_length = np.linalg.norm(skel[SPINE,:] - skel[SHOULDER_CENTER,:]) + \
			np.linalg.norm(skel[SHOULDER_CENTER,:] - skel[HEAD,:])
		#leftarm_length = np.linalg.norm(skel[SHOULDER_CENTER,:] - skel[SHOULDER_LEFT,:]) + np.linalg.norm(skel[SHOULDER_LEFT,:] - skel[ELBOW_LEFT,:]) + \
		#	np.linalg.norm(skel[ELBOW_LEFT,:] - skel[WRIST_LEFT,:]) + np.linalg.norm(skel[WRIST_LEFT,:] - skel[HAND_LEFT, :])
		#rightarm_length = np.linalg.norm(skel[SHOULDER_CENTER,:] - skel[SHOULDER_RIGHT,:]) + np.linalg.norm(skel[SHOULDER_RIGHT,:] - skel[ELBOW_RIGHT,:]) + \
		#	np.linalg.norm(skel[ELBOW_RIGHT,:] - skel[WRIST_RIGHT,:]) + np.linalg.norm(skel[WRIST_RIGHT,:] - skel[HAND_RIGHT, :])
		#leftleg_length = np.linalg.norm(skel[HIP_CENTER,:] - skel[HIP_LEFT,:]) + np.linalg.norm(skel[HIP_LEFT,:] - skel[KNEE_LEFT,:]) + \
		#	np.linalg.norm(skel[KNEE_LEFT,:] - skel[ANKLE_LEFT,:])
		#rightleg_length = np.linalg.norm(skel[HIP_CENTER,:] - skel[HIP_RIGHT,:]) + np.linalg.norm(skel[HIP_RIGHT,:] - skel[KNEE_RIGHT,:]) + \
		#	np.linalg.norm(skel[KNEE_RIGHT,:] - skel[ANKLE_RIGHT,:])
		# get maximum length of torso
		#max_length = np.max([spine_length, leftarm_length, rightarm_length, leftleg_length, rightleg_length])
		# normalize postions
		norm_skel = (skel - center) / spine_length
		norm_seq[idx, :] = norm_skel.flatten()
		#print(norm_skel)
	return norm_seq

def feat_extrac(seq):
	'''
	Feature extraction for skeleton sequence

	:param seq: skeleton joints sequence , (seq#, 60) array
	:return: feats
   '''
	np.seterr(over='raise')
	seq_num = seq.shape[0]
	bones_num = len(nui_skeleton_conn)
	feats = np.zeros([seq_num, 162]) # joints: 3*20, bones:6*17
	# extrac features for each joint and bone
	for frame in tqdm(range(seq_num)):
		pos = np.reshape(seq[frame,:], [NUI_SKELETON_POSITION_COUNT, 3]) # get position for all 20 joints at current frame
		# set features for each joint
		feats[frame, 0:NUI_SKELETON_POSITION_COUNT*3] = pos.ravel() # joint 3d position
		# set features for each bone
		for i in range(bones_num):
			offset = NUI_SKELETON_POSITION_COUNT*3 # offset index for bone features
			# center point of bone
			center = (pos[nui_skeleton_conn[i][0], :] + pos[nui_skeleton_conn[i][1], :]) / 2
			feats[frame, i*6+offset:i*6+3+offset] = center

			# get the bone vector
			vec = pos[nui_skeleton_conn[i][0], :] - pos[nui_skeleton_conn[i][1], :]
			# norm vector
			norms = np.linalg.norm(vec)
			if norms == 0:
				norm_vec = np.array([0,0,0])
			else:
				norm_vec = vec / norms
			feats[frame, i*6+3+offset:i*6+6+offset] = norm_vec
	return feats


def main():
	# extract features and label array for our UA-CAR dataset single person data
	data_root = '/home/yi/PycharmProjects/RCN/data/'
	save_dir = '/home/yi/PycharmProjects/relation_network/data/UA_new/'
	for i in range(201):
		seq_num = "%03d" % int(i+1)
		print('Process sequence : ', seq_num)
		folder_name = 'seq_' + seq_num
		skelfile = os.path.join(data_root, folder_name, folder_name+'_c_s.mat')
		labelfile = os.path.join(data_root, folder_name, folder_name+'_label.txt')

		if os.path.exists(os.path.join(save_dir, folder_name+'.pkl')):
			print('file exists')
			continue

		#print(skelfile, labelfile)
		seq, nf = data_loader(skelfile)      # numpy array storing all 3D points of sequence
		norm_seq = normalize_skel(seq)
		feats = feat_extrac(norm_seq)
		labels = label_loader(labelfile, nf)
		print(feats.shape)
		print(labels.shape)
		g_data = {
			'feat':feats,
			'label':labels
		}
		save_dict(g_data, os.path.join(save_dir, folder_name+'.pkl'))

def test():
	data_root = '/home/yi/PycharmProjects/RCN/data/'
	seq_num = "%03d" % int(6)
	print('sequence number: ', seq_num)
	folder_name = 'seq_' + seq_num
	skelfile = os.path.join(data_root, folder_name, folder_name+'_c_s.mat')
	labelfile = os.path.join(data_root, folder_name, folder_name+'_label.txt')

	seq, nf = data_loader(skelfile)      # numpy array storing all 3D points of sequence
	feats = feat_extrac(seq)
	labels = label_loader(labelfile, nf)
	print(feats.shape)
	print(labels.shape)
	g_data = {
		'feat':feats,
		'label':labels
	}
	save_dict(g_data, './data.pkl')
	g_data2 = load_dict('./data.pkl')
	print(g_data['feat'] == g_data2['feat'])
	print(g_data['label'] == g_data2['label'])

def dataset_stat():
	data_root = '/home/yi/PycharmProjects/RCN/data/'
	stat = np.zeros((201, 35))  # statistics for dataset to denote if the class appear at each video
	for i in range(201):
		seq_num = "%03d" % int(i+1)
		print('sequence number: ', seq_num)
		folder_name = 'seq_' + seq_num
		skelfile = os.path.join(data_root, folder_name, folder_name+'_c_s.mat')
		labelfile = os.path.join(data_root, folder_name, folder_name+'_label.txt')

		skelstct = loadmat(skelfile)
		skels_3dpos = skelstct['body']['Position'] # get the 3d joints position
		_, nf = skels_3dpos.shape
		labels = label_loader(labelfile, nf)
		positive = np.sum(labels, axis=0) > 1
		stat[i][positive] = 1
		#print(i+1, ': ',  positive)
	np.savetxt('stat.csv', stat, fmt='%d', delimiter=',')

def save_gt():
	data_root = '/home/yi/PycharmProjects/RCN/data/'
	for i in range(67):
		seq_num = "%03d" % int(i*3+3)
		print('sequence number: ', seq_num)
		folder_name = 'seq_' + seq_num
		skelfile = os.path.join(data_root, folder_name, folder_name+'_c_s.mat')
		labelfile = os.path.join(data_root, folder_name, folder_name+'_label.txt')

		seq, nf = data_loader(skelfile)      # numpy array storing all 3D points of sequence
		#feats = feat_extrac(seq)
		labels = label_loader(labelfile, nf)
		print(labels.shape)
		np.savetxt('seq_'+seq_num+'.txt', labels, fmt='%0.5f')

if __name__ == '__main__':
	#save_gt()
	main()
