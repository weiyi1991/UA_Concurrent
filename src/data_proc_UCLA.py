'''
Feature extraction for the 3D skeleton joints sequences

Author: Yi Wei
Date: 06/20/2018

'''

import os
import numpy as np
from tqdm import tqdm
import sys
import pickle

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
	#[ANKLE_RIGHT, FOOT_RIGHT ]
]

# label classes
label_classes = [
		'drink', 'make_a_call', 'turn_on_monitor',
		'type_on_keyboard', 'fetch_water', 'pour_water',
		'press_button', 'pick_up_trash', 'throw_trash',
		'bend_down', 'sit', 'stand']

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
		#https://stackoverflow.com/questions/46046752/python-3-unicodedecodeerror-ascii-codec-cant-decode-byte-0xe2-in-position-0
        ret_di = pickle.load(f, encoding='latin1')
    return ret_di

def label_loader(file, frm_num_list):
	'''
	Reading label file, parsing the labels for a sequence to a binary matrix.

	input:
		file - label file contains 12 class with corresponding frame gaps
		frm_num_list - list contains all frame numbers of this sequence
	:return:
		[#frame, #class] numpy array. 1 represents action with label class occurs at current frame.
	'''
	label_mat = np.zeros((len(frm_num_list), len(label_classes)))
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
				for i in range(int(start_idx), int(end_idx)+1):
					try:
						label_mat[frm_num_list.index('%07d' % i)][class_idx] = 1
					except Exception as e:
						print('Errors in sequence file ', file)
						print(str(e))
	return label_mat

def _get_frm_num(file_name):
	# return the frame id of a frame skeleton file, e.g. 'frame_0000114_tc_18100530_skeletons.txt' -> 0000114
	return file_name.split('_')[1]

def data_loader(seqList):
	'''
	Loading skeleton data of Concurrent Activity Dataset

	Input:
		Input sequence list, containing all skeleton data files of a sequence.

	:return:
		[#frame,  60] Numpy array. 60 for 20 joints 3D coordinates
	'''
	seq = np.zeros([len(seqList), 60])     # 3D point array for sequence
	for i, frame in enumerate(seqList):
		try:
			data = np.loadtxt(frame, dtype=np.float64, delimiter=',', skiprows=1)
		except:
			print('File format error when opening:  ' + frame)
		else:
			points = data[:, :3]
			seq[i] = np.reshape(points, [1, 60])
	return seq

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
	feats = np.zeros([seq_num, 162]) # joints: 3*20, bones: 6*17
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
	data_root = '/home/yi/PycharmProjects/double rnn/data/Concurrent Action Dataset'
	for i in range(61):
		seq_num = "%03d" % int(i+1)
		print('sequence number: ', seq_num)
		folder_name = 'sequence_' + seq_num
		frame_folder = os.path.join(data_root, 'sequence skeleton', folder_name)
		label_folder = os.path.join(data_root, 'sequence label')
		frame_files = os.listdir(frame_folder)
		frame_files.sort()  # sort files in time order
		frm_list = []  # frame file list path
		frm_num_list = []  # frame index num for all files of a sequence in the same order of frm_list.
		for i in range(len(frame_files)):
			frm_list.append(os.path.join(frame_folder, frame_files[i]))
			frm_num_list.append(_get_frm_num(frame_files[i]))
		label_file = os.path.join(label_folder, folder_name+'.txt')  # label file path

		seq = data_loader(frm_list)      # numpy array storing all 3D points of sequence
		norm_seq = normalize_skel(seq)
		feats = feat_extrac(norm_seq)
		labels = label_loader(label_file, frm_num_list)

		g_data = {
            'feat':feats,
            'label':labels
        }
		print(feats.shape)
		print(labels.shape)
		save_dict(g_data, os.path.join('../data', folder_name+'.pkl'))
		print('========================')

def test():
	data_root = '/home/yi/PycharmProjects/double rnn/data/Concurrent Action Dataset'
	seq_num = "%03d" % int(58)
	print('sequence number: ', seq_num)
	folder_name = 'sequence_' + seq_num
	folder = os.path.join(data_root, 'sequence skeleton', folder_name)
	files = os.listdir(folder)
	files.sort()    # sort files in time order
	seqList = []    # sequence file list
	for i in range(len(files)):
		seqList.append(os.path.join(folder, files[i]))
	seq = data_loader(seqList)      # numpy array storing all 3D points of sequence
	norm_seq = normalize_skel(seq)
	feats = feat_extrac(norm_seq)
	print(feats.shape, 'features:\n', feats[0,:])
	print('sequence: \n', seq[0,:])

if __name__ == '__main__':
	main()
