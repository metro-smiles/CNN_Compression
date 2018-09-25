from numpy import *
import os
from pylab import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import sys
import math
from PIL import Image
import pylab
from numpy import linalg as LA

import sklearn as sk
from sklearn.cluster import KMeans

#from caffe_classes import class_names
import pickle

# Perform Orthogonal Matching Pursuit -  to determine which filters to prune
def omp_appx(clust_arr, thresh = 0.1):
	# Obtain the dimensions of the Data Matrix
	tot_samp, nos_dim = clust_arr.shape
	chsn_nrn = np.floor(thresh * nos_dim)
	#dat = clust_arr;
	
	
	# One n_list for the full layer across all samples
	n_list = np.zeros((1, nos_dim)) # Keeps the ordering of the different neurons	

	# Compute the Norms of the Projections
	proj_nrm = LA.norm(clust_arr, axis = 0)
	# Sort the array in descending order
	ids = proj_nrm.argsort()[::-1][:chsn_nrn]
	n_list[0, ids] = 1.0;
	
	return n_list

# Folder Details:
fldr = '/folder/to/layer/wise/feature/activations/'
dirs = os.listdir(fldr); 
op_fldr = '/path/to/store/the/output/of/the/filter/ordering/'

class_ctr = 0;

# Read in the Feature Activation Files
for dir1 in dirs:  
	print("Currently in the ImageNet_Proc folder.: " + str(dir1) + " \n") 
	directory = fldr + dir1 
	
	# Load the saved activations from disk -- Assuming AlexNet
	# Load Conv Layer 1 response
	with open( directory + '/Op_Conv1.pickle', 'rb') as f: # Load the sample information for Maxpool 1 Layer 
		c1_arr = pickle.load(f)
	
	# Load Conv 2 response
	with open( directory + '/Op_Conv2.pickle', 'rb') as f: # Load the sample information for Maxpool 2 Layer 
		c2_arr = pickle.load(f)
	
	# Load Conv 3 response
	with open( directory + '/Op_Conv3.pickle', 'rb') as f: # Load the sample information for Conv 3 Layer 
		c3_arr = pickle.load(f)
	
	# Load Conv 4 response
	with open( directory + '/Op_Conv4.pickle', 'rb') as f: # Load the sample information for Conv 4 Layer 
		c4_arr = pickle.load(f)
	
	# Load Conv 5 response
	with open( directory + '/Op_Conv5.pickle', 'rb') as f: # Load the sample information for Maxpool 5 Layer 
		c5_arr = pickle.load(f)
	
	## For the Fully-Connected Components: FC 1
	with open( directory + '/Op_FC1.pickle', 'rb') as f: # Load the sample information for FC 6 Layer 
		fc1_arr = pickle.load(f)
	
	# For FC 2
	with open( directory + '/Op_FC2.pickle', 'rb') as f: # Load the sample information for FC 7 Layer 
		fc2_arr = pickle.load(f)
	
	# List of the thresholds for pruning the filters	
	t_fc1_lst = [0.0003]; t_fc2_lst = [0.0003]; t_conv5_lst = [0.0003]; t_conv4_lst = [0.0003]; t_conv3_lst = [0.0003]; t_conv2_lst = [0.0003]; t_conv1_lst = [0.0003];
	
	# Compute the clusters and the masks for each of the Convolution Layers
	# The loops potentially allow us to potentually save the compression results at different rates
	for thresh_conv in t_conv1_lst: 
		n_list1 = compute_node_list(c1_arr, thresh_conv)
		with open( op_fldr + dir1 + '/Vars_Conv1_' + str(thresh_conv) + '.pickle', 'wb') as f: # Save the neuron list information
			pickle.dump(n_list1, f)
		print("Acceptances Conv1: " + str( np.sum(((np.sum(n_list1, axis = 0) > 0) * 1.0).reshape(1, n_list1.shape[1])) ))
	
	for thresh_conv in t_conv2_lst:
		n_list2 = compute_node_list(c2_arr, thresh_conv)
		with open( op_fldr + dir1 + '/Vars_Conv2_' + str(thresh_conv) + '.pickle', 'wb') as f: # Save the neuron list information
			pickle.dump(n_list2, f)
		print("Acceptances Conv2: " + str( np.sum(((np.sum(n_list2, axis = 0) > 0) * 1.0).reshape(1, n_list2.shape[1])) ))
	
	for thresh_conv in t_conv3_lst:
		n_list3 = compute_node_list(c3_arr, thresh_conv)
		with open( op_fldr + dir1 + '/Vars_Conv3_' + str(thresh_conv) + '.pickle', 'wb') as f: # Save the neuron list information
			pickle.dump(n_list3, f)
		print("Acceptances Conv3: " + str( np.sum(((np.sum(n_list3, axis = 0) > 0) * 1.0).reshape(1, n_list3.shape[1])) ))
	
	for thresh_conv in t_conv4_lst:
		n_list4 = compute_node_list(c4_arr, thresh_conv)
		with open( op_fldr + dir1 + '/Vars_Conv4_' + str(thresh_conv) + '.pickle', 'wb') as f: # Save the neuron list information
			pickle.dump(n_list4, f)
		print("Acceptances Conv4: " + str( np.sum(((np.sum(n_list4, axis = 0) > 0) * 1.0).reshape(1, n_list4.shape[1])) ))

	for thresh_conv in t_conv5_lst:
		n_list5 = omp_appx(c5_arr, thresh_conv)
		with open( op_fldr + dir1 + '/Vars_Conv5_' + str(thresh_conv) + '.pickle', 'wb') as f: # Save the neuron list information
			pickle.dump(n_list5, f)
		print("Acceptances Conv5: " + str( np.sum(((np.sum(n_list5, axis = 0) > 0) * 1.0).reshape(1, n_list5.shape[1])) ))
	
	for thresh in t_fc1_lst:
		n_list6 = omp_appx(fc1_arr, thresh)
		with open( op_fldr + dir1 + '/Vars_FC1_' + str(thresh) + '.pickle', 'wb') as f: # Save the neuron list information
			pickle.dump(n_list6, f)
		print("Acceptances FC1: " + str( np.sum(((np.sum(n_list6, axis = 0) > 0) * 1.0).reshape(1, n_list6.shape[1])) ))
	
	for thresh in t_fc2_lst: 
		n_list7 = omp_appx(fc2_arr, thresh)
		with open( op_fldr + dir1 + '/Vars_FC2_Omp_' + str(thresh) + '.pickle', 'wb') as f: # Save the neuron list information
			pickle.dump(n_list7, f)
		print("Acceptances FC2: " + str( np.sum(((np.sum(n_list7, axis = 0) > 0) * 1.0).reshape(1, n_list7.shape[1])) ))

	class_ctr += 1;