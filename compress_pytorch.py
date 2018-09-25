import torch
import os
import numpy as np
from sklearn.decomposition import TruncatedSVD, SparsePCA
import sys
import math
import argparse

def compress_l2(W, b, fac): # This the k-Means Coreset compression
	W_s = W.reshape(W.shape[0], W.size/W.shape[0])
	b_s = b.reshape(b.shape[0], b.size/b.shape[0])

	X = np.concatenate([W_s,b_s], axis=1).transpose()

	if fac == -1:
		# Use (4/sqrt(3) * median eigenvalue as trunc-value)
		# we have to do SVD on the full matrix to 
		# figure out median eigenvalue
		Ux, sx, Vx = np.linalg.svd(X)
		r = (4.0/(3.0**0.5))*np.median(sx)
		n_comp = np.sum(np.array([int(j>=r) for j in sx]))
		print 'Optimal Eigenvalue: %f, Number of components selected: %d/%d' % (r, n_comp, len(sx))
	elif fac == 0:
		print 'No compression, Number of components selected: %d/%d' % (W.shape[0], W.shape[0])
		return (W, b)
	else:
		n_comp= int(W.shape[0]*fac)
		print 'Predefined ratio, Number of components selected: %d/%d' % (n_comp, W.shape[0])

	# perform truncated SVD

	pca = TruncatedSVD(n_components=n_comp)
	pca.fit(X)
	Xr = pca.inverse_transform(pca.fit_transform(X))
	# Approximate the original weights and biases
	Wf = Xr.transpose()[:,:-1].reshape(W.shape)
	bf = Xr.transpose()[:,-1].reshape(b.shape)

	return Wf, bf

def compress_l1(W, b, fac, alpha): # This is the Sparse-Coreset setting
	W_s = W.reshape(W.shape[0], W.size/W.shape[0])
	b_s = b.reshape(b.shape[0], b.size/b.shape[0])

	X = np.concatenate([W_s,b_s], axis=1).transpose()

	if fac == -1:
		# Use (4/sqrt(3) * median eigenvalue as trunc-value)
		# we have to do SVD on the full matrix to 
		# figure out median eigenvalue
		Ux, sx, Vx = np.linalg.svd(X)
		r = (4.0/(3.0**0.5))*np.median(sx)
		n_comp = np.sum(np.array([int(j>=r) for j in sx]))
		print 'Optimal Eigenvalue: %f, Number of components selected: %d/%d' % (r, n_comp, len(sx))
	elif fac == 0:
		print 'No compression, Number of components selected: %d/%d' % (W.shape[0], W.shape[0])
		return (W, b)
	else:
		n_comp=int(W.shape[0]*fac)
		print 'Predefined ratio, Number of components selected: %d/%d' % (n_comp, W.shape[0])

	# perform truncation

	pca = SparsePCA(n_components=n_comp, alpha=alpha)
	Xs = pca.fit_transform(X)
	Xr = np.dot(Xs, pca.components_)
	# Approximate the original weights
	Wf = Xr.transpose()[:,:-1].reshape(W.shape)
	bf = Xr.transpose()[:,-1].reshape(b.shape)

	return Wf, bf

def compress_network(filename, output_file, facs=None, alphas=None):
	model = torch.load(filename)
	n_layers = len(model['state_dict'].keys())/2
	# assuming each layer has weights and biases

	for i in range(n_layers):
		W_i = model['state_dict'].values()[2*i].cpu().numpy()
		b_i = model['state_dict'].values()[2*i +1].cpu().numpy()
		
		if facs == None:
			fac_i = -1
		else:
			fac_i = facs[i]

		if alphas == None:
			# do L2 SVD
			W_n, b_n = compress_l2(W_i, b_i, fac_i)
		else:
			W_n, b_n = compress_l1(W_i, b_i, fac_i, alpha[i])

		err = np.linalg.norm((W_n-W_i).flatten(),ord=2)/(np.linalg.norm(W_n.flatten(), ord=2)*np.linalg.norm(W_i.flatten(), ord=2))
		print 'For layer %d, the reconstruction has Normalized MSE %f' % (i+1, err)

		model['state_dict'].values()[2*i] = torch.from_numpy(W_n).cuda()
		model['state_dict'].values()[2*i+1] = torch.from_numpy(b_n).cuda()

	torch.save(model, output_file)
# Argument Parser
parser = argparse.ArgumentParser('Compress using K- or S- coresets')
parser.add_argument('--input', '-i', help='Input pytorch model')
parser.add_argument('--output', '-o', help='Output pytorch model')
parser.add_argument('--type', '-t', help='Type of coreset (1 for Coreset-S and 2 for Coreset-K, default 2)', type=int, default=2, required=False)
parser.add_argument('--ratios', '-r', help='Compression ratios per layer (enter -1 for 4/sqrt(3)*median_eigenvalue and 0 for no compression)', nargs='+', type=float, default=None)
parser.add_argument('--alphas', '-a', help='Alphas for L1 Coresets', type=float, nargs='+', default=None)

args = parser.parse_args()
print args

args.alphas = None if not args.alphas else [float(x) for x in args.alphas]
args.ratios = [float(x) for x in args.ratios]

compress_network(args.input, args.output, facs=args.ratios, alphas=args.alphas)





