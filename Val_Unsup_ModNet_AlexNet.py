import argparse
import os
import shutil
import time
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])) # Dictionary of model names available with PyTorch

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Network Modification')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset') # Path to dataset
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')  # The architecture to use
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)') # The number of workers for the dataset loader
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)') # The batch size for inference
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)') # The number of batches after which the performance is displayed in the console
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)') # The flag indicating if a saved model is to be loaded
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set') # A flag to specify if the network is to be used in evaluation mode or not
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model') # A flag indicating if pre-trained weights are to be used from the model zoo

ip_fldr_path = '/path/where/the/clustering/info/is/saved/' 
save_path = '/path/to/save/the/modified/network/';

best_prec1 = 0

# Set the thresholds for the different layers -- Assuming AlexNet, should sync with the flags computed in Gen_Clust_New_AllLayers_Omp.py
t_fc1 = 0.0003; t_fc2 = 0.0003; t_conv5 = 0.0003; t_conv4 = 0.0003; t_conv3 = 0.0003; t_conv2 = 0.0003; t_conv1 = 0.0003;
# Geneate the masks at every layer
# Variables for saving the clustering information for the overall map

n_list1 = np.zeros((1, 64)); # The flags for every filter
n_list2 = np.zeros((1, 192)); # The flags for every filter
n_list3 = np.zeros((1, 384)); # The flags for every filter
n_list4 = np.zeros((1, 256)); # The flags for every filter
n_list5 = np.zeros((1, 256)); # The flags for every filter
n_list6 = np.zeros((1, 4096)); # The flags for every filter
n_list7 = np.zeros((1, 4096)); # The flags for every filter

class_ctr = 0; dirs = os.listdir(ip_fldr_path); dirs = dirs[:];

for dir1 in dirs: # For each folder
    directory = ip_fldr_path + dir1 #+ '/'
    print('In Directory: ' + dir1 + '\n')

    with open( directory + '/Vars_FC2_' + str(t_fc2) + '.pickle', 'rb') as f: # Load the cluster information for the top FC 2 Layers
        n_list7_t = pickle.load(f)
    n_list7_t = ((np.sum(n_list7_t, axis = 0) > 0) * 1.0).reshape(1, n_list7.shape[1])
    n_list7 +=  n_list7_t # Sum over classes 
    n_list7 = (n_list7 > 0) * 1.0

    with open( directory + '/Vars_FC1_' + str(t_fc1) + '.pickle', 'rb') as f: # Load the cluster information for the top FC 1 Layers
        n_list6_t = pickle.load(f)
    n_list6_t = ((np.sum(n_list6_t, axis = 0) > 0) * 1.0).reshape(1, n_list6.shape[1])
    n_list6 +=  n_list6_t # Sum over classes 
    n_list6 = (n_list6 > 0) * 1.0

    with open( directory + '/Vars_Conv5_' + str(t_conv5) + '.pickle', 'rb') as f: # Load the cluster information for the Conv5 Layers
        n_list5_1_t = pickle.load(f)
    n_list5_1_t = ((np.sum(n_list5_1_t, axis = 0) > 0) * 1.0).reshape(1, n_list5.shape[1])
    n_list5 += n_list5_1_t # ( np.sum(n_list5_1_t[0], axis = 0) / (n_clust5[class_ctr] * 1.0))
    n_list5 = (n_list5 > 0) * 1.0

    with open( directory + '/Vars_Conv4_' + str(t_conv4) + '.pickle', 'rb') as f: # Load the cluster information for the Conv4 Layers
        n_list4_1_t = pickle.load(f)
    n_list4_1_t = ((np.sum(n_list4_1_t, axis = 0) > 0) * 1.0).reshape(1, n_list4.shape[1]) # The 0 index is because it was saved as a list
    n_list4 += n_list4_1_t # ( np.sum(n_list4_1_t[0], axis = 0) / (n_clust4[class_ctr] * 1.0))
    n_list4 = (n_list4 > 0) * 1.0

    with open( directory + '/Vars_Conv3_' + str(t_conv3) + '.pickle', 'rb') as f: # Load the cluster information for the Conv3 Layers
        n_list3_1_t = pickle.load(f)
    n_list3_1_t = ((np.sum(n_list3_1_t, axis = 0) > 0) * 1.0).reshape(1, n_list3.shape[1])
    n_list3 += n_list3_1_t # ( np.sum(n_list3_1_t[0], axis = 0) / (n_clust3[class_ctr] * 1.0))
    n_list3 = (n_list3 > 0) * 1.0

    with open( directory + '/Vars_Conv2_' + str(t_conv2) + '.pickle', 'rb') as f: # Load the cluster information for the Conv2 Layers
        n_list2_1_t = pickle.load(f)

    n_list2_1_t = ((np.sum(n_list2_1_t, axis = 0) > 0) * 1.0).reshape(1, n_list2.shape[1])
    n_list2 += n_list2_1_t # ( np.sum(n_list2_1_t[0], axis = 0) / (n_clust2[class_ctr] * 1.0))
    n_list2 = (n_list2 > 0) * 1.0

    with open( directory + '/Vars_Conv1_' + str(t_conv1) + '.pickle', 'rb') as f: # Load the cluster information for the Conv1 Layers
        n_list1_1_t = pickle.load(f)
    n_list1_1_t = ((np.sum(n_list1_1_t, axis = 0) > 0) * 1.0).reshape(1, n_list1.shape[1])
    n_list1 += n_list1_1_t # ( np.sum(n_list1_1_t[0], axis = 0) / (n_clust1[class_ctr] * 1.0))
    n_list1 = (n_list1 > 0) * 1.0

    class_ctr += 1


print("Acceptances of the number of filters: " + str(np.sum(n_list1)) + " " + str(np.sum(n_list2)) + " " + str(np.sum(n_list3)) + " " + str(np.sum(n_list4)) + " " + str(np.sum(n_list5)) + " " + str(np.sum(n_list6)) + " " + str(np.sum(n_list7)))

args = parser.parse_args()
batch_size=args.batch_size;


# main() function starts here:
def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained: # Load pre-trained model
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'): # Push only the convolution layers to data-parallel for these 2 networks
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally load a saved model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        # Re-adjust the parameters appropriately for AlexNet
        if args.arch.startswith('alexnet'):
            
            # Convolution Layer 1
            c =  (model.features._modules['0'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.features._modules['0'].bias).cpu().data.numpy() # Get the bias
            # Apply the compression
            for i in range((n_list1.shape)[1]): # Iterate over each filter
                if n_list1[0, i] == 0: # If the filter is not selected
                    c[i, :, :, :] = np.zeros(( (c.shape)[1], (c.shape)[2], (c.shape)[3] ))
                    c_b[i] = 0

            model.features._modules['0'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.features._modules['0'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias

            # Convolution Layer 2
            c =  (model.features._modules['3'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.features._modules['3'].bias).cpu().data.numpy() # Get the bias
            # Apply the compression from Conv1 to Conv2
            for filt in range((n_list2.shape)[1]): # Iterate over each filter at Conv2
                for i in range((n_list1.shape)[1]): # Iterate over each filter at Conv1
                    if n_list1[0, i] == 0: # If the filter is not selected
                        c[filt, i, :, :] = np.zeros(( (c.shape)[2], (c.shape)[3] ))
                        #c_b[i] = 0
            # Apply the compression
            for i in range((n_list2.shape)[1]): # Iterate over each filter
                if n_list2[0, i] == 0: # If the filter is not selected
                    c[i, :, :, :] = np.zeros(( (c.shape)[1], (c.shape)[2], (c.shape)[3] ))
                    c_b[i] = 0

            model.features._modules['3'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.features._modules['3'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias

            # Convolution Layer 3
            c =  (model.features._modules['6'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.features._modules['6'].bias).cpu().data.numpy() # Get the bias
            # Apply the compression from Conv2 to Conv3
            for filt in range((n_list3.shape)[1]): # Iterate over each filter at Conv3
                for i in range((n_list2.shape)[1]): # Iterate over each filter at Conv2
                    if n_list2[0, i] == 0: # If the filter is not selected
                        c[filt, i, :, :] = np.zeros(( (c.shape)[2], (c.shape)[3] ))
                        #c_b[i] = 0
            # Apply the compression
            for i in range((n_list3.shape)[1]): # Iterate over each filter
                if n_list3[0, i] == 0: # If the filter is not selected
                    c[i, :, :, :] = np.zeros(( (c.shape)[1], (c.shape)[2], (c.shape)[3] ))
                    c_b[i] = 0

            model.features._modules['6'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.features._modules['6'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias

            # Convolution Layer 4
            c =  (model.features._modules['8'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.features._modules['8'].bias).cpu().data.numpy() # Get the bias
            # Apply the compression from Conv3 to Conv4
            for filt in range((n_list4.shape)[1]): # Iterate over each filter at Conv4
                for i in range((n_list3.shape)[1]): # Iterate over each filter at Conv3
                    if n_list3[0, i] == 0: # If the filter is not selected
                        c[filt, i, :, :] = np.zeros(( (c.shape)[2], (c.shape)[3] ))
                        #c_b[i] = 0
            # Apply the compression
            for i in range((n_list4.shape)[1]): # Iterate over each filter
                if n_list4[0, i] == 0: # If the filter is not selected
                    c[i, :, :, :] = np.zeros(( (c.shape)[1], (c.shape)[2], (c.shape)[3] ))
                    c_b[i] = 0

            model.features._modules['8'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.features._modules['8'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias

            # Convolution Layer 5
            c =  (model.features._modules['10'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.features._modules['10'].bias).cpu().data.numpy() # Get the bias
            # Apply the compression from Conv4 to Conv5
            for filt in range((n_list5.shape)[1]): # Iterate over each filter at Conv5
                for i in range((n_list4.shape)[1]): # Iterate over each filter at Conv4
                    if n_list4[0, i] == 0: # If the filter is not selected
                        c[filt, i, :, :] = np.zeros(( (c.shape)[2], (c.shape)[3] ))
                        #c_b[i] = 0
            # Apply the compression
            for i in range((n_list5.shape)[1]): # Iterate over each filter
                if n_list5[0, i] == 0: # If the filter is not selected
                    c[i, :, :, :] = np.zeros(( (c.shape)[1], (c.shape)[2], (c.shape)[3] ))
                    c_b[i] = 0

            model.features._modules['10'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.features._modules['10'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias

            # Store the number of responses in each filter at Conv5
            c_w = 6; c_h = 6; tot_neuron = c_w * c_h;

            # Fully Connected Layer 1
            c =  (model.classifier._modules['1'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.classifier._modules['1'].bias).cpu().data.numpy() # Get the bias
            # Apply the compression from Conv5 to FC1
            for filt in range((n_list6.shape)[1]): # Iterate over each filter at FC1
                for i in range((n_list5.shape)[1]): # Iterate over each filter at Conv5
                    if n_list5[0, i] == 0: # If the filter is not selected
                        c[filt, (i * tot_neuron) : ((i + 1) * tot_neuron) ] = np.zeros(( tot_neuron ))
                        #c_b[i] = 0
            # Apply the compression
            for i in range((n_list6.shape)[1]): # Iterate over each filter
                if n_list6[0, i] == 0: # If the filter is not selected
                    c[i, :] = np.zeros(( (c.shape)[1] ))
                    c_b[i] = 0

            model.classifier._modules['1'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.classifier._modules['1'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias
            
            # Fully Connected Layer 2
            c =  (model.classifier._modules['4'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.classifier._modules['4'].bias).cpu().data.numpy() # Get the bias
            
            # Apply the compression from FC1 to FC2
            for filt in range((n_list7.shape)[1]): # Iterate over each filter at FC2
                for i in range((n_list6.shape)[1]): # Iterate over each filter at FC1
                    if n_list6[0, i] == 0: # If the filter is not selected
                        c[filt, i] = 0
                        #c_b[i] = 0
            # Apply the compression
            for i in range((n_list7.shape)[1]): # Iterate over each filter
                if n_list7[0, i] == 0: # If the filter is not selected
                    c[i, :] = np.zeros(( (c.shape)[1] ))
                    c_b[i] = 0
                        
            model.classifier._modules['4'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.classifier._modules['4'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias
            
            # Fully Connected Layer 3 - For the Softmax Layer
            c =  (model.classifier._modules['6'].weight).cpu().data.numpy() # Get the weights
            c_b = (model.classifier._modules['6'].bias).cpu().data.numpy() # Get the bias
            # Apply the compression
            for i in range((n_list7.shape)[1]): # Iterate over each filter
                if n_list7[0, i] == 0: # If the filter is not selected
                    c[:, i] = np.zeros(( (c.shape)[0] ))
                    #c_b[i] = 0

            model.classifier._modules['6'].weight = nn.Parameter(torch.Tensor(c).cuda()) # Re-assign the weights
            model.classifier._modules['6'].bias = nn.Parameter(torch.Tensor(c_b).cuda()) # Re-assign the bias
            

    cudnn.benchmark = True

    # Data loader code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.evaluate: # Evaluate the model on the validation set
        validate(val_loader, model, criterion)
        epoch = 0;
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, True, save_path)
    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    prev_label = -1 # Initialize Previous Label

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, save_path = './', filename='checkpoint.pth.tar'):
    torch.save(state, save_path + filename)
    if is_best:
        shutil.copyfile(save_path + filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


#if __name__ == '__alexNet_NList__':
main()
