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
parser = argparse.ArgumentParser(description='PyTorch ImageNet Processing')
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

op_fldr_path = op_fldr_path_dir  = '/path/to/save feature/files/'
save_path = '/path/to/save/models/';

best_prec1 = 0

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

    # optionally load the model from a checkpoint
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

    cudnn.benchmark = True

    # Data loading code
    # Create the dataloader for processing the training set
    valdir = os.path.join(args.data, 'train') # 'val' 
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

    if args.evaluate: # If model is being run in inference mode
        prec1 = validate(val_loader, model, criterion)
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

    # Initialize the storage variables -- This is for AlexNet
    fc2_arr = None; fc1_arr = None; mp5_arr = None; conv4_arr = None; conv3_arr = None; mp2_arr = None; mp1_arr = None; # Assume AlexNet

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        op = target_var.cpu().data.numpy() # Convert the ouput 

        # Assuming AlexNet
        # Process the outputs from every layer:
        output = ((((model._modules.items())[0])[1]).module)[0](input_var.cuda()) # .module
        output = ((((model._modules.items())[0])[1]).module)[1](output)
        output = ((((model._modules.items())[0])[1]).module)[2](output) # Gives Maxpool 1 as output
        layer_op1 = output.cpu().data.numpy();  # Convert the Conv1 output to Numpy array #layer_op1 = np.mean(layer_op1, axis = 1)
        tmp = np.zeros(( (layer_op1.shape)[0], (layer_op1.shape)[1] ))
        # Loop over every feature map of every sample
        for samp in range(tmp.shape[0]):
            for fm in range(tmp.shape[1]):
                tmp[samp, fm] = np.amax(layer_op1[samp, fm, :, :]) # Extract the maximum response from every feature map
        layer_op1 = tmp
        #print("Size of Maxpool 1 in mini-batch: " + str(layer_op1.shape) + "\n")

        output = ((((model._modules.items())[0])[1]).module)[3](output) 
        output = ((((model._modules.items())[0])[1]).module)[4](output) 
        output = ((((model._modules.items())[0])[1]).module)[5](output) # Gives Maxpool 2 as output
        layer_op2 = output.cpu().data.numpy(); # Convert the Conv2 output to Numpy array # layer_op2 = np.mean(layer_op2, axis = 1)
        tmp = np.zeros(( (layer_op2.shape)[0], (layer_op2.shape)[1] ))
        # Loop over every feature map of every sample
        for samp in range(tmp.shape[0]):
            for fm in range(tmp.shape[1]):
                tmp[samp, fm] = np.amax(layer_op2[samp, fm, :, :]) # Extract the maximum response from every feature map
        layer_op2 = tmp
        #print("Size of Maxpool 2 in mini-batch: " + str(layer_op2.shape) + "\n")

        output = ((((model._modules.items())[0])[1]).module)[6](output)
        output = ((((model._modules.items())[0])[1]).module)[7](output) # Gives Conv 3 as output
        layer_op3 = output.cpu().data.numpy(); # Convert the Conv3 output to Numpy array # layer_op3 = np.mean(layer_op3, axis = 1)
        tmp = np.zeros(( (layer_op3.shape)[0], (layer_op3.shape)[1] ))
        # Loop over every feature map of every sample
        for samp in range(tmp.shape[0]):
            for fm in range(tmp.shape[1]):
                tmp[samp, fm] = np.amax(layer_op3[samp, fm, :, :]) # Extract the maximum response from every feature map
        layer_op3 = tmp
        #print("Size of Conv 3 in mini-batch: " + str(layer_op3.shape) + "\n")

        output = ((((model._modules.items())[0])[1]).module)[8](output)
        output = ((((model._modules.items())[0])[1]).module)[9](output) # Gives Conv 4 as output
        layer_op4 = output.cpu().data.numpy(); # Convert the Conv4 output to Numpy array # layer_op4 = np.mean(layer_op4, axis = 1) 
        tmp = np.zeros(( (layer_op4.shape)[0], (layer_op4.shape)[1] ))
        # Loop over every feature map of every sample
        for samp in range(tmp.shape[0]):
            for fm in range(tmp.shape[1]):
                tmp[samp, fm] = np.amax(layer_op4[samp, fm, :, :]) # Extract the maximum response from every feature map
        layer_op4 = tmp
        #print("Size of Conv 4 in mini-batch: " + str(layer_op4.shape) + "\n")

        output = ((((model._modules.items())[0])[1]).module)[10](output)
        output = ((((model._modules.items())[0])[1]).module)[11](output)
        output = ((((model._modules.items())[0])[1]).module)[12](output) # Gives Maxppol 5 as output
        layer_op5 = output.cpu().data.numpy(); # Convert the Conv5 output to Numpy array

        # Flatten the output of the Maxpool 5 Layer for processing by the FC Layers
        flt_layer_op5 = np.reshape(layer_op5, (layer_op5.shape[0], layer_op5.shape[1] * layer_op5.shape[2] * layer_op5.shape[3]))
        tmp = np.zeros(( (layer_op5.shape)[0], (layer_op5.shape)[1] ))
        # Loop over every feature map of every sample
        for samp in range(tmp.shape[0]):
            for fm in range(tmp.shape[1]):
                tmp[samp, fm] = np.amax(layer_op5[samp, fm, :, :]) # Extract the maximum response from every feature map
        layer_op5 = tmp
        #layer_op5 = np.mean(layer_op5, axis = 1)
        #print("Size of Maxpool 5 in mini-batch: " + str(layer_op5.shape) + "\n")

        #print("Size of the Flattened Maxpool 5 in mini-batch: " + str(flt_layer_op5.shape) + "\n")
        output = torch.autograd.Variable(torch.from_numpy(flt_layer_op5).float().cuda(), volatile=True)
        #output = model.features(input_var.cuda()) # Re-Initialize for the Classifier stage
        #output = (((model._modules.items())[1])[1])[0](output) 
        output = (((model._modules.items())[1])[1])[1](output) 
        output = (((model._modules.items())[1])[1])[2](output) # Gives FC1 as output
        layer_op6 = output.cpu().data.numpy() # Convert the FC1 output to Numpy array
        #print("Size of FC1 in mini-batch: " + str(layer_op6.shape) + "\n")

        #output = (((model._modules.items())[1])[1])[3](output)
        output = (((model._modules.items())[1])[1])[4](output)  
        output = (((model._modules.items())[1])[1])[5](output) # Gives FC2 as output
        layer_op7 = output.cpu().data.numpy() # Convert the FC2 output to Numpy array
        #print("Size of FC2 in mini-batch: " + str(layer_op7.shape) + "\n")

        #"""
        for j in range((layer_op7.shape)[0]): # For each sample in the batch
        	if op[j] == prev_label: # Check if the current sample also belongs to the same class as the previous sample
        		fc2_arr = np.vstack([fc2_arr, layer_op7[j, :].reshape(1, (layer_op7.shape)[1])]) # Append to the FC 7 responses
        		fc1_arr = np.vstack([fc1_arr, layer_op6[j, :].reshape(1, (layer_op6.shape)[1])]) # Append to the FC 6 responses
        		mp5_arr = np.vstack([mp5_arr, layer_op5[j, :].reshape(1, (layer_op5.shape)[1])]) # Append to the Conv 5 responses
        		conv4_arr = np.vstack([conv4_arr, layer_op4[j, :].reshape(1, (layer_op4.shape)[1])]) # Append to the Conv 4 responses
        		conv3_arr = np.vstack([conv3_arr, layer_op3[j, :].reshape(1, (layer_op3.shape)[1])]) # Append to the Conv 3 responses
        		mp2_arr = np.vstack([mp2_arr, layer_op2[j, :].reshape(1, (layer_op2.shape)[1])]) # Append to the Conv 2 responses
        		mp1_arr = np.vstack([mp1_arr, layer_op1[j, :].reshape(1, (layer_op1.shape)[1])]) # Append to the Conv 1 responses
    		elif (op[j] != prev_label) and (prev_label != -1): # A new class begins
        		fldr = op_fldr_path + 'Class_' + str(prev_label)
        		fldr_dir = op_fldr_path_dir + 'Class_' + str(prev_label)
        		os.system('mkdir ' + fldr_dir) # Create a new folder for the class
        		print("Size of FC2 in Class: " + str(fc2_arr.shape) + "\n")
        		with open( fldr + '/Op_FC2.pickle', 'wb') as f: # Save the ouput information for FC 7 Layer
        			pickle.dump(fc2_arr, f)
        		print("Size of FC1 in Class: " + str(fc1_arr.shape) + "\n")
        		with open( fldr + '/Op_FC1.pickle', 'wb') as f: # Save the ouput information for FC 6 Layer
        			pickle.dump(fc1_arr, f)
        		print("Size of Conv5 in Class: " + str(mp5_arr.shape) + "\n")
        		with open( fldr + '/Op_Conv5.pickle', 'wb') as f: # Save the ouput information for Conv 5 Layer
        			pickle.dump(mp5_arr, f)
        		print("Size of Conv4 in Class: " + str(conv4_arr.shape) + "\n")
        		with open( fldr + '/Op_Conv4.pickle', 'wb') as f: # Save the ouput information for Conv 4 Layer
        			pickle.dump(conv4_arr, f)
        		print("Size of Conv3 in Class: " + str(conv3_arr.shape) + "\n")
        		with open( fldr + '/Op_Conv3.pickle', 'wb') as f: # Save the ouput information for Conv 3 Layer
        			pickle.dump(conv3_arr, f)
        		print("Size of Conv2 in Class: " + str(mp2_arr.shape) + "\n")
        		with open( fldr + '/Op_Conv2.pickle', 'wb') as f: # Save the ouput information for Conv 2 Layer
        			pickle.dump(mp2_arr, f)
        		print("Size of Conv1 in Class: " + str(mp1_arr.shape) + "\n")
        		with open( fldr + '/Op_Conv1.pickle', 'wb') as f: # Save the ouput information for Conv 1 Layer
        			pickle.dump(mp1_arr, f)
        		fc2_arr = layer_op7[j, :].reshape(1, (layer_op7.shape)[1])
        		fc1_arr = layer_op6[j, :].reshape(1, (layer_op6.shape)[1])
        		mp5_arr = layer_op5[j, :].reshape(1, (layer_op5.shape)[1])
        		conv4_arr = layer_op4[j, :].reshape(1, (layer_op4.shape)[1])
        		conv3_arr = layer_op3[j, :].reshape(1, (layer_op3.shape)[1])
        		mp2_arr = layer_op2[j, :].reshape(1, (layer_op2.shape)[1])
        		mp1_arr = layer_op1[j, :].reshape(1, (layer_op1.shape)[1])
        		prev_label = op[j]
        	elif (op[j] != prev_label) and (prev_label == -1): # Very first class and first sample
        		fc2_arr = layer_op7[j, :].reshape(1, (layer_op7.shape)[1])
        		fc1_arr = layer_op6[j, :].reshape(1, (layer_op6.shape)[1])
        		mp5_arr = layer_op5[j, :].reshape(1, (layer_op5.shape)[1])
        		conv4_arr = layer_op4[j, :].reshape(1, (layer_op4.shape)[1])
        		conv3_arr = layer_op3[j, :].reshape(1, (layer_op3.shape)[1])
        		mp2_arr = layer_op2[j, :].reshape(1, (layer_op2.shape)[1])
        		mp1_arr = layer_op1[j, :].reshape(1, (layer_op1.shape)[1])
        		prev_label = op[j]
        #"""
        output = (((model._modules.items())[1])[1])[6](output) # Gives the Linear Combination as output
        # output = model(input_var)  # Final Predictions for the Sample
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

    # Write the final class
    fldr = op_fldr_path + 'Class_' + str(prev_label)
    fldr_dir = op_fldr_path_dir + 'Class_' + str(prev_label)
    os.system('mkdir ' + fldr_dir) # Create a new folder for the class
    print("Size of FC2 in Class: " + str(fc2_arr.shape) + "\n")
    with open( fldr + '/Op_FC2.pickle', 'wb') as f: # Save the ouput information for FC 7 Layer
    	pickle.dump(fc2_arr, f)
    print("Size of FC1 in Class: " + str(fc1_arr.shape) + "\n")
    with open( fldr + '/Op_FC1.pickle', 'wb') as f: # Save the ouput information for FC 6 Layer
    	pickle.dump(fc1_arr, f)
    print("Size of Conv5 in Class: " + str(mp5_arr.shape) + "\n")
    with open( fldr + '/Op_Conv5.pickle', 'wb') as f: # Save the ouput information for Conv 5 Layer
    	pickle.dump(mp5_arr, f)
    print("Size of Conv4 in Class: " + str(conv4_arr.shape) + "\n")
    with open( fldr + '/Op_Conv4.pickle', 'wb') as f: # Save the ouput information for Conv 4 Layer
    	pickle.dump(conv4_arr, f)
    print("Size of Conv3 in Class: " + str(conv3_arr.shape) + "\n")
    with open( fldr + '/Op_Conv3.pickle', 'wb') as f: # Save the ouput information for Conv 3 Layer
    	pickle.dump(conv3_arr, f)
    print("Size of Conv2 in Class: " + str(mp2_arr.shape) + "\n")
    with open( fldr + '/Op_Conv2.pickle', 'wb') as f: # Save the ouput information for Conv 2 Layer
    	pickle.dump(mp2_arr, f)
    print("Size of Conv1 in Class: " + str(mp1_arr.shape) + "\n")
    with open( fldr + '/Op_Conv1.pickle', 'wb') as f: # Save the ouput information for Conv 1 Layer
    	pickle.dump(mp1_arr, f)

    return top1.avg


def save_checkpoint(state, is_best, save_path = './', filename='checkpoint.pth.tar'):
    torch.save(state, save_path + filename)
    if is_best:
        shutil.copyfile(save_path + filename, save_path + 'model_best.pth.tar')


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
