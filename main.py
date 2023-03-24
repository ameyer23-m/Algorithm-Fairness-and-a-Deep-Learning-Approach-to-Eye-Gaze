import math, shutil, os, time, argparse
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from ITrackerData import ITrackerData
from ITrackerModel import ITrackerModel

# Additional Files for the Algorithm Fairness metrics
import pandas as pd 
from collections import defaultdict

# To run: 
    # Train: python main.py --data_path 'Location of metadata file' --reset
    # Train: python main.py --data_path 'Location of metadata file' --sink

# ----------------- My Additions to test GPU ---------------------------------
# Checks if GPU is available
# import torch

# # Check if PyTorch is using a GPU
# if torch.cuda.is_available():
#     print("PyTorch is using a GPU")
# else:
#     print("PyTorch is not using a GPU")

# # Print the number of GPUs available
# print(f"Number of GPUs available: {torch.cuda.device_count()}")

# from tensorflow.python.client import device_lib

# def get_available_gpus(x):
#     local_device_protos = device_lib.list_local_devices()
#     x = False
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']


# x = True
# if x == True:
#     get_available_gpus(x)

#-----------------------------------------------------------------------------

'''
Train/test code for iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='iTracker-pytorch-Trainer.')
parser.add_argument('--data_path', help="Path to processed dataset. It should contain metadata.mat. Use prepareDataset.py.")
parser.add_argument('--sink', type=str2bool, nargs='?', const=True, default=False, help="Just sink and terminate.")
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False, help="Start from scratch (do not load).")
args = parser.parse_args()

# Change there flags to control what happens.
doLoad = not args.reset # Load checkpoint at the beginning
doTest = args.sink # Only run test, no training

workers = 16

# For my study I used 5 epochs, which took 3 days to run, due to time constraints. For more accurate results change back to 25
epochs = 5 



batch_size = torch.cuda.device_count()*100 

base_lr = 0.0001
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 1e20
lr = base_lr

count_test = 0
count = 0




def main():
    global args, best_prec1, weight_decay, momentum

    model = ITrackerModel()
    model = torch.nn.DataParallel(model)
    
    # Add the model to be used in the GPU
    model.cuda()  
    imSize=(224,224)
    cudnn.benchmark = True   

    epoch = 0
    if doLoad:
        saved = load_checkpoint()
        if saved:
            print('Loading checkpoint for epoch %05d with loss %.5f (which is the mean squared error not the actual linear error)...' % (saved['epoch'], saved['best_prec1']))
            state = saved['state_dict']
            try:
                model.module.load_state_dict(state)
            except:
                model.load_state_dict(state)
            epoch = saved['epoch']
            best_prec1 = saved['best_prec1']
        else:
            print('Warning: Could not read checkpoint!')

    
    dataTrain = ITrackerData(dataPath = args.data_path, split='train', imSize = imSize)
    dataVal = ITrackerData(dataPath = args.data_path, split='test', imSize = imSize)
   
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataVal,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)



    criterion = nn.MSELoss() #.conda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    # Quick test
    if doTest:
        validate(val_loader, model, criterion, epoch)
        return

    for epoch in range(0, epoch):
        adjust_learning_rate(optimizer, epoch)
    
    model.to(torch.device("cuda"))        
    
    for epoch in range(epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)
        # print('Check on prec1: ', prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion,optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # Added the participant information
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, participant) in enumerate(train_loader):   
        
        # measure data loading time
        data_time.update(time.time() - end)

        # Add all images and information to the cuda to be used on GPU
        imFace = imFace.cuda()  
        imEyeL = imEyeL.cuda()      
        imEyeR = imEyeR.cuda()       
        faceGrid = faceGrid.cuda()      
        gaze = gaze.cuda()             
        
        imFace = torch.autograd.Variable(imFace, requires_grad = True)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = True)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = True)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = True)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)
            
        # compute output
        output = model(imFace, imEyeL, imEyeR, faceGrid)

        loss = criterion(output, gaze)
        
        losses.update(loss.data.item(), imFace.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        count=count+1

        print('Epoch (train): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss_val:.4f} ({loss_avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_val=losses.val, loss_avg=losses.avg))




def validate(val_loader, model, criterion, epoch):
    global count_test
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesLin = AverageMeter()

    # Added the following metrics to calculate algorithm fairness
    male_losses = AverageMeter() 
    male_lossesLin = AverageMeter() 
    female_losses = AverageMeter()
    female_lossesLin = AverageMeter() 
    black_losses = AverageMeter()
    black_lossesLin = AverageMeter()
    ea_losses = AverageMeter()
    ea_lossesLin = AverageMeter()
    i_losses = AverageMeter()
    i_lossesLin = AverageMeter()
    latino_losses = AverageMeter()
    latino_lossesLin = AverageMeter()
    mid_eastern_losses = AverageMeter()
    mid_eastern_lossesLin = AverageMeter()
    swa_losses = AverageMeter()
    swa_lossesLin = AverageMeter()
    white_losses = AverageMeter()
    white_lossesLin = AverageMeter()
    
    

    # switch to evaluate mode
    model.eval()
    end = time.time()


    oIndex = 0
    # Added the participant data
    for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze, participant) in enumerate(val_loader): 
        # measure data loading time
        data_time.update(time.time() - end)
        # Add data to the CUDA to run with GPU
        imFace = imFace.cuda()         
        imEyeL = imEyeL.cuda()       
        imEyeR = imEyeR.cuda()        
        faceGrid = faceGrid.cuda()    
        gaze = gaze.cuda()         
        

        
        imFace = torch.autograd.Variable(imFace, requires_grad = False)
        imEyeL = torch.autograd.Variable(imEyeL, requires_grad = False)
        imEyeR = torch.autograd.Variable(imEyeR, requires_grad = False)
        faceGrid = torch.autograd.Variable(faceGrid, requires_grad = False)
        gaze = torch.autograd.Variable(gaze, requires_grad = False)


        # Additions made to calculate algorithm fairness
        # import demographics csv
        df = pd.read_csv(r"E:\GazeCapture_demographics.csv")

        df['participant'] = df['participant'].astype(str)
        df['participant'] = df['participant'].str.zfill(5)


        count_dict = {}
        # find the most frequant participant
        for item in participant:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1
        most_frequent_value = max(count_dict, key=count_dict.get)

        # add the batch's results to the specified race and gender
        for index, row in df.iterrows():
            if row['participant'] == most_frequent_value:
                gender = row['gender']
                race = row['race']
                break
        else:
            print("No match found.")

        print(most_frequent_value, gender, race)

        # compute output
        with torch.no_grad():
            output = model(imFace, imEyeL, imEyeR, faceGrid)

        # Creates metics for gender variables 
        if gender == 'Male':
            male_loss = criterion(output, gaze)
            male_lossLin = output - gaze
            male_lossLin = torch.mul(male_lossLin,male_lossLin)
            male_lossLin = torch.sum(male_lossLin,1)
            male_lossLin = torch.mean(torch.sqrt(male_lossLin))

            male_losses.update(male_loss.data.item(), imFace.size(0))
            male_lossesLin.update(male_lossLin.item(), imFace.size(0))
        else:
            female_loss = criterion(output, gaze)
            female_lossLin = output - gaze
            female_lossLin = torch.mul(female_lossLin,female_lossLin)
            female_lossLin = torch.sum(female_lossLin,1)
            female_lossLin = torch.mean(torch.sqrt(female_lossLin))

            female_losses.update(female_loss.data.item(), imFace.size(0))
            female_lossesLin.update(female_lossLin.item(), imFace.size(0))


        # Creates metics for race variables 
        if race == 'Black':
            black_loss = criterion(output, gaze)
            black_lossLin = output - gaze
            black_lossLin = torch.mul(black_lossLin,black_lossLin)
            black_lossLin = torch.sum(black_lossLin,1)
            black_lossLin = torch.mean(torch.sqrt(black_lossLin))

            black_losses.update(black_loss.data.item(), imFace.size(0))
            black_lossesLin.update(black_lossLin.item(), imFace.size(0))
        elif race == 'East Asian':
            ea_loss = criterion(output, gaze)
            ea_lossLin = output - gaze
            ea_lossLin = torch.mul(ea_lossLin,ea_lossLin)
            ea_lossLin = torch.sum(ea_lossLin,1)
            ea_lossLin = torch.mean(torch.sqrt(ea_lossLin))

            ea_losses.update(ea_loss.data.item(), imFace.size(0))
            ea_lossesLin.update(ea_lossLin.item(), imFace.size(0))
        elif race == 'Indian':
            i_loss = criterion(output, gaze)
            i_lossLin = output - gaze
            i_lossLin = torch.mul(i_lossLin,i_lossLin)
            i_lossLin = torch.sum(i_lossLin,1)
            i_lossLin = torch.mean(torch.sqrt(i_lossLin))

            i_losses.update(i_loss.data.item(), imFace.size(0))
            i_lossesLin.update(i_lossLin.item(), imFace.size(0))
        elif race == 'Latino_Hispanic':
            latino_loss = criterion(output, gaze)
            latino_lossLin = output - gaze
            latino_lossLin = torch.mul(latino_lossLin,latino_lossLin)
            latino_lossLin = torch.sum(latino_lossLin,1)
            latino_lossLin = torch.mean(torch.sqrt(latino_lossLin))

            latino_losses.update(latino_loss.data.item(), imFace.size(0))
            latino_lossesLin.update(latino_lossLin.item(), imFace.size(0))
        elif race == 'Middle Eastern':
            mid_eastern_loss = criterion(output, gaze)
            mid_eastern_lossLin = output - gaze
            mid_eastern_lossLin = torch.mul(mid_eastern_lossLin,mid_eastern_lossLin)
            mid_eastern_lossLin = torch.sum(mid_eastern_lossLin,1)
            mid_eastern_lossLin = torch.mean(torch.sqrt(mid_eastern_lossLin))

            mid_eastern_losses.update(mid_eastern_loss.data.item(), imFace.size(0))
            mid_eastern_lossesLin.update(mid_eastern_lossLin.item(), imFace.size(0))
        elif race == 'Southeast Asian':
            swa_loss = criterion(output, gaze)
            swa_lossLin = output - gaze
            swa_lossLin = torch.mul(swa_lossLin,swa_lossLin)
            swa_lossLin = torch.sum(swa_lossLin,1)
            swa_lossLin = torch.mean(torch.sqrt(swa_lossLin))

            swa_losses.update(swa_loss.data.item(), imFace.size(0))
            swa_lossesLin.update(swa_lossLin.item(), imFace.size(0))
        else:
            white_loss = criterion(output, gaze)
            white_lossLin = output - gaze
            white_lossLin = torch.mul(white_lossLin,white_lossLin)
            white_lossLin = torch.sum(white_lossLin,1)
            white_lossLin = torch.mean(torch.sqrt(white_lossLin))

            white_losses.update(white_loss.data.item(), imFace.size(0))
            white_lossesLin.update(white_lossLin.item(), imFace.size(0))

        loss = criterion(output, gaze)
        lossLin = output - gaze
        lossLin = torch.mul(lossLin,lossLin)
        lossLin = torch.sum(lossLin,1)
        lossLin = torch.mean(torch.sqrt(lossLin))

        losses.update(loss.data.item(), imFace.size(0))
        lossesLin.update(lossLin.item(), imFace.size(0))
     
        # compute gradient and do SGD step
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

       

        print('Epoch (val): [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error L2 {lossLin.val:.4f} ({lossLin.avg:.4f})\n'.format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss=losses, lossLin=lossesLin))

        print('\t\tGender Fairness Scores: '
                'Male ({male_loss_avg:.4f})\t'
                'Female ({female_loss_avg:.4f})\t'
                'Male std ({male_std:.4f})\t'
                'Female std ({female_std:.4f})\t'
                'Male size ({male_size:.4f})\t'
                'Female size ({female_size:.4f})\n'
                .format(
                   male_loss_avg=male_lossesLin.avg, female_loss_avg=female_lossesLin.avg,
                   male_std = male_lossesLin.std, female_std = female_lossesLin.std,
                   male_size = male_lossesLin.count, female_size = female_lossesLin.count))

        print('\t\tRace Fairness Scores: '
                'White ({white_loss_avg:.4f})\t'
                'Black ({black_loss_avg:.4f})\t'
                'EA ({ea_loss_avg:.4f})\t'
                'SWA ({swa_loss_avg:.4f})\t'
                'I ({i_loss_avg:.4f})\t'
                'me ({me_loss_avg:.4f})\t'
                'lat ({latino_loss_avg:.4f})\t'
                'White std ({white_std:.4f})\t'
                'Black std ({black_std:.4f})\t'
                'EA std ({ea_std:.4f})\t'
                'SWA std ({swa_std:.4f})\t'
                'I std ({i_std:.4f})\t'
                'me std ({me_std:.4f})\t'
                'lat std ({latino_std:.4f})\t'
                'White size ({white_size:.4f})\t'
                'Black size ({black_size:.4f})\t'
                'EA size ({ea_size:.4f})\t'
                'SWA size ({swa_size:.4f})\t'
                'I size ({i_size:.4f})\t'
                'me size ({me_size:.4f})\t'
                'lat size ({latino_size:.4f})\n\n'
                .format(
                   white_loss_avg=white_lossesLin.avg, black_loss_avg=black_lossesLin.avg,
                   ea_loss_avg=ea_lossesLin.avg, swa_loss_avg=swa_lossesLin.avg,
                   i_loss_avg=i_lossesLin.avg, me_loss_avg = mid_eastern_lossesLin.avg, latino_loss_avg=latino_lossesLin.avg,
                   white_std = white_lossesLin.std, black_std = black_lossesLin.std, ea_std = ea_lossesLin.std, swa_std = swa_lossesLin.std, i_std = i_lossesLin.std, me_std = mid_eastern_lossesLin.std, latino_std = latino_lossesLin.std,
                   white_size = white_lossesLin.count, black_size = black_lossesLin.count, ea_size = ea_lossesLin.count, swa_size = swa_lossesLin.count, i_size = i_lossesLin.count, me_size = mid_eastern_lossesLin.count, latino_size = latino_lossesLin.count))
        

    return lossesLin.avg

CHECKPOINTS_PATH = '.'

def load_checkpoint(filename='checkpoint.pth.tar'):
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    print(filename)
    if not os.path.isfile(filename):
        return None
    state = torch.load(filename, map_location = torch.device('cpu'))
    return state

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.isdir(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH, 0o777)
    bestFilename = os.path.join(CHECKPOINTS_PATH, 'best_' + filename)
    filename = os.path.join(CHECKPOINTS_PATH, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestFilename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    # Added information to calculate standard deviation
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.se = 0
        self.std = 0
        
    # Added information to calculate standard deviation
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.se += (val - self.avg)**2 * n
        self.std = math.sqrt(self.se / (self.count - 1)) if self.count > 1 else 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
    print('DONE')