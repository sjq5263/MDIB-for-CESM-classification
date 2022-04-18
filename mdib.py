#coding=gbk
import numpy as np
import time
import torch
import math  # sjq add  改变了solver43里 beta的大小
import torch.nn as nn
from torch.nn.parallel import DataParallel# for multi-GPU training
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from solver43 import NetwithIB
from data_processing import Dataset
import argparse
import os
import shutil
import torch.cuda as cuda
from sklearn.metrics import roc_curve, auc

from sklearn import metrics
import matplotlib.pyplot as plt


LISTS = './list/'
TRAIN = LISTS + 'ttrain/'
TEST = LISTS + 'ttest/'
CLASSES = './classes.txt'

TRAIN_OUT = './ttrain_lists.txt'
TEST_OUT = './ttest_lists.txt'
VAL_OUT = './ttval_lists.txt'

best_prec1 = 0

train_loss = []
train_acc = []
val_loss = []
val_acc = []



parser = argparse.ArgumentParser(description='PyTorch MVCNN Training')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (# default: 0.001)')

parser.add_argument('--epochs', default=201, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

#parser.add_argument('--b', default=8, type=int,
parser.add_argument('--b', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    help='weight decay (default: 1e-4)')


# load dataset
def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    #listfiles, labels  = zip(*[(l[0], int(l[1])) for l in listfile_labels])
    listfiles, labels  = list(zip(*[(l[0], int(l[1])) for l in listfile_labels]))
    return listfiles, labels

def main():

    global args, best_prec1
    args = parser.parse_args()

    # Read list of training and validation data
    listfiles_train, labels_train = read_lists(TRAIN_OUT)  #listfiles_train = ./list/train/normal/36.txt
    listfiles_val, labels_val = read_lists(VAL_OUT)
    listfiles_test, labels_test = read_lists(TEST_OUT)
    dataset_train = Dataset(listfiles_train, labels_train, subtract_mean=True, V=4)
    dataset_val = Dataset(listfiles_val, labels_val, subtract_mean=True, V=4)
    dataset_test = Dataset(listfiles_test, labels_test, subtract_mean=True, V=4)

    # shuffle data
    dataset_train.shuffle()
    dataset_val.shuffle()
    dataset_test.shuffle()
    tra_data_size, val_data_size, test_data_size= dataset_train.size(), dataset_val.size(), dataset_test.size()     # dataset_train.size = 130, validation size: 20 , test_data_size:32
    print ('training size:', tra_data_size)
    print ('validation size:', val_data_size)
    print ('testing size:', test_data_size)

    batch_size = args.b
    print("batch_size is :" + str(batch_size)) #  batch_size
    learning_rate = args.lr
    print("learning_rate is :" + str(learning_rate))
    num_cuda = cuda.device_count()
    print("number of GPUs have been detected:"+str(num_cuda)) 

    # creat model
    print("model building...")
    mvcnn = DataParallel(NetwithIB(num_cuda, batch_size))
    mvcnn.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint'{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            mvcnn.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    #print(mvcnn)

    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adadelta(mvcnn.parameters(), weight_decay=1e-4)
    # evaluate performance only
    if args.evaluate:
        print ('------------------ testing mode ------------------')
        tests(dataset_test, mvcnn, criterion, optimizer, batch_size)      
        return

    print ('------------------ training mode ------------------')
    # for epoch in xrange(args.start_epoch, args.epochs): # yuandaima
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch:', epoch)

        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(dataset_train, mvcnn, criterion, optimizer, epoch, batch_size)

        # evaluate on validation set
        prec1 = validate(dataset_val, mvcnn, criterion, optimizer, batch_size)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
          save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': mvcnn.state_dict(),
              'best_prec1': best_prec1,
          }, is_best, epoch)
        elif epoch % 10 == 0:
          save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': mvcnn.state_dict(),
              'best_prec1': best_prec1,
          }, is_best, epoch)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    ax.plot(train_loss,'g', label='Train Loss')
    ax.plot(val_loss, 'r', label='Validation Loss')

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    fig.savefig('NewtwithIB43b4loss ', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    plt.close(fig)

    fig2 = plt.figure(1)
    ax2 = fig2.add_subplot(111)
    plt.ylim(0, 105)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    ax2.plot(train_acc , 'g', label='Train Accuracy')
    ax2.plot(val_acc, 'r', label='Validation Accuracy')

    handles, labels = ax2.get_legend_handles_labels()
    lgd = ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))

    fig2.savefig('NewtwithIBb4acc43', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
        
# train
def train(dataset_train, mvcnn, criterion, optimizer, epoch, batch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
      
    total_loss = 0.0
    total_acc = 0.0
    # switch to train mode
    mvcnn.train()

    step = 0
    end = time.time()
    for batch_x, batch_y in dataset_train.batches(batch_size):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_x, batch_y = torch.from_numpy(batch_x), torch.from_numpy(batch_y)

        batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.LongTensor)

        input, target = batch_x.cuda(), batch_y.cuda(async=True)
        # input, target = batch_x.cuda(), batch_y.cuda(async=True) YUAN DAI MA

        input_var, target_var = torch.autograd.Variable(input), torch.autograd.Variable(target)

        lossib, output_ib, loss_mvib, output = mvcnn(input_var)
        
        loss = criterion(output, target_var) # IZY_bound
        
        loss_ib = criterion(output_ib, target_var)
        
        total_lossib = loss + lossib + loss_mvib + loss_ib
        
        prec11, prec5 = accuracy(output.data, target_var, check_result=False, topk=(1,2))
        precib, prec5ib = accuracy(output_ib.data, target_var, check_result=False, topk=(1,2))
        
        prec1 = ( prec11 + precib )/ 2.
        prec5 = ( prec5 + prec5ib )/ 2.
        
        #losses.update(loss.data[0], input.size(0)) yuan dai ma
        losses.update(total_lossib.item(), input.size(0))

        total_loss += total_lossib.item()
        total_acc += top1.val
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_lossib.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print training information
        if step % 40 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, step, dataset_train.size()/batch_size, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            print('NewtwithIB_batch4', step , '   loss_mvib:', loss_mvib.item(), '   loss:', loss.item())
            print('prec1zhong42:  ', prec11.item())
            print('precib42:  ', precib.item()) 
            
        step += 1
        
        del total_lossib
        del input_var
        del target_var
        
    train_loss.append(total_loss/(dataset_train.size()/batch_size))
    train_acc.append(total_acc/(dataset_train.size()/batch_size))  

def validate(dataset_val, mvcnn, criterion, optimizer, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total_loss = 0.0
    total_acc = 0.0
    # switch to evaluate mode
    mvcnn.eval()

    for batch_x, batch_y in dataset_val.batches(batch_size):
        batch_x, batch_y = torch.from_numpy(batch_x), torch.from_numpy(batch_y)
        batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.LongTensor)
        input, target = batch_x.cuda(), batch_y.cuda(async=True)
        
        #input, target = batch_x.cuda(), batch_y.cuda(async=True) YAN DAI MA 
        input_var, target_var = torch.autograd.Variable(input, volatile=False), torch.autograd.Variable(target, volatile=False)
        
        # compute output
        lossib, output_ib, loss_mvib, output = mvcnn(input_var)
        #output = mvcnn(input_var)
        loss = criterion(output, target_var) # IZY_bound
        
        loss_ib = criterion(output_ib, target_var)
        
        total_lossib = loss + lossib + loss_mvib + loss_ib
        
        prec1, prec5 = accuracy(output.data, target_var, check_result=False, topk=(1,2))
        precib, prec5ib = accuracy(output_ib.data, target_var, check_result=False, topk=(1,2))
        prec1 = ( prec1 + precib )/ 2.
        prec5 = ( prec5 + prec5ib )/ 2.
        
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
               
        total_loss += total_lossib.item()
        total_acc += top1.val
        
        optimizer.zero_grad()
        total_lossib.backward()
        optimizer.step()
        del total_lossib
        print(' Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    val_loss.append(total_loss/(dataset_val.size()/batch_size))
    val_acc.append(total_acc/(dataset_val.size()/batch_size))
    
    return top1.avg

def tests(dataset_val, mvcnn, criterion, optimizer, batch_size):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc
    from sklearn.metrics import cohen_kappa_score,matthews_corrcoef
    import matplotlib.pyplot as plt
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    outputlist = []
    targetlist = []
    # switch to evaluate mode
    mvcnn.eval()

    for batch_x, batch_y in dataset_val.batches(batch_size):
        batch_x, batch_y = torch.from_numpy(batch_x), torch.from_numpy(batch_y)
        batch_x, batch_y = batch_x.type(torch.FloatTensor), batch_y.type(torch.LongTensor)
        input, target = batch_x.cuda(), batch_y.cuda(async=True)
        

        input_var, target_var = torch.autograd.Variable(input, volatile=False), torch.autograd.Variable(target, volatile=False)
        
        # compute output
        #output = mvcnn(input_var)
        lossib, output_ib, loss_mvib, output = mvcnn(input_var)
        #_ , output = mvcnn(input_var)
        maxk = max((1,))
        _, pred = output.data.topk(maxk, 1, True, True)
        _, predib = output_ib.data.topk(maxk, 1, True, True)
        pred = (pred + predib) / 2.
        y_pred = pred.t().cpu()
        y_true = target_var.cpu() # cong GPU ti qu dao CPU 
        outputlist.extend(y_pred[0].numpy().tolist()) # numpy(): tensor bian cheng numpy. tolist(): array bian chen list. extend:lianjie
        targetlist.extend(y_true.numpy().tolist())
        
        prec1, prec5 = accuracy(output.data, target_var, check_result=True, topk=(1,2))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        optimizer.zero_grad()
        optimizer.step()
        print(' Prec@1 {top1.avg:.3f}%'
          .format(top1=top1))
           
    print('accuracy_score:{:.3f}%'.format(accuracy_score(targetlist,outputlist)*100),'\n')
    print('precision_score:{:.3f}%'.format(precision_score(targetlist,outputlist)*100),'\n')
    print('recall_score:{:.3f}%'.format(recall_score(targetlist,outputlist)*100),'\n')
    print('f1_score:{:.4f}'.format(f1_score(targetlist,outputlist)),'\n')
    print('cohen_kappa_score:{:.4f}'.format(cohen_kappa_score(targetlist,outputlist)),'\n')
    print('matthews_corrcoef:{:.4f}'.format(matthews_corrcoef(targetlist,outputlist)),'\n')
    fpr,tpr,threshold = roc_curve(outputlist, targetlist)
    print('specificity:{:.4f}'.format(1-fpr[1]),'\n')
    roc_auc = auc(fpr,tpr)
    print('roc_auc:{:.4f}'.format(roc_auc))
    plt.figure(figsize=(5, 5))  
    plt.title('ROC Curve')       
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.plot(fpr, tpr, color='darkorange',
         lw=2, label="ROC curve"+'(area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.savefig('NewtwithIB43rocb4_auc.png')
    return top1.avg

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
        self.sum += val * n  #   val*batch_size
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target_var, check_result = False, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    batch_size = 4

    maxk = max((1,))
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
   
    if check_result is True:
        print("Prediction:\n")
        print(pred)
        print("Ground_Truth:\n")
        print(target_var.unsqueeze(0).permute(0,1).contiguous())  # unsqueeze zai 0 wei zengjia weidu. permute diao huan wei du. contiguous ba tensor bian cheng nei cun zhong lian xu fen bu de xing shi
    correct = pred.eq(target_var.data.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, epoch, filename='_epoch01_NewtwithIB4.pth.tar'):
    torch.save(state, "model"+str(epoch)+filename)
    if is_best:
        shutil.copyfile("model"+str(epoch)+filename, "model"+str(epoch)+'_epoch01_NewtwithIB4_best.pth.tar')

if __name__ == '__main__':

    main()
