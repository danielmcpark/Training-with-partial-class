import argparse
import inspect
import pandas as pd
import os, sys
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable
from torchvision import datasets, transforms

import models
import learning
from grad import LookUpGrad

torch.manual_seed(0)
torch.set_deterministic(True)

parser = argparse.ArgumentParser(description='MCPARK Gradient Pruning Skeleton')
parser.add_argument('--arch', type=str, default='LeNet_5', metavar='string')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='string')
parser.add_argument('--epochs', type=int, default=2, metavar='int')
parser.add_argument('--batch-size', type=int, default=128, metavar='int')
parser.add_argument('--save', type=str, default='./logs/', metavar='string')
parser.add_argument('--idx', type=int, default=1, metavar='int')

args = parser.parse_args()

trainset = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3801))
                          ]))

data_classes = [i for i in range(10)] # MNIST
locals()['train_subset_per_class_{}'.format(args.idx)] = list()
for j in range(len(trainset)):
    if int(trainset[j][1]) == args.idx:
        locals()['train_subset_per_class_{}'.format(args.idx)].append(j)
locals()['trainset_{}'.format(args.idx)] = torch.utils.data.Subset(trainset,
                                           locals()['train_subset_per_class_{}'.format(args.idx)])

train_loader = torch.utils.data.DataLoader(locals()['trainset_{}'.format(args.idx)],
                                           batch_size=args.batch_size,
                                           shuffle=True
                                           )

test_loader = torch.utils.data.DataLoader(datasets.MNIST(
                '../data/', train=False, download=True,
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=128, shuffle=False)

def set_argument(sig, args):
    model_keys = list(sig.parameters.keys())
    kwargs = {}
    if len(model_keys) == 1:
        return kwargs

    for idx in range(len(model_keys[1:])):
        if model_keys[1:][idx] in list(vars(args).keys()):
            kwargs[model_keys[1:][idx]] = vars(args)[model_keys[1:][idx]]
    return kwargs

sig = inspect.signature(models.__dict__[args.arch].__init__)
kwargs = set_argument(sig, args)
model = models.__dict__[args.arch](**kwargs)

loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
GradProfiler = LookUpGrad(optimizer=optimizer)
scheduler = optim.lr_scheduler.MultiStepLR(GradProfiler._optim, milestones=[15, 30, 45], gamma=0.1)

train_loss = list()
test_acc = list()
best_prec = 0.
#grads_pool = torch.zeros([1, 410])
grads_pool = torch.zeros([1, 580])

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

for epoch in range(1, args.epochs):
    print('Learning rate: {}'.format(scheduler.optimizer.param_groups[0]['lr']))
    train_metrics, b_grad = learning.train(model, loss, GradProfiler, epoch, train_loader)
    test_metrics = learning.test(model, epoch, test_loader)

    grads_pool = torch.cat([grads_pool, b_grad], dim=0)
    test_acc_ = test_metrics['Acc_s'].item() * 100/len(test_loader.dataset)
    train_loss.append(train_metrics['Loss_s'])
    test_acc.append(test_acc_)
    
    is_best = test_acc_ > best_prec
    best_prec = max(test_acc_, best_prec)
    
    save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec
        }, is_best, filepath=args.save)
    
    print("Best accuracy: "+str(best_prec))
    scheduler.step()
    torch.save(grads_pool[1:], os.path.join(args.save, '{}-class_grads.pth.tar'.format(args.idx)))
