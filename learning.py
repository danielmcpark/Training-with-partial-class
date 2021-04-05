import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        tot_correct = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            tot_correct.append(correct_k)
    return tot_correct

def train(model, loss_fn, optimizer, epoch, loader):
    model.train()
    len_data = 0
    batch_grads = torch.zeros([1, 580])
    for batch_idx, (data, targets) in enumerate(loader):
        outputs = model(data)

        loss = loss_fn(outputs, targets)
        batch_n_grad = optimizer.look_backward(loss)
        batch_grads = torch.cat([batch_grads, torch.FloatTensor(batch_n_grad).unsqueeze(0)], dim=0)
        optimizer.step()

        len_data += len(data)
        show_metrics = {}
        show_metrics['Loss_s'] = loss.item()
        if batch_idx % 1 == 0:
            front = '[Epoch_{}] Train [{}/{} ({:.0f}%)]\t'.format(epoch, len_data,
                len(loader.dataset), 100. * batch_idx / len(loader))
            end = "\t".join("{}: {:.3f}".format(k, v) for k, v in show_metrics.items())
            print(front+end)
    return show_metrics, batch_grads[1:]

def test(model, epoch, loader):
    model.eval()
    corr = 0
    len_data = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            outputs = model(data)

            corr += accuracy(outputs.data, targets.data, topk=(1,))[0]

            len_data += len(data)
            show_metrics = {}
            show_metrics['Acc_s'] = corr
            if batch_idx % 78 == 0 and batch_idx != 0:
                front = '[Epoch_{}] Test [{}/{} ({:.0f}%)]\t'.format(epoch,
                        len_data, len(loader.dataset),
                        100. * batch_idx / len(loader))
                end = "\t".join("{}: {:.3f}".format(k, (torch.mul(v, 100/len(loader.dataset))).item()) \
                        for k, v in show_metrics.items())
                print(front+end)
    return show_metrics
