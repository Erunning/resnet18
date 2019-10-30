from datetime import datetime
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(net, train_data, valid_data, num_epochs, optimizer, criterion):
    writer = SummaryWriter('log')
    #tensorboard --logdir=./log 进入目录，命令行输入，可以打开曲线网页
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    max_acc = 0.7
    niter = 0
    for epoch in range(num_epochs):

        net = net.train()

        for im, label in train_data:

            niter += 1
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()


            output = net(im)
            loss = criterion(output, label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if niter % 20 == 0:

                train_loss = loss.item()
                train_acc = get_acc(output, label)
                print( "Epoch %d. Iteration %d. Train Loss: %f, Train Acc: %f"% (epoch, niter, train_loss , train_acc))
                writer.add_scalar('Train/Loss', train_loss, niter)
                writer.add_scalar('Train/Acc', train_acc, niter)
            if niter % 200 == 0:
                valid_loss = 0
                valid_acc = 0
                net = net.eval()
                for im, label in valid_data:
                    if torch.cuda.is_available():
                        im = im.cuda()
                        label = label.cuda()

                    output = net(im)
                    loss = criterion(output, label)
                    valid_loss += loss.item()
                    valid_acc += get_acc(output, label)
                print("Epoch %d. Iteration %d. Test Loss: %f, Test Acc: %f" % (
                    epoch, niter , valid_loss/len(valid_data), valid_acc/len(valid_data)))
                writer.add_scalar('Test/Loss', valid_loss/len(valid_data), niter)
                writer.add_scalar('Test/Acc', valid_acc/len(valid_data), niter)
                if valid_acc > max_acc:
                    max_acc = valid_acc
                    torch.save(net.state_dict(),
                               './model/model_raw_{0}_{1:.4f}.pkl'.format(niter, valid_acc / len(valid_data)))
            # model = TheModelClass(*args, **kwargs)
            # model.load_state_dict(torch.load(PATH))
            # model.eval()

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print(time_str)
        prev_time = cur_time



