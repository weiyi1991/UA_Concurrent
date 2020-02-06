import argparse
import os
import torch
import torch.nn.functional as F
from model_ST import *
import data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import sys
from predict import evaluate_MA
from tensorboardX import SummaryWriter

# print model parameter
def print_model(model):
    print('=================== Print model parameters ================')
    print(model.state_dict().keys())
    for i, j in model.named_parameters():
        print(i)
        print(j)

# Training settings
parser = argparse.ArgumentParser(description='Relation network for concurrent activity detection')
parser.add_argument('--BATCH_SIZE', type=int, default=256, help='Training batch size. Default=256')
parser.add_argument('--save_every', type=int, default=5, help='Save model every save_every epochs. Defualt=5')
parser.add_argument('--EPOCH', type=int, default=500, help='Number of epochs to train. Default=600')
parser.add_argument('--LR', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--TRAIN', action='store_true', default=True, help='Train or test? ')
parser.add_argument('--DEBUG', action='store_true', default=False, help='Debug mode (load less data)? Defualt=False')
parser.add_argument('--clip_grad', type=float, default=5.0, help='Gradient clipping parameter. Default=5,0')
parser.add_argument('--dataPath', type=str, default='/home/yi/PycharmProjects/relation_network/data/UCLA/new273',
                    help='path to the data folder')
parser.add_argument('--checkpoint', type=str, help='Checkpoint folder name under ./model/')
parser.add_argument('--verbose', type=int, default=1, help='Print verbose information? Default=True')
# model parameters
parser.add_argument('--n_input', type=int, default=37, help='Input feature vector size. Default=37')
parser.add_argument('--n_hidden', type=int, default=128, help='Hidden units for LSTM baseline. Default=128')
parser.add_argument('--n_layers', type=int, default=2, help='LSTM layer number. Default=2')
parser.add_argument('--n_class', type=int, default=12, help='Class label number. Default=12')
parser.add_argument('--use_lstm', action='store_true', default=True, help='Use LSTM for relation network classifier. Default=True')
parser.add_argument('--df', type=int, default=64, help='Relation feature dimension. Default=64')
parser.add_argument('--dk', type=int, default=8, help='Key feature dim. Default=8')
parser.add_argument('--nr', type=int, default=4, help='Multihead number. Default=4')
opt = parser.parse_args()

checkpoint_dir = './model/{}/'.format(opt.checkpoint)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
orig_stdout = sys.stdout
f = open(checkpoint_dir + '/parameter.txt', 'w')
sys.stdout = f
print(opt)
f.close()
sys.stdout = orig_stdout

# data preparation
train_dataset = data.ConActDataset(opt.dataPath)
test_dataset = data.ConActDataset(opt.dataPath, train=not opt.TRAIN)

writer = SummaryWriter()
# only take few sequences for debuging
debug_seq = 3
if opt.DEBUG:
    train_data = []
    for i in range(debug_seq):
        input, labels = train_dataset[i]
        train_data.append((input, labels))
        print("%s loaded." % train_dataset.seq_list[i])
else:
    print('Loading training data ----------------------')
    train_data = []
    train_labels = []
    for i, (input, labels) in enumerate(train_dataset):
        train_data.append((input, labels))
        train_labels.append(labels)
        print("%s loaded." % train_dataset.seq_list[i])

    print('Loading testing data ----------------------')
    test_data = []
    for i, (input, labels) in enumerate(test_dataset):
        test_data.append((input, labels))
        print("%s loaded." % test_dataset.seq_list[i])

# for model_lstm
if opt.use_lstm:
    rnn = RNN(opt.n_input, opt.n_hidden, opt.n_layers, opt.n_class, opt.BATCH_SIZE, opt.df, opt.dk, opt.nr).cuda()  # use lstm as classifier
else:
    rnn = RNN(opt.n_input, opt.n_hidden, opt.n_layers, opt.n_class, opt.use_lstm).cuda()  # use fc as classifier
print(rnn.state_dict().keys())

optimizer = torch.optim.Adam(rnn.parameters(), lr=opt.LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler

# Keep track of losses for plotting
best_loss = 10000
all_losses = []
current_loss = 3
FAA = []  # false area ration on test set
INTAP = [] # overall interval AP on test set
save_epoch = [] # list to save the model saving epoch
# train model
total_step = len(train_data)
for epoch in range(opt.EPOCH):
    all_losses.append(current_loss)
    current_loss = 0


    for i, (input, labels) in enumerate(train_data):
        optimizer.zero_grad()

        feats = torch.from_numpy(input).float()
        nframes, _ = input.shape
        feats = feats.reshape(-1, nframes, 273).cuda()
        #feats = feats.reshape(-1, nframes, opt.n_input*6).cuda()
        # change label 0 to -1
        labels[labels<1]=-1
        labels = torch.from_numpy(labels)
        labels = labels.float().cuda()

        # Forward pass
        outputs = rnn(feats)
        outputs = torch.squeeze(outputs)

        loss = F.mse_loss(outputs, labels)
        # print model parameter if loss is NaN
        if opt.verbose > 0:
            if torch.isnan(loss):
                print_model(rnn)
                print('Epoch {}, step {}'.format(epoch+1, i+1))
                raw_input("Press Enter to continue ...")

        # Backward and optimize
        loss.backward()
        # This line is used to prevent the vanishing / exploding gradient problem
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), opt.clip_grad)
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch + 1, opt.EPOCH, i + 1, total_step, loss.item()))
        current_loss = current_loss + loss.item()

    writer.add_scalar('loss/loss', current_loss, epoch)
    scheduler.step(current_loss)  # update lr if needed

# save model parameters and loss figure
    if ((epoch+1) % opt.save_every) == 0:
        # compute false area on test set
        if not opt.DEBUG:
            false_area, overall_IAPlist =  evaluate_MA(rnn, test_data)
            FAA.append(torch.sum(false_area).item())
            INTAP.append(overall_IAPlist[-2]) # get the interval AP at threshold 0.8
            save_epoch.append(epoch+1)
            if FAA[-1] == min(FAA):
                # if has the minimum test error, save model
                checkpoint_dir = './model/{}/'.format(opt.checkpoint)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                if epoch > 100:
                    model_str = checkpoint_dir + 'net-best.pth'
                    torch.save(rnn, model_str)

        checkpoint_dir = './model/{}/'.format(opt.checkpoint)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            if opt.verbose == 2:
                print('Making dir: {}'.format(checkpoint_dir))
        model_str = checkpoint_dir + 'net-{}'.format(str(epoch+1))
        if opt.verbose > 0:
            print('Model saved to: {}.pth'.format(model_str))
        if epoch >= 100:
            torch.save(rnn, model_str+'.pth')
            # save interval AP
            np.savetxt(model_str + 'AP.csv', np.asarray(overall_IAPlist), fmt='%0.5f')
            # save miss detection
            np.savetxt(model_str + 'MD.txt', np.asarray(FAA), fmt='%0.5f')
            # draw miss detection v.s. epoch figure
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.plot(range(epoch+1), all_losses, color=color)
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss', color=color)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Miss detection area ratio', color=color)
            ax2.plot(save_epoch, FAA, 'bd')
            fig.savefig(model_str+'.png')
            plt.close()

            # draw intervalAP v.s. epoch figure
            fig1, ax3 = plt.subplots()
            color = 'tab:red'
            ax3.plot(range(epoch+1), all_losses, color=color)
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Loss', color=color)

            ax4 = ax3.twinx()
            color = 'tab:blue'
            ax4.set_ylabel('Overall interval AP', color=color)
            ax4.plot(save_epoch, INTAP, 'bd')
            fig1.savefig(model_str+'_AP.png')
            plt.close()

# print the loss on training set and evaluation metrics on test set to file
orig_stdout = sys.stdout
f = open(checkpoint_dir + '/loss.txt', 'w')
sys.stdout = f
print('Loss over epochs:')
print(all_losses)
if not opt.DEBUG:
    print('Miss detection area ratio:')
    print(FAA)
f.close()
sys.stdout = orig_stdout
