import torch
import data
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from eval import interval_AP


def miss_area(pred, label):
    '''
    Compute the miss detected area for each action class of one sequence
    :param pred: n*d array with value [0,1], n - number of frames, d - number of classes
    :param label: n*d binary (0 or 1) array as the ground truth label, same size as prediction.
    :return:
        miss_area: area ratio of miss detection for each action class.
        frm_num: frame numbers of current sequence
    '''
    frm_num, class_num = pred.size()
    miss_area = torch.zeros(class_num)

    for i in range(class_num):
        fp = 1 - pred[:,i][label[:,i] > 0.9]
        tn = pred[:,i][label[:,i] < 0.1]
        miss_area[i] = torch.sum(fp) + torch.sum(tn)
    return miss_area, frm_num

def INTERAP(output_list, label_list):
    interAP = []
    overall_IAPlist = []
    for theta in np.arange(0, 1, 0.1):
        inter_AP, overall_IAP = interval_AP(output_list, label_list, theta)
        interAP.append(inter_AP)
        overall_IAPlist.append(overall_IAP)
    return interAP, overall_IAPlist

def evaluate_MA(rnn, test_data):
    '''
    compute the overall miss-detected area ration of all classes over all sequences
    compute the interval AP and overall intervalAP of all classes over all sequences
    :param rnn: rnn model
    :param test_data:   testing dataset
    :return:
    '''
    output_list = []
    label_list = []
    with torch.no_grad():
        for i, (input, labels) in enumerate(test_data):
            # convert feats and labels to torch.FloatTensor to avoid type mismatch
            # refer to https://stackoverflow.com/questions/49407303/runtimeerror-expected-object-of-type-torch-doubletensor-but-found-type-torch-fl
            feats = torch.from_numpy(input).float()
            nframes, _ = input.shape
            feats = feats.reshape(-1, nframes, 273).cuda()
            # change label 0 to -1
            labels = torch.from_numpy(labels)
            labels = labels.float().cuda()

            # Forward pass
            outputs = rnn(feats)
            outputs = torch.squeeze(outputs.detach())
            # convert [-1, 1] value to [0, 1]
            ones = torch.ones_like(outputs)
            outputs = (outputs + ones) / 2

            output_list.append(outputs.detach())
            label_list.append(labels.detach())
        false_area, frm_num = miss_area(torch.cat(output_list, 0), torch.cat(label_list, 0))
        mis_det = false_area / frm_num
        interAP, overall_IAPlist = INTERAP(output_list, label_list)
    return mis_det, overall_IAPlist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict on testing set')
    parser.add_argument('--dataPath', type=str, default='/home/yi/PycharmProjects/relation_network/data/UCLA/new273',
                        help='path to the data folder')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint folder name under ./model/')
    opt = parser.parse_args()

    model_dir = os.path.join('model', opt.checkpoint)  # checkpoint path
    model_name = opt.checkpoint.split('/')[0]          # folder name for the folder
    output_dir = os.path.join('output', model_name)    # output path
    # create output folder if doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    orig_stdout = sys.stdout
    f = open(output_dir+'/out.txt', 'w')
    sys.stdout = f

    print(opt)

    test_dataset = data.ConActDataset(opt.dataPath, train=False)
    test_data = []


    for i, (input, labels) in enumerate(test_dataset):
        test_data.append((input, labels))
        print("%s loaded." % test_dataset.seq_list[i])

    L = 0
    # load trained model
    rnn = torch.load(model_dir)
    # creat list to store all the predictions and labels to compute AP
    output_list = []
    label_list = []
    for i, (input, labels) in enumerate(test_data):
        feats = torch.from_numpy(input).float()
        nframes, _ = input.shape
        feats = feats.reshape(-1, nframes, 273).cuda()
        labels = torch.from_numpy(labels)
        labels = labels.float().cuda()

        # Forward pass
        outputs = rnn(feats)
        outputs = torch.squeeze(outputs)

        # convert [-1, 1] value to [0, 1]
        ones = torch.ones_like(outputs)
        outputs = (outputs + ones) / 2

        np.savetxt(output_dir+'/'+test_dataset.seq_list[i]+'.txt', outputs.data.cpu().numpy(), fmt='%0.5f')
        loss = F.binary_cross_entropy(outputs, labels)

        print('Loss: {:.4f} at {:d}'.format(loss.item(), i))
        L = L + loss.item()

        output_list.append(outputs.detach())
        label_list.append(labels.detach())
    print(L)

    false_area, frm_num = miss_area(torch.cat(output_list, 0), torch.cat(label_list, 0))
    print('Overall miss detection area: ', false_area / frm_num)
    print('Sum of overall miss detection area: ', torch.sum(false_area / frm_num))
    sys.stdout = orig_stdout
    f.close()

    all_label = torch.cat(label_list, 0).cpu().numpy()
    all_output = torch.cat(output_list, 0).cpu().numpy()
    ap = average_precision_score(all_label > 0, all_output, average=None)
    overallap = average_precision_score(all_label.flatten() > 0, all_output.flatten(), average=None)
    #ap.append(overallap)
    auc = roc_auc_score(all_label > 0, all_output, average=None)
    overallauc = roc_auc_score(all_label.flatten() > 0, all_output.flatten(), average=None)
    interAP = []
    overall_IAPlist = []
    for theta in np.arange(0, 1, 0.1):
        inter_AP, overall_IAP = interval_AP(output_list, label_list, theta)
        interAP.append(inter_AP)
        overall_IAPlist.append(overall_IAP)
    print(np.vstack(interAP).shape)
    np.savetxt(output_dir + '/ap.csv', ap, delimiter=',', fmt='%0.5f')
    np.savetxt(output_dir + '/auc.csv', auc, delimiter=',', fmt='%0.5f')
    np.savetxt(output_dir + '/interAP.csv', np.vstack(interAP), delimiter=',', fmt='%0.5f')
    overallap.tofile(output_dir+'/overallapp.csv', sep=' ', format='%0.5f')
    overallauc.tofile(output_dir + '/overallauc.csv', sep=' ', format='%0.5f')
    np.savetxt(output_dir + '/overallinterAP.csv', np.asarray(overall_IAPlist), fmt='%0.5f')
    #overall_IAPlist.tofile(output_dir + '/overallinterAP.csv', sep=' ', format='%0.5f')
   # np.savetxt(output_dir + '/overallap.csv', overallap, delimiter=',', fmt='%0.5f')
