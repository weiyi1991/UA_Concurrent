import numpy as np
#from apmeter import APMeter
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

def get_intervals(seq, thres=0.0001, mean=False):
    '''
    Get the positive interval of a prediction score sequence with the given threshold
    Input:
        seq - Input sequence with prediction score, 1*N
        thres - threshold value to determine the positive interval, default=0.0001
        mean - get the mean value of each interval and return as the interval score, default=False
    Output:
        intervals - a list containing (start, end) frame of the positive intervals
    '''
    intervals = []
    score = []
    flag = True # flag to note if not in a positive interval
    for i, pos in enumerate(seq >= thres):
        if pos and flag:
            start = i   # start frame of positive interval
            flag = False
        if (not pos) and (not flag):
            end = i-1   # end frame of positive interval
            flag = True
            intervals.append((start, end))
            if mean:
                score.append(np.mean(seq[start:end+1]))
    # if in a positive interval at the end of the sequence
    if not flag:
        end = i
        intervals.append((start, end))
        if mean:
            score.append(np.mean(seq[start:end + 1]))
    if mean:
        return intervals, np.array(score)
    else:
        return intervals

def pred_label(pred_interval, gt_interval):
    '''
    get the true or false label for predicted interval
    :param pred_interval: list contain predicted interval (start, end) pairs
    :param gt_interval: list contain ground truth intervals
    :return:
       pred_label: list with the same number of elements of pred_interval, denote the prediction is true or false
    '''
    pred_label = np.zeros(len(pred_interval))
    for i, (s_i, e_i) in enumerate(pred_interval):
        for j, (s_j, e_j) in enumerate(gt_interval):
            # 3 cases when predicted interval is regarded as TP
            # predicted interval within ground truth interval
            if (s_i >= s_j) and (e_i <= e_j):
                pred_label[i] = 1
                break
            # predicted interval is behind of ground truth interval, the IoU > 60%
            if (s_i >= s_j) and (e_i > e_j):
                if (e_j-s_i)/(e_i-s_j) > 0.6:
                    pred_label[i] = 1
                    break
            # predicted interval is ahead of ground truth interval, the IoU > 60%
            if (s_i < s_j) and (e_i <= e_j):
                if (e_i - s_j) / (e_j - s_i) > 0.6:
                    pred_label[i] = 1
                    break
            if e_i < s_j:
                break
    return pred_label


def test():
    gt = np.loadtxt('output/gt/sequence_001.txt')
    pred = np.loadtxt('output/baseline/sequence_001.txt')
    a = np.array([0.3, 0.4, 0.5, 0.6, 0.1, 0.2, 0.5, 0.5, 0.6, 0.1, 0.1, 0.1, 0.2, 0.9])
    g = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    ai, s = get_intervals(a, 0.3, True)
    print(ai)
    print(s)
    gi = get_intervals(g)
    print(gi)
    # get predicted interval label
    l = pred_label(ai, gi)
    print(l)
    print(average_precision_score(g, a))
    print(average_precision_score(l, s))
    #apmeter = APMeter()
    #apmeter.add(pred, gt)
    #print(apmeter.value())
    print('AP and AUC')


def test_seq(pred, gt, theta):
    '''
    pred: [sample_num * class_num] numpy array
    gt: [sample_num * class_num] numpy array
    theta: threshold to determine the positive intervals for pred
    '''
    pfrm, pclass = pred.shape
    gfrm, gclass = gt.shape
    if pfrm==gfrm and pclass==gclass:
        pass
    else:
        print('dimension not match')
    ap = np.zeros(pclass)
    for i in range(pclass):
        predi, si = get_intervals(pred[:,i], theta, True)
        # if get no positive prediction
        if not predi:
            ap[i] = 0
            continue
        gti = get_intervals(gt[:,i])
        li = pred_label(predi, gti)
        print('label and score at {}:\n'.format(i), li, '\n', si)
        ap[i] = average_precision_score(li, si, average=None)
    return ap

def interval_AP(pred_list, gt_list, theta):
    '''
    pred: [sample_num * class_num] tensor list, each tensor is an array of one sequence
    gt: [sample_num * class_num] tensor list, each tensor is an array of one sequence
    theta: threshold to determine the positive intervals for pred
    '''
    class_num = 12
    ap = np.zeros(class_num)
    overall_score = []
    overall_label = []
    for i in range(class_num):
        labeli_list = [] # list to store all detected interval labels for class i
        scorei_list = [] # list to store all detected interval scores for class i
        for pred, gt in zip(pred_list, gt_list):
            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()
            pfrm, pclass = pred.shape
            gfrm, gclass = gt.shape
            if pfrm==gfrm and pclass==gclass:
                pass
            else:
                print('dimension not match')
            # for each class, generate interval label and score for each sequence and stack into one array
            predi, si = get_intervals(pred[:,i].T, theta, True)
            # if get no positive prediction
            if not predi:
                continue
            gti = get_intervals(gt[:,i].T)
            li = pred_label(predi, gti) # label for detected intervals at current sequence and class i
            #print('label and score at {}-{}:\n'.format(len(labeli_list), i), li, '\n', si)
            labeli_list.append(li)
            scorei_list.append(si)
        if len(labeli_list) == 0:
            continue
        ap[i] = average_precision_score(np.concatenate(labeli_list, axis=0), np.concatenate(scorei_list, axis=0), average=None)
        overall_label.append(np.concatenate(labeli_list, axis=0))
        overall_score.append(np.concatenate(scorei_list, axis=0))
    if len(overall_score) != 0:
        overall_ap = average_precision_score(np.concatenate(overall_label, axis=0), np.concatenate(overall_score, axis=0), average=None)
    else:
        overall_ap = 0
    return ap, overall_ap

if __name__ == '__main__':
    #print(test_seq(pred, gt, 0.5))
    print(test())
