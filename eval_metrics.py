import numpy as np
from scipy import interpolate
from sklearn.model_selection import KFold


def evaluate(distances, labels, nrof_folds=5):
    '''
        This function does the calculation of usefule metrics which we need to find out tpr,fpr , accuracy and other metrics with different thresholds applied
    '''

    thresholds = np.arange(0.3, 5, 0.1)
    ## calculation of tpr , fpr and accuracy from the roc curve
    pre , rec , f1_s , accuracy = calculate_roc(thresholds, distances,
                                       labels, nrof_folds=nrof_folds)

    thresholds = np.arange(0, 30, 0.001)
    ## calculation of variance , standard deviation and false acceptance rate 
    val, val_std, far = calculate_val(thresholds, distances,
                                      labels, 1e-3, nrof_folds=nrof_folds)
    return pre , rec , f1_s , accuracy, val, val_std, far


def calculate_roc(thresholds, distances, labels, nrof_folds=10):
    """
       We use the n_folds split to calcuate roc since doing the calculation all at once wont be feasible if we have large number of triplets
    """
    nrof_pairs = min(len(labels), len(distances)) ## no. of pairs to be used for comparison
    nrof_thresholds = len(thresholds)             ## the no. of thresholds being used
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)  ## initializing the KFold split 

    ## initializing numpy arrays for calculation of tpr,fpr and accuracy
    precision = np.zeros((nrof_folds, nrof_thresholds))  
    recall = np.zeros((nrof_folds, nrof_thresholds))
    f1_score = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    ## creating and indices array for the list of pairs
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        
        ## finding the best threshold index
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        
        best_threshold_index = np.argmax(acc_train)
        
        ## calcuating the tpr,fpr for the given fold
        for threshold_idx, threshold in enumerate(thresholds):
            precision[fold_idx, threshold_idx], recall[fold_idx, threshold_idx],f1_score[fold_idx,threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 distances[test_set],
                                                                                                 labels[test_set])
        ## calcuating the accuracy for the given fold
        _, _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set],
                                                      labels[test_set])

    pre = np.mean(precision, 0)
    rec = np.mean(recall, 0)
    f1_s = np.mean(f1_score, 0)

    return pre, rec, f1_s, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    """
        This function does the accuracy calculation by taking the distances , labels and the threshold as its arguments
    """
    predict_issame = np.less(dist, threshold)  ## compares two inputs elementwise and return their results in boolean form
    tp = np.sum(np.logical_and(predict_issame, actual_issame)) ## calculation of the true positives 
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame))) ## calcualtion of the false positives
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))) ## calculation of the true negatives
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame)) ## calculation of the false negatives

    precision = 0 if (tp + fp == 0) else float(tp) / float(tp + fp) ## finding out the precision
    recall = 0 if (tp + fn == 0) else float(tp) / float(tp + fn) ## finding out the recall
    f1_score = 2*(precision*recall)/ (precision + recall)

    accuracy = float(tp + tn) / dist.size
    return precision, recall,f1_score, accuracy


def calculate_val(thresholds, distances, labels, far_target=1e-3, nrof_folds=10):
    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, distances[train_set], labels[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, distances[test_set], labels[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    '''
    This function calculates and returns the true positve rate and false positive rate with the given threshold applied
    '''
    predict_issame = np.less(dist, threshold) ## compares two inputs elementwise and return their results in boolean form
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame)) ## computes logical and of predicted and actual classes (True Accept Rate)
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame))) ## computes logical and of predicted and negation of the actual classes (False Acceptance Rate)
    n_same = np.sum(actual_issame)  ## computes actual number of true classes
    n_diff = np.sum(np.logical_not(actual_issame)) ## computes actual number of not true classes
    if n_diff == 0:
        n_diff = 1
    if n_same == 0:
        return 0, 0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def plot_roc(fpr, tpr, figure_name="roc.png"):
    """
        This function plots the roc curve for the given values of tpr and fpr
    """
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='#16a085',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='#2c3e50', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right", frameon=False)
    fig.savefig(figure_name, dpi=fig.dpi)
