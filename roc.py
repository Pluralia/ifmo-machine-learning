import numpy as np
import matplotlib.pyplot as plt


def roc(data, labels):
    sorted_index = np.argsort(data)

    labels1 = np.sum(labels)
    labels0 = labels.shape[0] - labels1

    true_positive = labels1
    false_positive = labels0
    tp_list = []
    fp_list = []
    for i, (x, curr_label) in enumerate(zip(data[sorted_index], labels[sorted_index])):
        if i == 0:
            prev = x
        if curr_label == 1:
            true_positive -= 1
        if curr_label == 0:
            false_positive -= 1
        if prev == x:
            continue
        prev = x
        fp_list = [false_positive] + fp_list
        tp_list = [true_positive] + tp_list
    return np.array(tp_list) / labels1, np.array(fp_list) / labels0


def roc_auc(data, labels):
    tp_list, fp_list = roc(data, labels)
    t_sum = (tp_list[1:] + tp_list[:-1]) / 2
    f_sub = fp_list[1:] - fp_list[:-1]
    return np.sum(t_sum * f_sub)


def evaluate_best_features(data, labels):
    features_auc = [(roc_auc(data[:, i], labels), i) for i in range(data.shape[1])]
    best_features = sorted(features_auc, reverse=True)[:3]
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    for i, (auc, feature_idx) in enumerate(best_features):
        tp_list, fp_list = roc(data[:, feature_idx], labels)
        axes[i].plot(fp_list, tp_list)
        axes[i].set_title("feature: %i, roc_auc: %.3f" % (feature_idx, auc))
    plt.show()
