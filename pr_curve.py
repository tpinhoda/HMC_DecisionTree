#import utils
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix
#from DatasetLoader import DatasetLoader
#from timeit import default_timer as DT

dataset_loader = None

def apply_thresholds(Y):
    ys_bin = []
    thresholds = np.arange(0.00, 1.01, 0.02)

    Y_original = np.asarray(Y).copy()
    # print 'target Y'
    # print Y
    # print thresholds
    for t in thresholds:
        # print 'Threshold', t
        Y = apply_threshold(Y_original.copy(), threshold=t)
        if dataset_loader is not None:
            Y = dataset_loader.correct_inconsistencies(Y)
        # Y.shape = shp
        ys_bin.append(Y.copy())
        # print Y_original.reshape(shp)
        # print Y
    return ys_bin

# def apply_threshold(y, threshold):
#     if threshold == 0:
#         idx = y <= threshold
#     else:
#         idx = y < threshold

#     y[idx] = 0.
#     idx = y >= threshold
#     y[idx] = 1.
#     return y

def apply_threshold(y, threshold):
    msk = (y < threshold)
    ny = np.ones(y.shape).astype(np.int)
    ny[msk] = 0
    return ny


def plot_curve(prec, rec, legend=None):

    from matplotlib import pyplot as plt
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    if legend is not None:
        plt.title('Average AUC: %5.4f' % legend)

    plt.legend(loc="lower right")
    plt.plot(rec, prec)

    plt.show()


def process_tp_fp(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).copy()
    y_pred = np.asarray(y_pred).copy()

    n_instances = y_true.shape[0]
    n_classes   = y_true.shape[1]

    TPs, FPs, FNs = [], [], []

    diff_matrix = (y_true * 10 + y_pred)
    # print diff_matrix.shape
    # print diff_matrix
    for i in range(n_classes):
        a = diff_matrix[:, i].copy()

        b = a[a == 11].copy().shape[0]
        c = a[a == 1].copy().shape[0]
        d = a[a == 10].copy().shape[0]

        TPs.append(b)
        FPs.append(c)
        FNs.append(d)

    # TPs, FPs, FNs = np.asarray(TPs), np.asarray(FPs), np.asarray(FNs)
    return TPs, FPs, FNs


def process_tp_fp_(y_true, y_pred):

    y_true = np.asarray(y_true).copy()
    y_pred = np.asarray(y_pred).copy()

    n_instances = y_true.shape[0]
    n_classes   = y_true.shape[1]

    TPs, FPs, FNs = [], [], []
    for i in range(n_classes):

        real = y_true[:, i]
        pred = y_pred[:, i]

        # print real
        # print pred

        tp, fp, fn = 0, 0, 0

        for r, p in zip(real, pred):
            if r == 1 and p == 1:
                tp += 1
            if r == 1 and p == 0:
                fn += 1
            if r == 0 and p == 1:
                fp += 1

        # print cm
        # tp = cm[0, 0]
        # fp = cm[1, 0]
        # fn = cm[0, 1]
        # tn = cm[1, 1]

        TPs.append(tp)
        FPs.append(fp)
        FNs.append(fn)

    # TPs, FPs, FNs = np.asarray(TPs), np.asarray(FPs), np.asarray(FNs)
    return TPs, FPs, FNs


def calc_avg_precision_recall(TP, FP, FN):

    # print 'True Positives  sum:', np.sum(TP)
    # print 'False Positives sum:', np.sum(FP)
    # print 'False Negatives sum:', np.sum(FN)

    avg_precision = np.sum(TP) / float(np.sum(TP) + np.sum(FP))
    avg_recall    = np.sum(TP) / float(np.sum(TP) + np.sum(FN))

    if np.isnan(avg_precision):
        avg_precision = 0

    if np.isnan(avg_recall):
        avg_recall = 0

    # if np.sum(TP) == 0 and np.sum(FP) == 0:
    #     avg_precision = 1
    #     avg_recall = 0

    # if np.sum(FP) == 0 and np.sum(FN) == 0:
    #     avg_precision = 0
    #     avg_recall = 1

    return avg_precision, avg_recall


def calculate_AUPRC(y_true, y_pred, plot=False):
    from sklearn.metrics import auc

    # print np.asarray(y_true).shape
    b = DT()
    # print y_true_bin.shape

    y_pred_bin = np.asarray(apply_thresholds(y_pred))
    y_true_bin = [y_true for a in range(len(y_pred_bin))]
    e = DT()
    threshold_time = e - b
    # print y_pred_bin.shape

    # print 'thres number: ', len(y_true_bin)
    # print 'thres number: ', len(y_pred_bin)
    # print 'shape       : ', y_pred_bin[0].shape
    # print 'first ypred :  ', y_pred_bin[0]

    b = DT()
    precisions = []
    recalls = []

    # Iterates points according to the thresholds
    # Calculate the average prec/revoc per class

    t = 52
    i = -1
    TPs, FPs, FNs = [], [], []
    precisions_ = [[],[]]
    recalls_ = [[],[]]

    # print 'Values precision recall size: ', len(y_true_bin)
    from tqdm import tqdm
    for ytb, ypb in zip(y_true_bin[::-1], y_pred_bin[::-1]):
        i += 1
        t -= 1
        # print ytb
        # print ypb
        TP, FP, FN = process_tp_fp(ytb, ypb)
        TPs.append(np.sum(TP))
        FPs.append(np.sum(FP))
        FNs.append(np.sum(FN))

        precision, recall = calc_avg_precision_recall(TP, FP, FN)

        if t == 0:
            recall = 1.
            precision = 0.

        precisions += [precision]
        recalls += [recall]


        # print '\nThreshold: %2i' % t
        # print 'precision: %10.9f' % precision
        # print 'recall   : %10.9f' % recall
        # # print 'TP/FP/FN : %5i, %5i, %5i' % (np.sum(TP), np.sum(FP), np.sum(FN))
        # print 'TP       : %5i' % np.sum(TP)
        # print 'FP       : %5i' % np.sum(FP)
        # print 'TP+FP    : %5i' % (np.sum(FP) + np.sum(TP))
        # print 'TP+FN    : %5i' % (np.sum(FN) + np.sum(TP))

        if i > 0:
            prec_a = np.asarray(precisions[i-1]).astype(np.float64)
            rec_a = np.asarray(recalls[i-1]).astype(np.float64)
            tpa = np.asarray(TPs[i-1]).astype(np.float64)
            fpa = np.asarray(FPs[i-1]).astype(np.float64)
            fna = np.asarray(FNs[i-1]).astype(np.float64)

            prec_b = np.asarray(precisions[i]).astype(np.float64)
            rec_b = np.asarray(recalls[i]).astype(np.float64)
            tpb = np.asarray(TPs[i]).astype(np.float64)
            fpb = np.array(FPs[i]).astype(np.float64)
            fnb = np.array(FNs[i]).astype(np.float64)

            points = get_points(prec_a, prec_b, rec_a, rec_b, tpa, tpb, fpa, fpb, fna, fnb, count=i)

            if t < (len(y_true_bin) - 2):
                del precisions_[-1][-1]
                del recalls_[-1][-1]


            precisions_.append(points[0])
            recalls_.append(points[1])

    e = DT()
    processing_tp_fp_PR_time = e - b

    b = DT()

    del precisions_[0]
    del recalls_[0]
    del precisions_[0]
    del recalls_[0]


    x, y, AUPRC = calculateAreaUnderCurve(recalls_, precisions_)
    e = DT()
    auc_time = e - b
    # print 'AUPRC: ', AUPRC

    if plot:
        plot_curve(x, y, legend = AUPRC)

    # print 'threshold_time\t', threshold_time
    # print 'processing_tp_\t', processing_tp_fp_PR_time
    # print 'auc_time      \t', auc_time

    return AUPRC, x, y


def calculateAreaUnderCurve(recall, precision):
    AUPRC = 0
    x = []
    y = []

    for i in range(len(recall)):
        for j in range(len(recall[i])):
            x.append(recall[i][j])
            y.append(precision[i][j])

        # AUPRC += (x.get(i + 1) - x.get(i)) * y.get(i + 1)
        #        + (x.get(i + 1) - x.get(i)) * (y.get(i) - y.get(i + 1)) / 2;

    for i in range(len(x)-1):
        AUPRC += (x[i + 1] - x[i]) * y[i + 1] + (x[i + 1] - x[i]) * (y[i] - y[i + 1]) / 2.

    return x, y, AUPRC


def get_points(prec_a, prec_b, rec_a, rec_b, tpa, tpb, fpa, fpb, fna, fnb, count):

    if (tpb - tpa) == 0:
        localSkew = 0.
    else:
        localSkew = (fpb - fpa) / (tpb - tpa)

    total = tpa + fna

    param = tpb - tpa

    prec = []
    reca = []

    points = []

    if count == 0 and rec_a > 0:
        prec.append(prec_a)
        reca.append(0.)

    if count == 0:
        prec.append(prec_a)
        reca.append(rec_a)

    if count > 0 and (prec_a != 0 or rec_a != 0 or prec_b != 0 or rec_b != 0):
        prec.append(prec_a)
        reca.append(rec_a)

    for tp in range(int(tpa) + 1, tpb):
        fp = fpa + localSkew * (tp - tpa)
        newPrec = tp / (tp + fp)
        newReca = tp / total
        prec.append(newPrec)
        reca.append(newReca)

    prec += [prec_b]
    reca += [rec_b]

    # print 'prec size', len(prec)
    # print 'rec size ', len(reca)

    points.append(prec)
    points.append(reca)

    return points


def pr_auc_sklearn(y_true, y_pred, plot=False):

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score


    y_score = np.asarray(y_pred)
    y_test = np.asarray(y_true)

    n_classes = y_score.shape[1]

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")
    # Plot Precision-Recall curve for each class
    if plot:
        plt.clf()
        plt.plot(recall["micro"], precision["micro"],
                 label='micro-average Precision-recall curve (area = {0:0.2f})'
                       ''.format(average_precision["micro"]))
        # for i in range(n_classes):
        #     plt.plot(recall[i], precision[i],
        #              label='Precision-recall curve of class {0} (area = {1:0.2f})'
        #                    ''.format(i, average_precision[i]))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(loc="lower right")
        plt.show()

    #print average_precision['micro']
    return average_precision['micro']

def get_markers(x, y, perc):
    n = len(x)
    nb = int(float(n) * perc)

    print n, nb

    x_, y_ = [], []
    for i in np.arange(0, n, nb):
        x_.append(x[i])
        y_.append(y[i])

    return x_, y_



def plot_all_curves(dataset):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    from utils import load_pickle, save_pickle
    from JHMC import JHMC
    import config
    name = config.datasets_names[dataset]

    plt.clf()
    mpl.rcParams['axes.color_cycle'] = ['green', 'red', 'black']
    hmc = JHMC(dataset.lower())

    dataset_loader = hmc.dl
    y_test = hmc.dl.y_test

    hmcens_path = 'AUPRC/reclusensembleresults/'
    wbc_path    = 'Predictions/WBC-Net/'
    mnn_path    = 'Predictions/MNN/'

    hmc_ens = np.genfromtxt(hmcens_path + '%s-overall.dat' % name.lower(), delimiter=',')
    wbc_net = load_pickle(wbc_path + '/%s_test.pred' % dataset.lower())
    mnn     = load_pickle(mnn_path + '/%s_test' % dataset.lower())

    a, wbc_x, wbc_y = calculate_AUPRC(y_test, wbc_net)
    a, mnn_x, mnn_y = calculate_AUPRC(y_test, mnn)

    save_pickle('AUPRC/wbc/points.pickle', (wbc_x, wbc_y))
    save_pickle('AUPRC/mnn/points.pickle', (mnn_x, mnn_y))

    # wbc_x, wbc_y = load_pickle('AUPRC/wbc/points.pickle')
    # mnn_x, mnn_y = load_pickle('AUPRC/mnn/points.pickle')

    plt.plot(hmc_ens[:, 0], hmc_ens[:, 1], 'green')
    x__, y__ = get_markers(hmc_ens[:, 0], hmc_ens[:, 1], perc=0.1)
    plt.plot(x__, y__, 'x')
    plt.plot([], [], 'green', marker='x', label='Clus-HMC-Ens')

    plt.plot(wbc_x, wbc_y, 'red')
    x__, y__ = get_markers(wbc_x, wbc_y, perc=0.10)
    plt.plot(x__, y__, '.', marker='+')
    plt.plot([], [], 'red', marker='+', label='WBC-Net')

    plt.plot(mnn_x, mnn_y, 'black')
    x__, y__ = get_markers(mnn_x, mnn_y, perc=0.13)
    plt.plot(x__, y__, '.', marker='^')
    plt.plot([], [], 'black', marker='^', label='MNN')

    plt.legend(fontsize=16)
    plt.grid()
    plt.title('%s' % name, fontsize=20)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    # plt.show()
    plt.savefig('figures/curves/%s.pdf' % name, bbox_inches='tight')


if __name__ == '__main__':
    #from utils import load_npz
    #import config

    #for dataset in config.datasets:
    # for dataset in ['cellcycle']:
    #    plot_all_curves(dataset)

    #exit()
    # y_true = load_npz('values_tr.npz')
    # y_pred = load_npz('values_pr.npz')
    # print y_true.shape

    #np.random.seed = 12345
    y_true = [  [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [0, 1, 0]]

    # y_pred = [  [0.1, 1, 0.1],
    #             [0.1, 0.3, 0.5],
    #             [0.5, 0.4, 0.4],
    #             [0.1, 0.99, 0.5],
    #             [0.3, 0.6, 0.9],
    #             [0.1, 0.2, 0.4]]


    y_pred = [  [0.2, 0.7, 0.8],
                [0.5, 0.6, 0.4],
                [0.9, 0.3, 0.7],
                [0.7, 0.4, 0.3],
                [0.6, 0.7, 0.9],
                [0.8, 0.1, 0.2],
                [0.5, 0.6, 0.5]]

    # y_true = np.asarray([[np.random.randint(0, 2), np.random.randint(0, 2), np.random.randint(0, 2)] for i in range(1000)])
    # # y_true = np.asarray([[np.random.randint(0, 2)] for i in range(1000)])
    # # y_true = np.asarray(y_true)/100.

    # y_pred = np.random.random((1000, 3))
    # # y_pred = np.random.random((1000, 1))


    # # # from utils import load_npz
    # y_pred = load_npz('valid_pred.npz')
    #y_true = np.genfromtxt('results/y_valid.csv')
    #y_pred = np.genfromtxt('/home/gpin/jonatas/proteins/results/results_cerri.csv', delimiter=',')
    #y_pred = y_pred[:, :499]
    # y_pred = np.asarray(y_pred)
    #y_true = np.asarray(y_true)

    # print y_true
    # print y_pred
    #print 'y true shape', y_true.shape
    #print 'y pred shape', y_pred.shape

    print pr_auc_sklearn(y_true, y_pred, plot=False)
    # print pr_auc_sklearn(y_true, y_pred, plot=False)

    #from timeit import default_timer as DT

    #time_begin = DT()
    # print y_pred
    print calculate_AUPRC(y_true, y_pred, plot=False)
    # plot_curve(precisions, recalls)

    #print 'Total time: ', DT() - time_begin
