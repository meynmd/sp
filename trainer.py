import random
from copy import deepcopy
from collections import defaultdict
import matplotlib.pyplot as plt
from tagger import *

num_epochs = 15

def update(model, x, z, y):
    for i, word in enumerate(x):
        if z[i] != y[i]:
            if i > 0:
                model[z[i - 1], z[i]] -= 1.
                model[y[i - 1], y[i]] += 1.
            model[z[i], word] -= 1.
            model[y[i], word] += 1.


def update_avgd(model, model_avg, x, z, y, c):
    for i, word in enumerate(x) :
        if z[i] != y[i]:
            if i > 0:
                # update w_0 transition
                model[z[i - 1], z[i]] -= 1.
                model[y[i - 1], y[i]] += 1.
                # update w_a transition
                model_avg[z[i - 1], z[i]] -= c
                model_avg[y[i - 1], y[i]] += c
            # update w_0 emission
            model[z[i], word] -= 1.
            model[y[i], word] += 1.
            # update w_a emission
            model_avg[z[i], word] -= c
            model_avg[y[i], word] += c


def train_epoch(filename, dictionary, model=None, model_avg=None, c=None):
    if model is None:
        model, model_avg = defaultdict(float), defaultdict(float)
    else:
        model = deepcopy(model)

    xys = [xy for xy in readfile(filename)]
    random.shuffle(xys)
    updates = 0
    for x, y in xys:
        if c is not None:
            c += 1
        z = decode(x, dictionary, model)
        if z != y:
            updates += 1
            if c is None:
                update(model, x, z, y)
            else:
                update_avgd(model, model_avg, x, z, y, c)

    if c is None:
        return model, updates
    else:
        model_final = deepcopy(model)
        for k in model_final:
            model_final[k] -= model_avg[k] / float(c)
        return model, updates, model_avg, model_final, c


def run_training(trainfile, devfile, dictionary, averaged=False, plot=True):
    model, model_avg, model_final, best_model = None, None, None, None
    c, best_epoch = 0, 0
    tr_err, dev_err, models = [], [], []
    least_err = float('inf')
    for i in range(num_epochs):
        if averaged:
            model, updates, model_avg, model_final, c = train_epoch(
                trainfile, dictionary, model, model_avg, c)
        else:
            model_final, updates = train_epoch(trainfile, dictionary, model)

        # if best_model is None:
        #     best_model = model_final
        #     best_epoch = i + 1

        models.append(model_final)
        tr_err.append(test(trainfile, dictionary, model_final))
        dev_err.append(test(devfile, dictionary, model_final))
        print "epoch {0:2}\tupdates: {1:3}\t".format(i + 1, updates),
        print "train err: {0:.2%}\tdev err: {1:.2%}".format(tr_err[-1], dev_err[-1])

        # if dev_err < least_err:
        #     best_model = model_final
        #     least_err = dev_err
        #     best_epoch = i + 1

    # return least_err, best_model, best_epoch
    return tr_err, dev_err, models

def train_and_report(trainfile, devfile, dictionary, averaged=False):
    print "-"*80
    if averaged:
        print "Averaged",
    print "Perceptron\nTraining..."

    # d_err, best_model, best_epoch = run_training(trainfile, devfile, dictionary, averaged)
    tr_err, dev_err, models = run_training(trainfile, devfile, dictionary, averaged)
    best_epoch = min([i for i in range(len(dev_err))], key=lambda x : dev_err[x])
    best_model = models[best_epoch]
    final_err = test(devfile, dictionary, best_model)

    print "\nbest epoch: {}".format(best_epoch + 1)
    print "final dev err {0:.2%}".format(final_err)
    print "-"*80 + "\n"

    return tr_err, dev_err


if __name__ == "__main__":
    average = False
    if len( sys.argv ) > 3:
        if sys.argv[3] == '-a':
            average = True

    trainfile, devfile = sys.argv[1:3]
    dictionary, blah = mle(trainfile)

    tr_err, dev_err = train_and_report(trainfile, devfile, dictionary, False)
    avg_tr_err, avg_dev_err = train_and_report(trainfile, devfile, dictionary, True)

    epochs = len(tr_err)
    plt.plot( range( epochs ), tr_err )
    plt.plot( range( epochs ), dev_err )
    plt.plot( range( epochs ), avg_tr_err )
    plt.plot( range( epochs ), avg_dev_err )
    plt.legend( ('Perc. Training', 'Perc. Dev', 'Avg. Training', 'Avg. Dev'), numpoints = 1)
    plt.show()
