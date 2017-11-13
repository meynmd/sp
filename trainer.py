import random
from copy import deepcopy
import numpy as np
from collections import defaultdict
from tagger import *

def train_perceptron(filename, dictionary, model=None):
    if model is None:
        model = defaultdict(float)
    else:
        model = deepcopy(model)

    xys = [xy for xy in readfile(filename)]
    random.shuffle(xys)
    for x, y in xys:
        z = decode(x, dictionary, model)
        if z != y:
            for i, w in enumerate(x):
                if z[i] != y[i]:
                    if i > 0:
                        model[z[i - 1], z[i]] -= 1.
                        model[y[i - 1], y[i]] += 1.
                    model[z[i], w] -= 1.
                    model[y[i], w] += 1.
    return model


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


def train_avg_perceptron(filename, dictionary, model=None, model_avg=None, c=0):
    if model is None:
        model, model_avg = defaultdict(float), defaultdict(float)
    else:
        model = deepcopy(model)

    xys = [xy for xy in readfile(filename)]
    random.shuffle(xys)
    for x, y in xys:
        z = decode(x, dictionary, model)
        if z != y:
            # for i, w in enumerate(x):
            #     if z[i] != y[i]:
            #         if i > 0:
            #             # update w_0
            #             model[z[i - 1], z[i]] -= 1.
            #             model[y[i - 1], y[i]] += 1.
            #             # update w_a
            #             model_avg[z[i - 1], z[i]] -= c
            #             model_avg[y[i - 1], y[i]] += c
            #         # update w_0
            #         model[z[i], w] -= 1.
            #         model[y[i], w] += 1.
            #         # update w_a
            #         model_avg[z[i], w] -= c
            #         model_avg[y[i], w] += c
            update_avgd(model, model_avg, x, z, y, c)
        c += 1

    model_final = deepcopy(model)
    for k in model_final:
        model_final[k] -= model_avg[k] / float(c)
    return model, model_avg, model_final, c


if __name__ == "__main__":
    average = False
    if len( sys.argv ) > 3:
        if sys.argv[3] == '-a':
            average = True

    trainfile, devfile = sys.argv[1:3]
    dictionary, blah = mle(trainfile)

    model, model_avg, model_final, best_model = None, None, None, None
    c, best_epoch = 0, 0
    least_err = float('inf')
    for i in range(5):
        if average:
            model, model_avg, model_final, c = train_avg_perceptron(
                trainfile, dictionary, model, model_avg, c)
        else:
            model_final = train_perceptron(trainfile, dictionary, model)
        if best_model is None:
            best_model = model_final
            best_epoch = i + 1
        tr_err = test(trainfile, dictionary, model_final)
        dev_err = test(devfile, dictionary, model_final)
        print "epoch {0:2}\t".format(i + 1),
        print "train_err {0:.2%}\tdev_err {1:.2%}".format(tr_err, dev_err)
        if dev_err < least_err:
            best_model = model_final
            least_err = dev_err
            best_epoch = i + 1

    d_err = test(devfile, dictionary, best_model)
    print "\nbest epoch: {}".format(best_epoch)
    print "dev_err {0:.2%}".format(d_err)