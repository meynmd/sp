import random
from copy import deepcopy
from collections import defaultdict
from tagger import *

num_epochs = 50

def update(model, model_avg, x, z, y, c, gram=2, w_range=(0,1)):
    active_features = []
    z, y = [startsym] + z + [stopsym], [startsym] + y + [stopsym]
    x = [startsym] + x + [stopsym]
    for i, word in enumerate(x):
        if z[i] != y[i]:
            # adjust emission weight
            for start in range(w_range[0], 1):
                for end in range(w_range[1], 0, -1):
                    modelkey = tuple( [z[i]] + x[i+start: i+end] )
                    model[tuple([z[i]] + x[i + start : i + end])] -= 1.
                    model[tuple([y[i]] + x[i + start : i + end])] += 1.


            model[z[i], word] -= 1.
            model[y[i], word] += 1.
            model_avg[z[i], word] -= c
            model_avg[y[i], word] += c
            active_features += [(z[i], word), (y[i], word)]

        # adjust transition weight for n-gram and smaller
        for n in range(1, gram):
            if i - n > 0:
                zgram = tuple([z[i - j] for j in range(n, -1, -1)])
                ygram = tuple([y[i - j] for j in range(n, -1, -1)])
                if zgram != ygram:
                    model[zgram] -= 1.
                    model[ygram] += 1.
                    model_avg[zgram] -= c
                    model_avg[ygram] += c
                    active_features += [ygram]

    return active_features


def train(trainfile, devfile, dictionary, tgram=2, wrange=(0, 1)):
    model, model_avg = defaultdict(float), defaultdict(float)
    xys = [xy for xy in readfile(trainfile)]
    tr_err, tr_avg_err, dev_err, avg_err, models, avg_models = [], [], [], [], [], []
    c = 0
    active_features = []
    for i in range(num_epochs):
        updates = 0
        random.shuffle(xys)
        for x, y in xys:
            choices = [len(dictionary[w]) for w in x]
            c += 1
            z = my_decode_var_gram( x, dictionary, model, tgram, (-1, 1) )
            if z != y:
                updates += 1
                active_features += update(model, model_avg, x, z, y, c, tgram, wrange)

        model_avg_complete = deepcopy(model)
        for k in model_avg_complete:
            model_avg_complete[k] -= model_avg[k] / float(c)

        completed, completed_avg = deepcopy(model), deepcopy(model_avg_complete)
        models.append(completed)
        avg_models.append(completed_avg)
        tr_err.append(test(trainfile, dictionary, completed, tgram, wrange))
        tr_avg_err.append(test(trainfile, dictionary, completed_avg, tgram, wrange))
        dev_err.append(test(devfile, dictionary, completed, tgram, wrange))
        avg_err.append(test(devfile, dictionary, completed_avg, tgram, wrange))
        w_length = len(set(active_features))

        print "epoch {0:2}\tupdates: {1:3}\t|w| = {2}\t".format(i + 1, updates, w_length),
        print "train err: {0:.2%}\ttrain avg err: {1:.2%}\tdev err: {2:.2%}\tavg err: {3:.2%}".format(
            tr_err[-1], tr_avg_err[-1], dev_err[-1], avg_err[-1]
        )

    return tr_err, tr_avg_err, dev_err, avg_err, models, avg_models


def train_and_report(trainfile, devfile, dictionary, gram=2, w_range=(0,1)):
    print "-"*80, "\nTraining..."

    tr_err, tr_avg_err, dev_err, avg_err, models, avg_models = train(trainfile, devfile, dictionary, gram, w_range)
    best_epoch = min([i for i in range(len(dev_err))], key=lambda x : dev_err[x])
    best_model = models[best_epoch]
    best_epoch_avg = min( [i for i in range( len( avg_err ) )], key=lambda x: avg_err[x] )
    best_avg_model = avg_models[best_epoch_avg]
    final_err = test(devfile, dictionary, best_model, gram, w_range)
    final_avg_err = test(devfile, dictionary, best_avg_model, gram, w_range)

    print "\nbest epoch (perceptron): {}\t(avg. perc.): {}".format(best_epoch + 1, best_epoch_avg + 1)
    print "best dev err (perc.): {0:.2%}\t(avg): {1:.2%}".format(final_err, final_avg_err)
    print "-"*80 + "\n"

    return tr_err, tr_avg_err, dev_err, avg_err


if __name__ == "__main__":
    random.seed(1)
    plot = False
    gram = 2
    if len( sys.argv ) > 3:
        if sys.argv[3] == '-p':
            plot = True
        elif sys.argv[3] == '-g':
            gram = int(sys.argv[4])

    word_range = (0, 1)
    if len(sys.argv) > 6:
        if sys.argv[4] == '-r':
            word_range = (int(sys.argv[5]), int(sys.argv[6]))

    trainfile, devfile = sys.argv[1:3]
    dictionary, _ = mle(trainfile)
    dictionary[startsym].add(startsym)
    dictionary[stopsym].add(stopsym)

    tr_err, avg_tr_err, dev_err, avg_err = train_and_report(trainfile, devfile, dictionary, gram, word_range)

    # avg_tr_err, avg_dev_err = train_and_report(trainfile, devfile, dictionary, True, gram=2)
    # avg_tr_err, avg_dev_err = [0 for i in range(len(tr_err))], [0 for i in range(len(tr_err))]

    epochs = len(tr_err)
    if plot:
        import matplotlib.pyplot as plt
        plt.plot( range( epochs ), tr_err )
        plt.plot( range( epochs ), dev_err )
        plt.plot( range( epochs ), avg_tr_err )
        plt.plot( range( epochs ), avg_err )
        plt.legend( ('Perc. Training', 'Perc. Dev', 'Avg. Training', 'Avg. Dev'), numpoints = 1)
        plt.show()
