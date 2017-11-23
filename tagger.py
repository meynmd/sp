#!/usr/bin/env python

from __future__ import division
from collections import defaultdict
import sys
from math import log
from itertools import product
startsym, stopsym = "<s>", "</s>"

def readfile(filename):
    for line in open(filename):
        wordtags = map(lambda x: x.rsplit("/", 1), line.split())
        yield [w for w,t in wordtags], [t for w,t in wordtags] # (word_seq, tag_seq) pair


def mle(filename): # Max Likelihood Estimation of HMM
    twfreq = defaultdict(lambda : defaultdict(int))
    ttfreq = defaultdict(lambda : defaultdict(int)) 
    tagfreq = defaultdict(int)    
    dictionary = defaultdict(set)

    for words, tags in readfile(filename):
        last = startsym
        tagfreq[last] += 1
        for word, tag in zip(words, tags) + [(stopsym, stopsym)]:
            #if tag == "VBP": tag = "VB" # +1 smoothing
            twfreq[tag][word] += 1            
            ttfreq[last][tag] += 1
            dictionary[word].add(tag)
            tagfreq[tag] += 1
            last = tag            
    
    model = defaultdict(float)
    num_tags = len(tagfreq)
    for tag, freq in tagfreq.iteritems(): 
        logfreq = log(freq)
        for word, f in twfreq[tag].iteritems():
            model[tag, word] = log(f) - logfreq 
        logfreq2 = log(freq + num_tags)
        for t in tagfreq: # all tags
            model[tag, t] = log(ttfreq[tag][t] + 1) - logfreq2 # +1 smoothing
        
    return dictionary, model


def my_decode_var_gram(words, dictionary, model, gram=3):
    pi = defaultdict(lambda : float("-inf"))
    bp = {}
    words = [startsym] + words + [stopsym]
    if len(words) < gram:
        gram = len(words)

    def get_trans_score(tags):
        tscore = 0.
        for j in range(len(tags)):
            tscore = model[tags[j:]]
            if tscore != 0.:
                return tscore
        return tscore

    pi[(0,) + tuple([startsym for g in range(gram - 1)])] = 1.
    for ngram in product(*[list(dictionary[w]) for w in words[1:gram]]):
        pi[(gram-1,) + ngram] = get_trans_score(ngram) + model[ngram[-1], words[gram-1]]

    def update_pi(k, tags):
        optimal = None
        for t in dictionary[words[k - gram + 1]]:
            score = pi[(k-1, t) + tags[:-1]] + get_trans_score((t,) + tags) + model[tags[-1], words[k]]
            if score > pi[(k,) + tags]:
                optimal = t
                pi[(k,) + tags] = score
        return optimal

    def backtrack(i, prev_tags):
        if i == gram - 1:
            return list(prev_tags)
        return backtrack(i-1, (bp[(i,) + prev_tags],) + prev_tags[:-1]) + [prev_tags[-1]]

    for i, word in enumerate(words[gram:], gram):
        for seq in product(*[dictionary[w] for w in words[i - gram + 2 : i + 1]]):
            bp[(i,) + seq] = update_pi( i, seq )

    last = max(product(*[list(dictionary[w]) for w in words[-gram+1:]]),
               key=lambda s : pi[(len(words)-1,) + s])
    return backtrack(len(words)-1, last)[:-1]


def test(filename, dictionary, model, gram=2):
    errors = tot = 0
    for words, tags in readfile(filename):
        # mytags = liangs_decode(words, dictionary, model, gram)
        mytags = my_decode_var_gram(words, dictionary, model, gram)
        errors += sum(t1!=t2 for (t1,t2) in zip(tags, mytags))
        tot += len(words) 
        
    return errors/tot


if __name__ == "__main__":
    trainfile, devfile = sys.argv[1:3]
    dictionary, model = mle(trainfile)
    dictionary[startsym].add(startsym)
    dictionary[stopsym].add(stopsym)

    # print "train_err {0:.2%}".format(test(trainfile, dictionary, model))
    # print "dev_err {0:.2%}".format(test(devfile, dictionary, model))

    for words, tags in readfile(devfile):
        print words
        print my_decode(words, dictionary, model, 3)
        print liangs_decode(words, dictionary, model)
