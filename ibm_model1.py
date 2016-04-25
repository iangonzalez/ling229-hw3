#!/usr/bin/env python
import optparse, sys, os, logging, itertools
from collections import defaultdict
# 500 its, no extras: AER 0.51
# 500 its, opt 1 and 2: 0.41 AER
# 500 its, opt 1 only: 0.40 AER

def init_opts(): 
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-r", "--rounds", dest="em_rounds", default=4, type="int", help="rounds of EM algorithm to do.")

    # extensions 
    optparser.add_option("-1", action="store_true", dest="e1", default=False, help="Include extension 1")
    optparser.add_option("-2", action="store_true", dest="e2", default=False, help="Include extension 2: adding pos tags")
    optparser.add_option("-3", action="store_true", dest="e3", default=False, help="Include extension 3: IBM Model 2")

    # debug 
    optparser.add_option("-D", "--debug", dest="DEBUG", default=False, help="Debugging option")

    (opts, _) = optparser.parse_args()
    return opts 


def train(opts, bitext):

    # initialize to uniform distribution
    l1 = set(reduce(lambda x, y: x+y, [f for (f,e) in bitext]))
    l2 = set(reduce(lambda x, y: x+y, [e for (f,e) in bitext]))

    # EM algorithm
    translation_probs = defaultdict(float)
    counts = defaultdict(float)

    unif_prob = float(1.0/len(l2))
    for f_i in l1:
        for e_j in l2:
            translation_probs[(f_i, e_j)] = unif_prob
            counts[(f_i, e_j)] = 0.0

    # execute em the number of times specified
    for i in range(opts.em_rounds):
        
        sys.stderr.write("Running E step " + str(i) +"\n")
        # E step
        for (n, (f, e)) in enumerate(bitext):

            for f_i in f:
                trans_prob_sum = sum([translation_probs[(f_i, e_j)] for e_j in e])

                for e_j in e:
                    alignment_prob = translation_probs[(f_i, e_j)] / trans_prob_sum
                    counts[(f_i, e_j)] += alignment_prob

        if opts.DEBUG:
            for f_i in l1:
                for e_j in l2:
                    print((f_i, e_j), counts[(f_i, e_j)])

        sys.stderr.write("Running M step " + str(i) +"\n")
        
        # M step: renormalize counts
        for e_j in l2:
            norm_sum = (sum([counts[(f_x, e_j)] for f_x in l1]))
            for f_i in l1:
                new_trans_prob = counts[(f_i, e_j)] / norm_sum
                translation_probs[(f_i, e_j)] = new_trans_prob

        # reset counts for next step
        for f_i in l1:
            for e_j in l2:
                counts[(f_i, e_j)] = 0.0

        sys.stderr.write("Completed step " + str(i) + "\n")

    return translation_probs

def decode(bitext, translation_probs):
    # decoding correct alignments
    alignments = list()

    for k, (f, e) in enumerate(bitext): 
        s = set()
        for i, f_i in enumerate(f): 
            bestp = 0 
            bestj = 0 
            for j, e_j in enumerate(e): 
                trans = translation_probs[ ( f_i, e_j ) ]
                if (trans > bestp): 
                    bestp = trans
                    bestj = j
            s.add((i, bestj))
        alignments.append(s)

    return alignments


def print_alignments(bitext, alignments): 
    for k in range(len(alignments)):
        for (i, j) in alignments[k]: 
            sys.stdout.write("%i-%i " % (i, j))
        sys.stdout.write("\n")

def reverse_bitext(bitext): 
    for (f, e) in bitext: 
        yield (e, f)


def intersect(a1, a2): 
    assert len(a1) == len(a2)
    
    alignments = list() 
    for i in range(len(a1)): 
        alignments.append(a1[i].intersection(a2[i]))

    return alignments


def main(): 
    # intialize the options for the script
    opts = init_opts()

    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    if opts.e2:
        e_data = os.path.join(opts.datadir, "english_tagged1000.txt")

    if opts.DEBUG: 
        f_data = opts.datadir + "/frenchtest.txt"
        e_data = opts.datadir + "/englishtest.txt"

    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    sys.stderr.write("Training with IBM model 1...\n")
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    
    
    translation_probs = train(opts, bitext)
    alignments = decode(bitext, translation_probs)
    
    if opts.e1: # set intersection of decoded alignments 
        bitext_reverse = list(reverse_bitext(bitext))
        # print bitext_reverse
        translation_probs_2 = train(opts, bitext_reverse)
        decoded = decode(bitext_reverse, translation_probs_2)
        alignments_2 = list()
        for a in decoded: 
            alignments_2.append(set (map( (lambda x: tuple(reversed(x))),  a )))
        alignments = intersect(alignments, alignments_2)

    print_alignments(bitext, alignments)

if __name__ == "__main__": 
    main()

"""
Documentation: According to the paper provided by the assignment spec
(http://aclweb.org/anthology/P/P04/P04-1066.pdf),  
we determine the number of null words to add to the source sentence using the following formula: 
Add a fixed number of null words per sentence. 
In our implementation, we are using 3. 
At each iteration of E/M, 
  multiple the transition probabilities for the null word by the number of null words per sentence
  to obtain the new transition probabilities for the null word 

"""

# better parameter initialization 
"""
Documentation: According to the papr provided by the assignment spec, 
(http://aclweb.org/anthology/P/P04/P04-1066.pdf), 
in order to improve parameter initialization, we 

"""


