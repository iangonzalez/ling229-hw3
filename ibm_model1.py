#!/usr/bin/env python
import optparse, sys, os, logging, itertools
from collections import defaultdict

DEBUG = False

optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-r", "--rounds", dest="em_rounds", default=4, type="int", help="rounds of EM algorithm to do.")
(opts, _) = optparser.parse_args()
# f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
# e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

f_data = opts.datadir + "/frenchtest.txt"
e_data = opts.datadir + "/englishtest.txt"

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

sys.stderr.write("Training with IBM model 1...\n")

bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

# EM algorithm

translation_probs = defaultdict(float)
counts = defaultdict(float)

# initialize to uniform distribution
all_foreign_words = set(reduce(lambda x, y: x+y, [f for (f,e) in bitext]))
all_english_words = set(reduce(lambda x, y: x+y, [e for (f,e) in bitext]))

unif_prob = float(1.0/len(all_english_words))

for f_i in all_foreign_words:
    for e_j in all_english_words:
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

    if DEBUG:
        for f_i in all_foreign_words:
            for e_j in all_english_words:
                print((f_i, e_j), counts[(f_i, e_j)])


    sys.stderr.write("Running M step " + str(i) +"\n")
    
    # M step: renormalize counts
    for e_j in all_english_words:
        norm_sum = (sum([counts[(f_x, e_j)] for f_x in all_foreign_words]))
        for f_i in all_foreign_words:
            new_trans_prob = counts[(f_i, e_j)] / norm_sum
            translation_probs[(f_i, e_j)] = new_trans_prob

    # reset counts for next step
    for f_i in all_foreign_words:
        for e_j in all_english_words:
            counts[(f_i, e_j)] = 0.0

    sys.stderr.write("Completed step " + str(i) + "\n")


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


# decoding correct alignments
for (f, e) in bitext: 
    for i, f_i in enumerate(f): 
        bestp = 0 
        bestj = 0 
        for j, e_j in enumerate(e): 
            trans = translation_probs[ ( f_i, e_j ) ]
            if (trans > bestp): 
                bestp = trans
                bestj = j
        sys.stdout.write("%i-%i " % (i, bestj))
    sys.stdout.write("\n")