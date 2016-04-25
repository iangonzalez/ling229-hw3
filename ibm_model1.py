#!/usr/bin/env python
import optparse, sys, os, logging, itertools,math
from collections import defaultdict
# 500 its, no extras: AER 0.51
# 500 its, opt 1 and 2: 0.41 AER
# 500 its, opt 1 only: 0.40 AER
import random 



"""
USAGE FOR THIS SCRIPT:
This script outputs alignments for a given bitext based on a modified version of IBM model 1.
This script takes all of the arguments taken by default.py and outputs alignments in the same format.
Some options have been added:
    -r: number of rounds for the EM algorithm
    -1: Extension 1. This corresponds to the first suggestion in the assignment, which does p(f|e) then p(e|f)
        and uses the intersection of the alignments calculated as the result.
    -2: Extension 2. This extension uses a POS-tagged version of the first 1000 english sentences instead of 
        the untagged version to calculate alignments.
    -3: Extension 3. This extension runs IBM model 2 after initalizing its translation probabilities from model 1
        and outputs the alignments that maximize its parameters.
"""

def negative_log(num):
    return -1 * math.log(num)

# Initialize the option parser and parse the arguments passed in by the user
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

# Function for training ibm model 1.
def train_ibm_model1(opts, bitext):
    """
    This implements the ibm model 1 EM training algorithm as described in the class lecture slides.
    Translation probabilities are initialized uniformly. In the E step, alignment probabilities are
    calculated based on word co-occurence. In the M step, these are renormalized. 
    (more details on p.57 of first MT lecture notes).

    Params: OptionParser object and the bitext.
    Output: translation probabilities in a dictionary mapping tuples to floats
    """

    # get the unique words in the bitext
    l1 = set(reduce(lambda x, y: x+y, [f for (f,e) in bitext]))
    l2 = set(reduce(lambda x, y: x+y, [e for (f,e) in bitext]))

    # EM algorithm
    translation_probs = defaultdict(float)
    counts = defaultdict(float)

    # initialize to uniform distribution
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
            # compute initial alignment probabilities
            for f_i in f:
                trans_prob_sum = sum([translation_probs[(f_i, e_j)] for e_j in e])

                for e_j in e:
                    alignment_prob = translation_probs[(f_i, e_j)] / trans_prob_sum
                    counts[(f_i, e_j)] += alignment_prob

        #print stuff in debug mode
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
    """
    Decodes the translation probabilities from IBM model 1 to produce alignments for the bitext.
    This follows the algorithm outlined in the assignment exactly.

    Params: bitext and the translation probabilities from IBM model 1.
    Output: alignments as list of tuples
    """

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

# reverse order of languages in bitext
def reverse_bitext(bitext): 
    for (f, e) in bitext: 
        yield (e, f)

# Compute the set intersection of two given alignments (tuple sets)
def intersect(a1, a2): 
    assert len(a1) == len(a2)
    
    alignments = list() 
    for i in range(len(a1)): 
        alignments.append(a1[i].intersection(a2[i]))

    return alignments


def train_ibm_model2(opts, bitext, translation_probs=None): 
    # initialize to random values 
    l1 = set(reduce(lambda x, y: x+y, [f for (f,e) in bitext]))
    l2 = set(reduce(lambda x, y: x+y, [e for (f,e) in bitext]))


    """
    Can be supplied with initialized translation probabilities from 
    IBM model 1; otherwise, use a random distribution 
    """
    trans_probs = translation_probs
    if not translation_probs: 
        trans_probs = defaultdict(float)
    
    distortion_probs = defaultdict(float)
    counts = defaultdict(float)

    """
    Initialize with a random distribution from 0 to 1 
    """
    for (k, (f, e)) in enumerate(bitext):
        m_k = len(f)
        l_k = len(e)
        for (i, f_i) in enumerate(f):
            for (j, e_j) in enumerate(e):
                if not translation_probs: 
                    trans_probs[ (f_i, e_j) ] = random.uniform(0.0, 1.0)
                distortion_probs[ (j, i, l_k, m_k) ] = random.uniform(0.0, 1.0)

    """
    Execute E/M for given number of rounds 
    """
    for d in range(opts.em_rounds):
        sys.stderr.write("IBM model 2: Running E/M " + str(d) +"\n")
        
        # reset counts for next step
        for k in counts: 
            counts[k] = 0.0
        
        """
        E step: update counts for each of (e_j, f_i), 
        (e_j), (j|i, l, m) & (i, l, m) by incrementing 
        by delta (the production of the distortion & transition probabilities 
        for the word pair divided by the normalization factor)
        """
        for (k, (f, e)) in enumerate(bitext):
            m_k = len(f)
            l_k = len(e)
            for (i, f_i) in enumerate(f):
                norm = 0
                for (j, e_j) in enumerate(e): 
                    norm += distortion_probs[ (j, i, l_k, m_k) ] *  trans_probs[ (f_i, e_j) ]

                for (j, e_j) in enumerate(e): 
                    d_kij = (distortion_probs[ (j, i, l_k, m_k) ] *  trans_probs[ (f_i, e_j) ]) / norm 
                    counts[ (e_j, f_i) ] += d_kij 
                    counts[ (e_j) ] += d_kij 
                    counts[ (j, i, l_k, m_k) ] += d_kij 
                    counts[ (i, l_k, m_k) ] += d_kij 

        """
        M step: normalize using counts determined above. 
            translation probability updated by normalizing by total count of e_j vs counts of (e_j, f_i) pair; 
            distortion probability updated by normalizing by total count of (i, l, m) vs counts of (j|i, l, m)
        """
        for (k, (f, e)) in enumerate(bitext):
            m_k = len(f)
            l_k = len(e)
            for (i, f_i) in enumerate(f):
                for (j, e_j) in enumerate(e): 
                    newtrans = counts[ (e_j, f_i) ] / counts[ (e_j) ] 
                    distort = counts[ (j, i, l_k, m_k) ] / counts[ (i, l_k, m_k) ]
                    trans_probs[ (f_i, e_j) ] = newtrans 
                    distortion_probs[ (j, i, l_k, m_k) ] = distort

        sys.stderr.write("IBM model 2: Completed step " + str(d) + "\n")
    return trans_probs, distortion_probs

def decode_model2(bitext, translation_probs, distortion_probs): 

    """
    Decodes the correct alignment based on translation probabilities and distortion probabilities. 
    Uses the formula described in the paper; 
        determines the maximum of the product of the translation & distortion probabilities
        for words f_i and e_j in a sentence pair (f, e) 
    """
    alignments = list()

    for k, (f, e) in enumerate(bitext): 
        m_k = len(f)
        l_k = len(e)
        s = set()
        for i, f_i in enumerate(f):
            bestp = 0
            bestj = 0
            for j, e_j in enumerate(e): 
                prod = translation_probs[ ( f_i, e_j ) ] * distortion_probs[ (j, i, l_k, m_k) ]

                if (prod > bestp): 
                    bestp = prod 
                    bestj = j
            s.add((i, bestj))
        alignments.append(s)

    return alignments

def main(): 
    # intialize the options for the script
    opts = init_opts()

    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    # extension 2, do the tagged english data
    if opts.e2:
        e_data = os.path.join(opts.datadir, "english_tagged1000.txt")

    if opts.DEBUG: 
        f_data = opts.datadir + "/frenchtest.txt"
        e_data = opts.datadir + "/englishtest.txt"

    # get bitext
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    bitext_reverse = list(reverse_bitext(bitext))
    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)


    sys.stderr.write("Training with IBM model 1...\n")
    translation_probs = train_ibm_model1(opts, bitext)
    if opts.e3: 
        translation_probs, distortion_probs = train_ibm_model2(opts, bitext, translation_probs)
        sys.stderr.write("Decoding with IBM model 2")
        alignments = decode_model2(bitext, translation_probs, distortion_probs)
    else: 
        alignments = decode(bitext, translation_probs)
    
    # with extension 1 do p(f|e) and then p(e|f), then compute intersection     
    if opts.e1: # set intersection of decoded alignments 
        sys.stderr.write("Using extension 1\n")
        
        # print bitext_reverse
        translation_probs_2 = train_ibm_model1(opts, bitext_reverse)
        decoded = decode(bitext_reverse, translation_probs_2)
        alignments_2 = list()
        for a in decoded: 
            alignments_2.append(set (map( (lambda x: tuple(reversed(x))),  a )))
        alignments = intersect(alignments, alignments_2)

    print_alignments(bitext, alignments)

if __name__ == "__main__": 
    main()
