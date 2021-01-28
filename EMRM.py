import json
import numpy as np
import pandas as pd

from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.stats import entropy

# ==============================================================
# ===================== Global variables =======================
# ==============================================================
_global_vars_fn = "./data/global_vars.json"
_global_vars = json.loads(open(_global_vars_fn, encoding = "utf-8").read())

CHARACTERS   = _global_vars["CHARACTERS"]
WLEN         = _global_vars["WLEN"]
SIGMA_SCALE  = _global_vars["SIGMA_SCALE"]
LAMBDA_SCALE = _global_vars["LAMBDA_SCALE"]
NCHAR        = len(CHARACTERS)

# ==============================================================
# ===================== Visual parameters ======================
# ==============================================================
def word2visvec(word):
    """ Convert a word into a vector that concatenats one-hot vectors of each character.

    Args:
        word (str): A string.

    Returns:
        vec (np.ndarray): A vector of size (1, NCHAR * WLEN).
    """
    vec = np.zeros((1, NCHAR * WLEN))
    
    for ic, c in enumerate(word):
        assert c in CHARACTERS, "Unable to convert word %s to vector: at least one character not in CHARACTERS" % word
        ichar = CHARACTERS.index(c)
        vec[0, ic * NCHAR + ichar] = 1
        
    return(vec)
        
def _get_wordvec_mat(words):
    """ Stack vectors of all words in the vocabulary into a matrix.

    Args:
        words (List[word(str), word_freq(float)]): A list of words (if you don't need word_freq, just set them to 0).

    Returns:
        wordvec_mat (np.ndarray): A matrix of size (n_words, NCHAR * WLEN), each row being a word's vector.
    """
    vec_list = []
    for w, _ in words:
        vec = word2visvec(w)
        vec_list.append(vec)
    wordvec_mat = np.concatenate(vec_list, axis = 0)
    return(wordvec_mat)

def lambda_eccentricity(loc):
    """ Compute eccentricity at a position.

    Args:
        loc (int/float): A number indicating the distance of a character to fixation (in terms of number of characters).

    Returns:
        A float, which is the eccentricity at loc according to our visual acuity function.
    """
    sigma_L = 2.41 * SIGMA_SCALE 
    sigma_R = 3.74 * SIGMA_SCALE    
    sigma = (sigma_L + sigma_R) / 2 # using a symmetrical, normally-distributed acuity function
    
    Z = np.sqrt(2 * np.pi) * sigma
    I = quad(lambda x: 1/Z * np.exp(-x*x/(2*sigma*sigma)), loc - 0.5, loc + 0.5)
    return(I[0])

def get_SIGMA(lpos):
    """ Compute SIGMA^(-1), a diagonal covariance matrix indicating visual input quality of each letter position relative to fixation position lpos.

    Args:
        lpos (int/float): A number indicating the landing position of a fixation.

    Returns:
        inv_SIGMA (np.ndarray): A diagnal matrix of size (WLEN*NCHAR, WLEN*NCHAR).
    """
    ecc_per_pos = [lambda_eccentricity(i - lpos) for i in range(1, WLEN + 1)]
    diag_ecc  = np.repeat(ecc_per_pos, NCHAR)
    inv_SIGMA = LAMBDA_SCALE * np.diag(diag_ecc)
    return(inv_SIGMA)

def _get_cBA(wordvec_mat, inv_SIGMA):
    """ Compute visual parameters c, B, and A.

    We have c[i] = (y'[v]*inv_SIGMA*y[v] - y'[i]*inv_SIGMA*y[i])/2, 
    B[i,:] = (y[i] - y[v])'*inv_SIGMA, and
    A = B*SIGMA^(1/2),
    such that delta_x[i] ~ Gaussian(c[i] + B[i,:] * y[i], A*A').

    Args:
        wordvec_mat (np.ndarray): A matrix of size (n_words, NCHAR * WLEN), where each row represents a vector of a word.
        inv_SIGMA (np.ndarray): A diagnal matrix indicating visual input quality of each position in a word.

    Returns:
        c (np.ndarray): A matrix of size (n_words - 1,).
        B (np.ndarray): A matrix of size (n_words - 1, WLEN * NCHAR).
        A (np.ndarray): A matrix of size (n_words - 1, WLEN * NCHAR).
    """
    ## Wordvec_mat is a v-by-d matrix, each row is the vector of a word
    ## By default, the last word in the wordvec_mat is the baseline word
    [v,d] = wordvec_mat.shape
    yv = wordvec_mat[v-1] # last word serving as baseline
    
    ## Calculating delta_x: c and B
    c = np.zeros(v-1)     # c is a v-1 dimensional vector
    B = np.zeros([v-1,d]) # B is a v-1 -by-d dimensional matrix

    for i in range(v-1):
        yi = wordvec_mat[i]
        c[i] = (yv.dot(inv_SIGMA).dot(np.transpose(yv))- yi.dot(inv_SIGMA).dot(np.transpose(yi)))/2
        B[i] = (yi - yv).dot(inv_SIGMA)

    A = np.dot(B, sqrtm(np.linalg.inv(inv_SIGMA)))   
    return (c, B, A)

def get_visual_params(words, lpos_range):
    """ Compute visual parameters for all possible landing positions.

    Args:
        words (List[word(str), word_freq(float)]): A list of words (if you don't need word_freq, just set them to 0).
        lpos_range (List[int/float]): A list of possible landing positions.

    Returns:
        c (dict): A dict of format {lpos: c[lpos]}, where each key is a landing position, and its value is visual parameter c at this position.
        B (dict): A dict of format {lpos: B[lpos]}, where each key is a landing position, and its value is visual parameter B at this position.
        A (dict): A dict of format {lpos: A[lpos]}, where each key is a landing position, and its value is visual parameter A at this position.
    """
    wordvec_mat = _get_wordvec_mat(words)
    c, B, A = {},{},{}
    
    for lpos in lpos_range:
        inv_SIGMA = get_SIGMA(lpos)
        (c[lpos], B[lpos], A[lpos]) = _get_cBA(wordvec_mat, inv_SIGMA)
    return(c, B, A)

# ==============================================================
# =================== Linguistic knowledge =====================
# ==============================================================
def csv2words(csv_fn):
    """ Read in csv file of words and reformat as a list with elements of (word, freq).

    Args:
        csv_fn (str): File name of a csv containing vocabulary info. The csv file must contain a column named "word" and a column named "logfreq".

    Returns:
        words (List[word(str), word_freq(float)]): A list of words and freq.
    """
    csv_df = pd.read_csv(csv_fn)
    assert "word" in csv_df.columns, csv_fn + " should have a column named `word`."
    assert "logfreq" in csv_df.columns, csv_fn + " should have a column named `logfreq`."
    
    words = []
    for i,row in csv_df.iterrows():
        words.append((row["word"], row["logfreq"]))
    return(words)
    

def get_vocabulary(words):
    """ Prepare necessary word-indices and prior from words, and save into a dict.

    Args:
        words (List[word(str), word_freq(float)]): A list of words and corresponding frequency.

    Returns:
        D (dict): A dict containing linguistic knowledge.
    """
    vocab = {}
    for iw, w in enumerate(words):
        vocab[w[0]] = iw
        
    init_x0 = np.log(lognormalize([w[1] for w in words])) # np.log(10) * 
    init_x = init_x0[:-1] - init_x0[-1]

    D = {"vocab": vocab,
         "d": NCHAR * WLEN,
         "v": len(words),
         "init_x": init_x,
         }
    return(D)

# ==============================================================
# =========== Run simulation & get posterior metrics ===========
# ==============================================================
def run_single_word_identify(c, B, A, D, word, lpos, time = 1):
    """ Run Bayesian belief updating and get posterior.

    Args:
        c (dict): A dict of format {lpos: c[lpos]}.
        B (dict): A dict of format {lpos: B[lpos]}.
        A (dict): A dict of format {lpos: A[lpos]}.
        D (dict): A dict containing linguistic knowledge.
        word (str): A string, the true word to be identified.
        lpos (int/float): A number indicating the position of fixation.
        time (int): An integer indicating how many time steps to take. Default is 1.

    Returns:
        x (np.ndarray): A vector indicating posterior log-odds, size is (n_words-1,).
    """
    d = D["d"]
    x = D["init_x"]

    lpos_c = c[lpos]
    lpos_B = B[lpos]
    lpos_A = A[lpos]
    
    yT = word2visvec(word)[0]
    mu = lpos_c + np.dot(lpos_B, yT)

    for t in range(time):
        standard_sample = np.random.normal(0, 1, d)
        delta_x = mu + np.dot(lpos_A, standard_sample)
        x = x + delta_x

    return(x)

def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return(np.exp(x - a))

def logodds2p(x):
    """ Logodds to probability.

    Args:
        x (np.ndarray): Log=odds vector, size (n_words-1,).

    Returns:
        p (np.ndarray): Probability of each word, size (n_words,).
    """
    x2 = np.append(x,0)
    p = lognormalize(x2)
    return(p)

def get_prob_true_word(x, index_true_word):
    """ Get the probability of the true word.

    Args:
        x (np.ndarray): Log=odds vector, size (n_words-1,).
        index_true_word (int): Index of the true word (or whatever word you care).

    Returns:
        true_p (float): Probability of the word.
    """
    pp = logodds2p(x)
    true_p = pp[index_true_word]
    return(true_p)

def get_postH(x):
    """ Get the posterior entropy of log-odds.

    Args:
        x (np.ndarray): Log=odds vector, size (n_words-1,).

    Returns:
        postH (float): Entropy of posterior.
    """
    pp = logodds2p(x)
    postH = entropy(pp, base = 2)
    return(postH)

def get_max_prob_per_pos(x, vocab):
    """ Get the max probability at each position.

    Args:
        x (np.ndarray): Log=odds vector, size (n_words-1,).
        vocab (dict): A dictionary of {word: index_of _word}.

    Returns:
        pos_p (np.ndarray): A vector of the max probab of characters at each position, size (WLEN,).
    """
    pp = logodds2p(x)
    PWC_mat = np.zeros([WLEN, len(vocab), NCHAR])
    
    for w in vocab:
        for pos, ch in enumerate(w):
            PWC_mat[pos, vocab[w], CHARACTERS.index(ch)] = 1
            
    PC_mat = np.dot(np.array(pp), PWC_mat)
    pos_p = np.max(PC_mat, axis = 1)
    return(pos_p)
    
