import numpy as np
import pandas as pd

from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.stats import entropy

class Vocabulary:
    def __init__(self, CHARACTERS, WLEN, csv_fn):
        self.CHARACTERS = CHARACTERS
        self.WLEN = WLEN
        self.NCHAR = len(self.CHARACTERS)

        csv_df = pd.read_csv(csv_fn)
        assert "word" in csv_df.columns, csv_fn + " should have a column named `word`."
        assert "logfreq" in csv_df.columns, csv_fn + " should have a column named `logfreq`."

        words = []
        for i,row in csv_df.iterrows():
            w = row["word"]
            if len(w) == WLEN and all(ch in CHARACTERS for ch in w):
                words.append((row["word"], row["logfreq"]))
        assert len(words) >= 2, "Vocabulary requires at least two words of word length %d. You only have %d." % (WLEN, len(words))
        self.words = words
        
        vocab_dict = {}
        for iw, w in enumerate(words):
            vocab_dict[w[0]] = iw
        self.vocab_dict = vocab_dict

        init_x0 = np.log(lognormalize([w[1] for w in words]))
        init_x = init_x0[:-1] - init_x0[-1]
        self.init_x = init_x

        self.dim = self.NCHAR * self.WLEN
        self.vocab_size = len(self.words)

class OneVirtualReader:
    def __init__(self, vocabulary, SIGMA_SCALE, LAMBDA_SCALE, lpos_range):
        self.SIGMA_SCALE = SIGMA_SCALE
        self.LAMBDA_SCALE = LAMBDA_SCALE
        self.vocab = vocabulary
        self.lpos_range = lpos_range
        self.get_cBA()

    def lambda_eccentricity(self, ecc):
        """ Compute eccentricity at a position.

        Args:
            ecc (int/float): A number indicating the distance of a character to fixation (in terms of number of characters).

        Returns:
            A float, which is the eccentricity at loc according to our visual acuity function.
        """
        sigma_L = 2.41 * self.SIGMA_SCALE 
        sigma_R = 3.74 * self.SIGMA_SCALE    
        sigma = (sigma_L + sigma_R) / 2 # using a symmetrical, normally-distributed acuity function
        
        Z = np.sqrt(2*np.pi)*sigma
        I = quad(lambda x: 1/Z * np.exp(-x*x/(2*sigma*sigma)), ecc - 0.5, ecc + 0.5)
        return(I[0])       

    def get_SIGMA(self, lpos):
        """ Compute SIGMA^(-1), a diagonal covariance matrix indicating visual input quality of each letter position relative to fixation position lpos.

        Args:
            lpos (int/float): A number indicating the landing position of a fixation.

        Returns:
            inv_SIGMA (np.ndarray): A diagnal matrix of size (WLEN*NCHAR, WLEN*NCHAR).
        """
        ecc_per_pos = [self.lambda_eccentricity(i - lpos) for i in range(1, self.vocab.WLEN + 1)]
        diag_ecc  = np.repeat(ecc_per_pos, len(self.vocab.CHARACTERS))
        inv_SIGMA = self.LAMBDA_SCALE * np.diag(diag_ecc)
        return(inv_SIGMA)

    def word2visvec(self, word):
        """ Convert a word into a vector that concatenats one-hot vectors of each character.

        Args:
            word (str): A string.

        Returns:
            vec (np.ndarray): A vector of size (1, NCHAR * WLEN).
        """
        vec = np.zeros((1, self.vocab.NCHAR * self.vocab.WLEN))
        
        for ic, c in enumerate(word):
            assert c in self.vocab.CHARACTERS, "Unable to convert word %s to vector: at least one character not in CHARACTERS" % word
            ichar = self.vocab.CHARACTERS.index(c)
            vec[0, ic * self.vocab.NCHAR + ichar] = 1
            
        return(vec)

    def get_cBA(self):
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
        vec_list = []
        for w, _ in self.vocab.words:
            vec = self.word2visvec(w)
            vec_list.append(vec)
        wordvec_mat = np.concatenate(vec_list, axis = 0)
        
        ## Wordvec_mat is a v-by-d matrix, each row is the vector of a word
        ## By default, the last word in the wordvec_mat is the baseline word
        yv = wordvec_mat[-1, :] # last word serving as baseline

        c, B, A = {},{},{}
        for lpos in self.lpos_range:
            inv_SIGMA = self.get_SIGMA(lpos)
            c[lpos] = np.diag((yv.dot(inv_SIGMA).dot(np.transpose(yv)) -
                             wordvec_mat[:-1, :].dot(inv_SIGMA).dot(np.transpose(wordvec_mat[:-1, :])))/2)
            B[lpos] = (wordvec_mat[:-1, :] - yv).dot(inv_SIGMA)
            A[lpos] = np.dot(B[lpos], sqrtm(np.linalg.inv(inv_SIGMA)))   

        self.c, self.B, self.A = c, B, A
        
    def get_delta_x(self, word, lpos):
        lpos_c = self.c[lpos]
        lpos_B = self.B[lpos]
        lpos_A = self.A[lpos]
        
        yT = self.word2visvec(word)[0]
        mu = lpos_c + np.dot(lpos_B, yT)
        
        sample_dim = lpos_A.shape[1]
        standard_sample = np.random.normal(0, 1, sample_dim)
        delta_x = mu + np.dot(lpos_A, standard_sample)
        return(delta_x)
        
class OneFixation:
    def __init__(self, lpos, fix_dur):
        self.lpos = lpos
        self.fix_dur = fix_dur

class OneTrial:
    def __init__(self, reader, word):
        self.reader = reader
        self.word = word
        self.x = self.reader.vocab.init_x

    def update_posterior_one_fixation(self, fixation):
        for t in range(fixation.fix_dur):
            delta_x = self.reader.get_delta_x(self.word, fixation.lpos)
            self.x = self.x + delta_x
            
    def update_posterior_scan_path(self, scan_path):
        for fixation in scan_path:
            self.update_posterior_one_fixation(fixation)
            
    def get_postH(self):
        """ Get the posterior entropy of log-odds.

        Args:
            x (np.ndarray): Log=odds vector, size (n_words-1,).

        Returns:
            postH (float): Entropy of posterior.
        """
        pp = logodds2p(self.x)
        postH = entropy(pp, base = 2)
        return(postH)
    
    def get_prob_true_word(self):
        """ Get the probability of the true word.

        Args:
            x (np.ndarray): Log=odds vector, size (n_words-1,).
            index_true_word (int): Index of the true word (or whatever word you care).

        Returns:
            true_p (float): Probability of the word.
        """
        pp = logodds2p(self.x)
        index_true_word = self.reader.vocab.vocab_dict[self.word]
        true_p = pp[index_true_word]
        return(true_p)

    def get_max_prob_per_pos(self):
        """ Get the max probability at each position.

        Args:
            x (np.ndarray): Log=odds vector, size (n_words-1,).
            vocab (dict): A dictionary of {word: index_of _word}.

        Returns:
            pos_p (np.ndarray): A vector of the max probab of characters at each position, size (WLEN,).
        """
        pp = logodds2p(self.x)
        PWC_mat = np.zeros([WLEN, len(vocab), len(CHARACTERS)])
        
        for w in vocab:
            for pos, ch in enumerate(w):
                PWC_mat[pos, vocab[w], CHARACTERS.index(ch)] = 1
                
        PC_mat = np.dot(np.array(pp), PWC_mat)
        pos_p = np.max(PC_mat, axis = 1)
        return(pos_p)

class OneBlock:
    def __init__(self, reader, trial_list):
        self.reader = reader
        self.trial_list = trial_list

    def get_block_metrics(self):
        ls_ptrue, ls_postH = [],[]
        for i in range(len(self.trial_list)):
            tmp_trial_info = self.trial_list[i]
            
            tmp_trial = OneTrial(self.reader, tmp_trial_info["word"])
            tmp_trial.update_posterior_scan_path(tmp_trial_info["scan_path"])
            tmp_postH = tmp_trial.get_postH()
            tmp_ptrue = tmp_trial.get_prob_true_word()
            
            ls_ptrue.append(tmp_ptrue)
            ls_postH.append(tmp_postH)

        block_res = {"n_trial": len(self.trial_list),
                     "avg_ptrue": np.mean(ls_ptrue),
                     "avg_postH": np.mean(ls_postH)
                     }
        return(block_res)

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
