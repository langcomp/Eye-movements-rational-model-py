import numpy as np
import pandas as pd

from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.stats import entropy

class Vocabulary:
    """ Store relevant language knowledge of all true words in a vocabulary.

    Attributes:
        characters: A string of all possible characters.
        wlen: An integer indicating word length.
        nchar: An integer indicating number of characters.
        input_file: A string indicating input file name (csv) of word information.
        words: A list of (word, logfreq) tuples.
        vocab_dict: A dictionary of {word: index} storing index of words.
        init_x: A np.ndarry indicating prior.
        dim: An integer indicating the dimension of one-hot vector of a word.
        vocab_size: An integer couunt of all words in vocabulary.
        pos_word_chr: A np.ndarry of size (wlen, vocab_size, nchar) storing one-hot vector of each character at each position for each word.
    """
    def __init__(self, characters, wlen, input_file):
        """Inits Vocabulary with characters, word length, and input file."""
        self.characters = "".join(set(characters))
        self.wlen = wlen
        self.nchar = len(self.characters)

        csv_df = pd.read_csv(input_file)
        assert "word" in csv_df.columns, input_file + " should have a column named `word`."
        assert "logfreq" in csv_df.columns, input_file + " should have a column named `logfreq`."

        self.words = []
        _wordset = set()
        for _, row in csv_df.iterrows():
            w = row["word"]
            if len(w) == wlen and all(ch in characters for ch in w) and (w not in _wordset):
                self.words.append((row["word"], row["logfreq"]))
                _wordset.add(w)
        assert len(self.words) >= 2, "Class Vocabulary requires at least two words of word length %d. You only have %d." % (wlen, len(self.words))
        
        self.vocab_dict = {}
        for i, w in enumerate(self.words):
            (word, _) = w
            self.vocab_dict[word] = i

        init_x0 = np.log(lognormalize([w[1] for w in self.words]))
        self.init_x = init_x0[:-1] - init_x0[-1]

        self.dim = self.nchar * self.wlen
        self.vocab_size = len(self.words)

        self.pos_word_chr = self.get_onehot_chr_rep()

    def get_onehot_chr_rep(self):
        """Get one-hot representation of each character at each position for each word."""
        pos_word_chr = np.zeros([self.wlen, self.vocab_size, self.nchar])       
        for w in self.vocab_dict:
            for pos, ch in enumerate(w):
                pos_word_chr[pos, self.vocab_dict[w], self.characters.index(ch)] = 1
        return(pos_word_chr)

class OneVirtualReader:
    """ A virtual reader with a vocabulary and some visual properties.

    Attributes:
        vocabulary: A Vocabulary class storing language knowledge.
        sigma_scale: A float that controls shape of visual acuity function; small number indicates narrow acuity function.
        Lambda_scale: A float that controls overall visual input quality; small number indicates poor visual input.
        lpos_range: A list of numbers indicating landing position range.
        c, B, A: Dictionaries storing pre-computed parameters for computing visual input samples.
    """    
    def __init__(self, vocabulary, sigma_scale, Lambda_scale, lpos_range):
        """Inits OneVirtualReader with vocabulary and visual parameters."""
        self.vocab = vocabulary
        self.sigma_scale = sigma_scale
        self.Lambda_scale = Lambda_scale
        self.lpos_range = lpos_range
        
        self.c, self.B, self.A = self.get_cBA()

    def lambda_eccentricity(self, ecc):
        """ Compute eccentricity at a position.

        Args:
            ecc (int/float): A number indicating the distance of a character to fixation (in terms of number of characters).

        Returns:
            A float, which is the eccentricity at loc according to our visual acuity function.
        """
        sigma_L = 2.41 * self.sigma_scale 
        sigma_R = 3.74 * self.sigma_scale    
        sigma = (sigma_L + sigma_R) / 2 # using a symmetrical, normally-distributed acuity function
        
        Z = np.sqrt(2*np.pi)*sigma
        I = quad(lambda x: 1/Z * np.exp(-x*x/(2*sigma*sigma)), ecc - 0.5, ecc + 0.5)
        return(I[0])       

    def get_inv_SIGMA(self, lpos):
        """ Compute SIGMA^(-1), a diagonal covariance matrix indicating visual input quality of each letter position relative to fixation position lpos.

        Args:
            lpos (int/float): A number indicating the landing position of a fixation.

        Returns:
            inv_SIGMA (np.ndarray): A diagnal matrix of size (wlen*nchar, wlen*nchar).
        """
        ecc_per_pos = [self.lambda_eccentricity(i - lpos) for i in range(1, self.vocab.wlen + 1)]
        diag_ecc  = np.repeat(ecc_per_pos, len(self.vocab.characters))
        inv_SIGMA = self.Lambda_scale * np.diag(diag_ecc)
        return(inv_SIGMA)

    def word2visvec(self, word):
        """ Convert a word into a vector that concatenats one-hot vectors of each character.

        Args:
            word (str): A string.

        Returns:
            vec (np.ndarray): A vector of size (1, nchar * wlen).
        """
        vec = np.zeros((1, self.vocab.nchar * self.vocab.wlen))
        
        for pos, ch in enumerate(word):
            assert ch in self.vocab.characters, "Unable to convert word %s to vector: at least one character not in characters" % word
            ichar = self.vocab.characters.index(ch)
            vec[0, pos * self.vocab.nchar + ichar] = 1
            
        return(vec)

    def get_cBA(self):
        """ Compute visual parameters c, B, and A.

        We have c[i] = (y'[v]*inv_SIGMA*y[v] - y'[i]*inv_SIGMA*y[i])/2, 
        B[i,:] = (y[i] - y[v])'*inv_SIGMA, and
        A = B*SIGMA^(1/2),
        such that delta_x[i] ~ Gaussian(c[i] + B[i,:] * y[i], A*A').

        Returns:
            c (np.ndarray): A matrix of size (n_words - 1,).
            B (np.ndarray): A matrix of size (n_words - 1, wlen * nchar).
            A (np.ndarray): A matrix of size (n_words - 1, wlen * nchar).
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
            inv_SIGMA = self.get_inv_SIGMA(lpos)
            c[lpos] = np.diag((yv.dot(inv_SIGMA).dot(np.transpose(yv)) -
                             wordvec_mat[:-1, :].dot(inv_SIGMA).dot(np.transpose(wordvec_mat[:-1, :])))/2)
            B[lpos] = (wordvec_mat[:-1, :] - yv).dot(inv_SIGMA)
            A[lpos] = np.dot(B[lpos], sqrtm(np.linalg.inv(inv_SIGMA)))   

        return(c, B, A)
        
    def get_delta_x(self, word, lpos):
        """ Get a random visual sample for belief updating if fixating `word` at `lpos`.

        Args:
            word (str): A string.
            lpos (float): A number indicating fixation position.

        Returns:
            delta_x (np.ndarray): A vector indicating change of posterior logodds after getting a visual sample.
        """        
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
    """ Class of a fixation dwelling at a location (in terms of characters) for a period of time.

    Attributes:
        lpos: A number indicating landing position in terms of characters.
        fix_dur: A number indicating fixation duration.
    """
    def __init__(self, lpos, fix_dur):
        self.lpos = lpos
        self.fix_dur = fix_dur

class OneTrial:
    """ Class of a trial of identifying a word.

    Attributes:
        reader: A OneVirtualReader with a vocabulary and some visual properties.
        word: A string indicating the true word to be identified.
        x: A vector indicating posterior distribution log-odds.
        elapsed_time: An integer indicating time steps elapsed.
        fix_loc: A number indicating current fixation location.
    """    
    def __init__(self, reader, word):
        """Inits OneTrial with a OneVirtualReader and a word to identify."""
        self.reader = reader
        self.word = word
        
        self.x = self.reader.vocab.init_x
        self.elapsed_time = 0
        self.fix_loc = None

    def update_posterior_one_fixation(self, fixation):
        """ Update belief after performing a fixation.

        Args:
            fixation: A OneFixation class having lpos and fix_dur attributes.

        """         
        for t in range(fixation.fix_dur):
            delta_x = self.reader.get_delta_x(self.word, fixation.lpos)
            self.x = self.x + delta_x
        self.elapsed_time += fixation.fix_dur
        self.fix_loc = fixation.lpos
        
    def update_posterior_scan_path(self, scan_path):
        """ Update belief after performing a list of fixations.

        Args:
            scanpath: A list of OneFixation objects.

        """         
        for fixation in scan_path:
            self.update_posterior_one_fixation(fixation)
            
    def get_postH(self):
        """ Get the posterior entropy of log-odds.

        Returns:
            postH (float): Entropy of posterior.
        """
        pp = logodds2p(self.x)
        postH = entropy(pp, base = 2)
        return(postH)
    
    def get_prob_true_word(self):
        """ Get the probability of the true word.

        Returns:
            true_p (float): Probability of the word.
        """
        pp = logodds2p(self.x)
        index_true_word = self.reader.vocab.vocab_dict[self.word]
        true_p = pp[index_true_word]
        return(true_p)

    def get_max_prob_per_pos(self):
        """ Get the max probability at each position.

        Returns:
            pos_p (np.ndarray): A vector of the max probab of characters at each position, size (wlen,).
        """
        pp = logodds2p(self.x)
        pos_word_chr = self.reader.vocab.pos_word_chr
        pos_chr = np.dot(np.array(pp), pos_word_chr)
        pos_p = np.max(pos_chr, axis = 1)
        return(pos_p)

    def before_max_time(self, max_time):
        return(self.elapsed_time < max_time)

    def alpha_beta_policy(self, alpha, beta, max_time):
        max_prob_chr = self.get_max_prob_per_pos()
        prob_current = max_prob_chr[self.fix_loc - 1]
        
        while self.before_max_time(max_time) and prob_current < alpha: # keep fixating current fix_loc
            self.update_posterior_one_fixation(OneFixation(self.fix_loc, 1))
            max_prob_chr = self.get_max_prob_per_pos()
            prob_current = max_prob_chr[self.fix_loc - 1]

        left = list(max_prob_chr[:(self.fix_loc-1)])
        right = list(max_prob_chr[self.fix_loc:])

        if prob_current < alpha: # fail to recognize current character
            return(None)
        else:
            if any(i < beta for i in left): # refixate leftward
                ind = max([(i,j) for i,j in enumerate(left) if j < beta])[0]
                return(ind + 1)
            elif any(i < alpha for i in right): # refixate rightward
                ind = min([(i,j) for i,j in enumerate(right) if j < alpha])[0]
                return(ind + self.fix_loc + 1)
            else: # move to next word
                return(self.reader.vocab.wlen + 2)

class OneBlock:
    """ Class of a block of trials.

    Attributes:
        reader: A OneVirtualReader with a vocabulary and some visual properties.
        trial_list: A list of trial information stored in dictionaries. Each dictionary has a `word` and a `scanpath` key.
    """     
    def __init__(self, reader, trial_list):
        """Inits OneBlock with a OneVirtualReader and a list of trials."""        
        self.reader = reader
        self.trial_list = trial_list

    def get_block_metrics(self):
        ls_ptrue, ls_postH = [],[]
        
        for trial in self.trial_list:
            tmp_trial = OneTrial(self.reader, trial["word"])
            tmp_trial.update_posterior_scan_path(trial["scan_path"])
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

def add_sac_err_swift_random(target, launch):
    sigma0 = 0.87
    sigma1 = 0.084
    sigma = sigma0 + sigma1*np.abs(target - launch)
    actpos = np.round(np.random.normal(target, sigma, 1)[0])
    return(actpos)

def add_sac_err_mrchips(target, launch):
    sac_len = np.abs(target - launch)
    sigma = sac_len * 0.3
    actpos = np.round(np.random.normal(target, sigma, 1)[0])
    return(actpos)

def add_sac_err_ezr_random(target, launch):
    sigma0 = 0.5
    sigma1 = 0.15
    sigma = sigma0 + sigma1*np.abs(target - launch)
    actpos = np.round(np.random.normal(target, sigma, 1)[0])
    return(actpos)
