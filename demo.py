import string
import numpy as np
import pandas as pd
import EMRM
from EMRM import Vocabulary, OneVirtualReader, OneTrial, OneFixation, OneBlock

np.random.seed(0)
# =================================================================
# =========================== Quick start =========================
# =================================================================
### Get the probability of a word and entropy of posterior distribution
### after fixating the first letter of the word for 5 time steps.
print("--- Section 1: Quick start ---")
csv_fn7 = "data/example_vocab.csv"
csv_df = pd.read_csv(csv_fn7)
vocab_seven = Vocabulary(characters = string.ascii_lowercase,
                         wlen = 7,
                         input_df = csv_df)

demo_reader = OneVirtualReader(vocabulary = vocab_seven,
                               sigma_scale = 1,
                               Lambda_scale = 3,
                               fix_loc_list = [1,4,7] # limit possible fixation locations to positions 1, 4, and 7 to reduce computation.
                               )

my_word = "quality"
demo_trial = OneTrial(demo_reader, word = my_word)
demo_fixation = OneFixation(fix_loc = 1, fix_dur = 5)

demo_trial.update_posterior_one_fixation(demo_fixation)
postH = demo_trial.get_postH()
ptrue = demo_trial.get_prob_true_word()
print(f"Fixate `{my_word}` at letter {demo_fixation.fix_loc} for {demo_fixation.fix_dur} steps: p({my_word}) = {ptrue:.2f}, postH = {postH:.2f}.")

# =================================================================
# ===================== Check model behaviors =====================
# =================================================================
print("--- Section 2: Check model behaviors ---")

### Visual parameters
# 2.1
print("--- 2.1: Influence of sigma_scale (visual acuity function)  ---")
sigma_scale_list = [0.5, 2] # narrow, wide visual acuity
for sigma in sigma_scale_list:
    reader_tmp = OneVirtualReader(vocabulary = vocab_seven,
                                  sigma_scale = sigma,
                                  Lambda_scale = 3,
                                  fix_loc_list = [1,7]
                                  )
    
    for fix_loc in reader_tmp.fix_loc_list:
        fixation_tmp = OneFixation(fix_loc = fix_loc, fix_dur = 5)
        my_trials = [{"word": my_word, "scan_path": [fixation_tmp]}] * 100

        demo_block = OneBlock(reader_tmp, my_trials)
        res = demo_block.get_block_metrics()
        print(f"Fixate {fix_loc}, sigma_scale = {sigma}: p({my_word}) = {res['avg_ptrue']:.2f}, postH = {res['avg_postH']:.2f}.")

# 2.2
print("--- 2.2: Influence of Lambda_scale (overall visual input quality)  ---")
Lambda_scale_list = [1, 10] # poor, good overall visual-input quality
for Lambda in Lambda_scale_list:
    reader_tmp = OneVirtualReader(vocabulary = vocab_seven,
                                   sigma_scale = 1,
                                   Lambda_scale = Lambda,
                                   fix_loc_list = [1,7]
                                   )
    
    for fix_loc in reader_tmp.fix_loc_list:
        fixation_tmp = OneFixation(fix_loc = fix_loc, fix_dur = 5)
        my_trials = [{"word": my_word, "scan_path": [fixation_tmp]}] * 100

        demo_block = OneBlock(reader_tmp, my_trials)
        res = demo_block.get_block_metrics()
        print(f"Fixate {fix_loc}, Lambda_scale = {Lambda}: p({my_word}) = {res['avg_ptrue']:.2f}, postH = {res['avg_postH']:.2f}.")

### High freq vs. low freq words
# 2.3
print("--- 2.3: It's easier to identify high-freq words (`quality`) than low-freq words (`odyssey`). ---")
hl_words = ["quality", "odyssey"]

for w in hl_words:
    my_trials = [{"word": w, "scan_path": [demo_fixation]}] * 100

    demo_block = OneBlock(demo_reader, my_trials)
    res = demo_block.get_block_metrics()
    print(f"Fixate {fix_loc}: p({w}) = {res['avg_ptrue']:.2f}, postH = {res['avg_postH']:.2f}.")

### Time
# 2.4
print("--- 2.4: As time step increases, probability of true word increases. ---")
for tm in range(0, 16, 3):
    fixation_tmp = OneFixation(fix_loc = 1, fix_dur = tm)
    my_trials = [{"word": my_word, "scan_path": [fixation_tmp]}] * 100

    demo_block = OneBlock(demo_reader, my_trials)
    res = demo_block.get_block_metrics()
    print(f"Fixate {fix_loc} for {tm} time steps: p({my_word}) = {res['avg_ptrue']:.2f}, postH = {res['avg_postH']:.2f}.")

# =================================================================
# ================== Run specific simulations =====================
# =================================================================
print("--- Section 3: Run specific simulations ---")
print("Check demo_skipping.py & demo_refixation.py for more information.")
