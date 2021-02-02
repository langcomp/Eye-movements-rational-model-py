from EMRM import *
import string

# =================================================================
# =========================== Quick start =========================
# =================================================================
### Get the probability of a word and entropy of posterior distribution
### after fixating the first letter of the word for 5 time steps.
print("--- Section 1: Quick start ---")
csv_fn7 = "./data/example_vocab.csv"

vocab_seven = Vocabulary(string.ascii_lowercase, WLEN = 7, csv_fn = csv_fn7)

demo_reader = OneVirtualReader(vocabulary = vocab_seven,
                               SIGMA_SCALE = 1,
                               LAMBDA_SCALE = 3,
                               lpos_range = [1,7]
                               )

my_word = "quality"
demo_trial = OneTrial(demo_reader, word = my_word)
demo_fixation = OneFixation(lpos = 1, fix_dur = 5)

demo_trial.update_posterior_one_fixation(demo_fixation)
postH = demo_trial.get_postH()
ptrue = demo_trial.get_prob_true_word()
print("Fixate `%s` at letter %s for %s steps: p(%s) = %f, postH = %f" % (my_word, str(demo_fixation.lpos), str(demo_fixation.fix_dur), my_word, ptrue, postH))

# =================================================================
# ===================== Check model behaviors =====================
# =================================================================
print("--- Section 2: Check model behaviors ---")

### Visual parameters
SIGMA_SCALE_list = [0.5, 2] # narrow, normal, wide visual acuity function
LAMBDA_SCALE_list = [1, 10] # poor, normal, good overall visual-input quality

# 2.1
print("--- 2.1: Influence of SIGMA_SCALE (visual acuity function)  ---")
for sg in SIGMA_SCALE_list:
    reader_tmp = OneVirtualReader(vocabulary = vocab_seven,
                                   SIGMA_SCALE = sg,
                                   LAMBDA_SCALE = 3,
                                   lpos_range = [1,7]
                                   )
    
    for lpos in reader_tmp.lpos_range:
        fixation_tmp = OneFixation(lpos = lpos, fix_dur = 5)
        my_trials = [{"word": my_word, "scan_path": [fixation_tmp]}] * 100

        demo_block = OneBlock(reader_tmp, my_trials)
        res = demo_block.get_block_metrics()
        print("Fixate %s, SIGMA_SCALE = %s: p(%s) = %f, postH = %f" % (str(lpos), str(sg), my_word, res["avg_ptrue"], res["avg_postH"]))


# 2.2
print("--- 2.2: Influence of LAMBDA_SCALE (overall visual input quality)  ---")
for ld in LAMBDA_SCALE_list:
    reader_tmp = OneVirtualReader(vocabulary = vocab_seven,
                                   SIGMA_SCALE = 1,
                                   LAMBDA_SCALE = ld,
                                   lpos_range = [1,7]
                                   )
    
    for lpos in reader_tmp.lpos_range:
        fixation_tmp = OneFixation(lpos = lpos, fix_dur = 5)
        my_trials = [{"word": my_word, "scan_path": [fixation_tmp]}] * 100

        demo_block = OneBlock(reader_tmp, my_trials)
        res = demo_block.get_block_metrics()
        print("Fixate %s, LAMBDA_SCALE = %s: p(%s) = %f, postH = %f" % (str(lpos), str(ld), my_word, res["avg_ptrue"], res["avg_postH"]))


### High freq vs. low freq words
# 2.3
print("--- 2.3: It's easier to identify high-freq words (`quality`) than low-freq words (`odyssey`). ---")
hl_words = ["odyssey", "quality"]

for w in hl_words:
    my_trials = [{"word": w, "scan_path": [demo_fixation]}] * 100

    demo_block = OneBlock(demo_reader, my_trials)
    res = demo_block.get_block_metrics()
    print("Fixate %s: p(%s) = %f, postH = %f" % (str(lpos), w, res["avg_ptrue"], res["avg_postH"]))

### Time
# 2.4
print("--- 2.4: As time step increases, probability of true word increases. ---")
for tm in range(0, 16, 3):
    fixation_tmp = OneFixation(lpos = 1, fix_dur = tm)
    my_trials = [{"word": my_word, "scan_path": [fixation_tmp]}] * 100

    demo_block = OneBlock(demo_reader, my_trials)
    res = demo_block.get_block_metrics()
    print("Fixate %s for %s time steps: p(%s) = %f, postH = %f" % (str(lpos), str(tm), my_word, res["avg_ptrue"], res["avg_postH"]))

# =================================================================
# ================== Run specific simulations =====================
# =================================================================
# TODO
#print("--- Section 3: Run specific simulations ---")
#print("Check demo_skipping.py & demo_refixation.py for more information.")
