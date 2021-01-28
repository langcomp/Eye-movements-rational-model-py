import EMRM
from EMRM import *
from importlib import reload

# =================================================================
# =========================== Quick start =========================
# =================================================================
### Get the probability of a word and entropy of posterior distribution
### after fixating the first letter of the word for 5 time steps.
print("--- Section 1: Quick start ---")
csv_fn = "./data/example_vocab.csv"
lpos_range = [1,7]

w = "quality"
time_steps = 1

words = csv2words(csv_fn)
c,B,A = get_visual_params(words, lpos_range)
D = get_vocabulary(words)
vocab = D["vocab"]
assert w in vocab, w + " not in vocabulary."

for lpos in lpos_range:
    new_x = run_single_word_identify(c, B, A, D, w, lpos, time = time_steps)
    ptrue = get_prob_true_word(new_x, vocab[w])
    postH = get_postH(new_x)
    print("Fixate `%s` at letter %s for %s steps: p(%s) = %f, postH = %f" % (w, str(lpos), str(time_steps), w, ptrue, postH))

# =================================================================
# ===================== Check model behaviors =====================
# =================================================================
print("--- Section 2: Check model behaviors ---")

### Visual parameters
default_global_var_dict = {"CHARACTERS": "abcdefghijklmnopqrstuvwxyz",
                           "WLEN": 7,
                           "SIGMA_SCALE": 1,
                           "LAMBDA_SCALE": 3}
    
SIGMA_SCALE_list = [0.5, 2] # narrow, normal, wide visual acuity function
LAMBDA_SCALE_list = [1, 10] # poor, normal, good overall visual-input quality

# 2.1
print("--- 2.1: Influence of SIGMA_SCALE (visual acuity function)  ---")
for sg in SIGMA_SCALE_list:
    global_var_dict = {"CHARACTERS": "abcdefghijklmnopqrstuvwxyz",
                        "WLEN": 7,
                        "SIGMA_SCALE": sg,
                        "LAMBDA_SCALE": 3}
    with open("./data/global_vars.json", "w", encoding = "utf-8") as wf:
        json.dump(global_var_dict, wf, ensure_ascii = False)

    # update visual parameters
    reload(EMRM)
    from EMRM import *

    # run simulation
    c,B,A = get_visual_params(words, lpos_range)
    for lpos in lpos_range:
        tmp_ptrue, tmp_postH = [],[]
        for trials in range(100):
            new_x = run_single_word_identify(c, B, A, D, w, lpos, time = 10)
            ptrue = get_prob_true_word(new_x, vocab[w])
            postH = get_postH(new_x)
            
            tmp_ptrue.append(ptrue)
            tmp_postH.append(postH)
        
        print("Fixate %s, SIGMA_SCALE = %s: p(%s) = %f, postH = %f" % (str(lpos), str(sg), w, np.mean(tmp_ptrue), np.mean(tmp_postH)))

# 2.2
print("--- 2.2: Influence of LAMBDA_SCALE (overall visual input quality)  ---")
for ld in LAMBDA_SCALE_list:
    global_var_dict = {"CHARACTERS": "abcdefghijklmnopqrstuvwxyz",
                        "WLEN": 7,
                        "SIGMA_SCALE": 1,
                        "LAMBDA_SCALE": ld}
    with open("./data/global_vars.json", "w", encoding = "utf-8") as wf:
        json.dump(global_var_dict, wf, ensure_ascii = False)

    # update visual parameters
    reload(EMRM)
    from EMRM import *

    # run simulation
    c,B,A = get_visual_params(words, lpos_range)
    for lpos in lpos_range:
        tmp_ptrue, tmp_postH = [],[]
        for trials in range(100):
            new_x = run_single_word_identify(c, B, A, D, w, lpos, time = 10)
            ptrue = get_prob_true_word(new_x, vocab[w])
            postH = get_postH(new_x)
            
            tmp_ptrue.append(ptrue)
            tmp_postH.append(postH)
        
        print("Fixate %s, LAMBDA_SCALE = %s: p(%s) = %f, postH = %f" % (str(lpos), str(ld), w, np.mean(tmp_ptrue), np.mean(tmp_postH)))

# reset global_vars.json
with open("./data/global_vars.json", "w", encoding = "utf-8") as wf:
    json.dump(default_global_var_dict, wf, ensure_ascii = False)
reload(EMRM)
from EMRM import *
c,B,A = get_visual_params(words, lpos_range)

### High freq vs. low freq words
# 2.3
print("--- 2.3: It's easier to identify high-freq words (`quality`) than low-freq words (`odyssey`). ---")
hl_words = ["odyssey", "quality"]
for lpos in list([lpos_range[0]]):
    for w in hl_words:
        tmp_ptrue, tmp_postH = [],[]
        for trials in range(100):
            new_x = run_single_word_identify(c, B, A, D, w, lpos, time = 10)
            ptrue = get_prob_true_word(new_x, vocab[w])
            postH = get_postH(new_x)
            
            tmp_ptrue.append(ptrue)
            tmp_postH.append(postH)
        print("Fixate %s: p(%s) = %f, postH = %f" % (str(lpos), w, np.mean(tmp_ptrue), np.mean(tmp_postH)))

### Time
# 2.4
print("--- 2.4: As time step increases, probability of true word increases. ---")
for lpos in list([lpos_range[0]]):
    for tm_step in range(0, 16, 3):
        tmp_ptrue, tmp_postH = [],[]
        for trials in range(100):
            new_x = run_single_word_identify(c, B, A, D, w, lpos, time = tm_step)
            ptrue = get_prob_true_word(new_x, vocab[w])
            postH = get_postH(new_x)
            
            tmp_ptrue.append(ptrue)
            tmp_postH.append(postH)
        print("Fixate %s for %s time steps: p(%s) = %f, postH = %f" % (str(lpos), str(tm_step), w, np.mean(tmp_ptrue), np.mean(tmp_postH)))

# =================================================================
# ================== Run specific simulations =====================
# =================================================================
print("--- Section 3: Run specific simulations ---")
print("Check demo_skipping.py & demo_refixation.py for more information.")


