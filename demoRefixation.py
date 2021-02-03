from EMRM import *
import random
import string

launch_list = range(-10,0)
wlen_list = range(3, 11)

launch_fix_dur = 2
lpos_fix_unit = 1
max_dur = 60
n_trials = 5

alpha = 0.9
beta = 0.7

vocab_fn = "./data/example_refix_vocab.csv"
output_fn = "./refixation_output.txt" 

output = ";".join(["wlen", "lpos", "launch", "word", "refix_act"]) + "\n"
for wlen in wlen_list:
    vocab_wlen = Vocabulary(string.ascii_lowercase,
                           WLEN = wlen,
                           csv_fn = vocab_fn)
    vocab_chr_mat = vocab_wlen.get_onehot_chr_rep()
    
    reader_wlen = OneVirtualReader(vocabulary = vocab_wlen,
                               SIGMA_SCALE = 1,
                               LAMBDA_SCALE = 1,
                               lpos_range = list(launch_list) + list(range(wlen+1))
                               )

    lpos_list = range(1, wlen + 1)
    for launch in launch_list:
        fixation0 = OneFixation(lpos = launch, fix_dur = launch_fix_dur)
        for lpos in lpos_list:
            for nt in range(n_trials):
                trl_word = random.choice(list(vocab_wlen.vocab_dict.keys()))
                trl = OneTrial(reader_wlen, word = trl_word)
                
                trl.update_posterior_one_fixation(fixation0)

                trl.fix_loc = lpos
                refix_target = trl.alpha_beta_policy(alpha, beta, max_dur)

                if refix_target == None:
                    refix_act = -999
                elif refix_target == wlen + 2:
                    refix_act = 999
                else:
                    refix_act = add_sac_err_swift_random(refix_target, lpos)

                output += ";".join([str(wlen), str(lpos), str(launch), trl_word, str(refix_act)]) + "\n"

with open(output_fn, "w") as wf:
    wf.write(output)
