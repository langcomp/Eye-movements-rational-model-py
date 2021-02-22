import numpy as np
import pandas as pd
import random
import string
import EMRM
from EMRM import Vocabulary, OneVirtualReader, OneFixation, OneTrial

np.random.seed(0)

launch_list = range(-10,0)
wlen_list = range(3, 11)

launch_fix_dur = 2
fix_loc_fix_unit = 1
max_dur = 60
n_trials = 5

alpha = 0.9
beta = 0.7

vocab_file = "data/example_Refixation_vocab.csv"
vocab_file_df = pd.read_csv(vocab_file)

output_file = "Refixation_output.txt" 
output = ";".join(["wlen", "fix_loc", "launch", "word", "refix_act"]) + "\n"

for wlen in wlen_list:
    vocab_wlen = Vocabulary(characters = string.ascii_lowercase,
                            wlen = wlen,
                            input_df = vocab_file_df)
    vocab_chr_mat = vocab_wlen.get_onehot_chr_rep()
    
    reader_wlen = OneVirtualReader(vocabulary = vocab_wlen,
                                   sigma_scale = 1,
                                   Lambda_scale = 1,
                                   fix_loc_list = list(launch_list) + list(range(wlen+1))
                                   )

    fix_loc_list = range(1, wlen + 1)
    for launch in launch_list:
        fixation0 = OneFixation(fix_loc = launch, fix_dur = launch_fix_dur)
        for fix_loc in fix_loc_list:
            for nt in range(n_trials):
                trl_word = random.choice(list(vocab_wlen.vocab_dict.keys()))
                trl = OneTrial(reader_wlen, word = trl_word)
                
                trl.update_posterior_one_fixation(fixation0)

                trl.fix_loc = fix_loc
                refix_target = trl.alpha_beta_policy(alpha, beta, max_dur)

                if refix_target == None:
                    refix_act = -999
                elif refix_target == wlen + 2:
                    refix_act = 999
                else:
                    refix_act = EMRM.add_sac_err_swift_random(refix_target, fix_loc)

                output += ";".join([str(wlen), str(fix_loc), str(launch), trl_word, str(refix_act)]) + "\n"

# save output
with open(output_file, "w") as wf:
    wf.write(output)
