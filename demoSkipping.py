import numpy as np
import os
import pandas as pd
import string
import EMRM
from EMRM import Vocabulary, OneVirtualReader, OneFixation, OneBlock

np.random.seed(0)

wlen_list = list(range(1, 9))
sigma_list = [1, 5]
lambda_list = [5, 10]
Ntrial = 50
fdur = 1

output_path = "Skipping_simu_output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
skip_csv_file = "data/example_Skipping_vocab.csv"
skip_csv_df = pd.read_csv(skip_csv_file)

def generate_output_filename(path, wlen, sigma_scale, Lamda_scale):
    s = "%swlen%d_S%s_L%s.txt" % (path, wlen, str(sigma_scale), str(Lamda_scale))
    return(s)

# human data
human_fix_file = "data/example_Skipping_human_fix.csv"
human_fix = pd.read_csv(human_fix_file)

for wlen in wlen_list:
    vocab_wlen= Vocabulary(characters = string.ascii_letters,
                           wlen = wlen,
                           input_df = skip_csv_df)
    
    human_fix_wlen = human_fix.loc[human_fix["wlen"] == wlen]
    launch_wlen = list(human_fix_wlen["launch"].unique())

    for sigma in sigma_list:
        for Lambda in lambda_list:
            tmp_reader = OneVirtualReader(vocabulary = vocab_wlen,
                                          sigma_scale = sigma,
                                          Lambda_scale = Lambda,
                                          fix_loc_list = launch_wlen)

            postH_list = []
            for _, row in human_fix_wlen.iterrows():
                fix = OneFixation(fix_loc = row["launch"], fix_dur = fdur)
                simu_info = [{"word": row["word"], "scan_path": [fix]}] * Ntrial
                block = OneBlock(tmp_reader, simu_info)
                res = block.get_block_metrics()
                postH = res["avg_postH"]
                postH_list.append(postH)
                
            # save simulation output
            hfw_copy = human_fix_wlen
            hfw_copy["postH"] = postH_list
            tmp_output_fn = generate_output_filename(output_path, wlen, sigma, Lambda)
            hfw_copy.to_csv(tmp_output_fn, index = False)
