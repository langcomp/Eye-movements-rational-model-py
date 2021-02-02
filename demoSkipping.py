from EMRM import *
import os
import string

wlen_list = list(range(1, 9))
sigma_list = [1, 5]
lambda_list = [5, 10]
Ntrial = 50
fdur = 1

skip_csv_fn = "./skip_data/example_skip_vocab.csv"
human_fix_fn = "./skip_data/example_skip_human_fix.csv"

output_path = "./skip_simu_output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

def generate_output_filename(path, wlen, sig_scale, lmd_scale):
    s = "%swlen%d_S%s_L%s.txt" % (path, wlen, str(sig_scale), str(lmd_scale))
    return(s)

human_fix = pd.read_csv(human_fix_fn)

for wlen in wlen_list:
    vocab_wlen= Vocabulary(string.ascii_letters,
                           WLEN = wlen,
                           csv_fn = skip_csv_fn)
    
    human_fix_wlen = human_fix.loc[human_fix["wlen"] == wlen]
    launch_wlen = list(human_fix_wlen["launch"].unique())

    for sg in sigma_list:
        for ld in lambda_list:
            tmp_reader = OneVirtualReader(vocabulary = vocab_wlen,
                                          SIGMA_SCALE = sg,
                                          LAMBDA_SCALE = ld,
                                          lpos_range = launch_wlen)

            postH_list = []
            for i, row in human_fix_wlen.iterrows():
                fix = OneFixation(lpos = row["launch"], fix_dur = fdur)
                simu_info = [{"word": row["word"], "scan_path": [fix]}] * Ntrial
                block = OneBlock(tmp_reader, simu_info)
                res = block.get_block_metrics()
                postH = res["avg_postH"]
                postH_list.append(postH)
                
            # save simulation output
            hfw_copy = human_fix_wlen
            hfw_copy["postH"] = postH_list
            tmp_output_fn = generate_output_filename(output_path, wlen, sg, ld)
            hfw_copy.to_csv(tmp_output_fn, index = False)
