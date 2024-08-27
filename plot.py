import numpy as np
import matplotlib.pyplot as plt
import pickle

import matplotlib
matplotlib.use("agg")

dataset = 'ijcnn1' ## 'w8a', 'a6a', 'a9a', 'mushrooms', 'ijcnn1', 'phishing', 'splice_scale', 'svmguide3', 'german.numer_scale', 'covtype', 'quadratic4'
file_name = 'pkl_files/'+ dataset + ".pkl"

with open(file_name, "rb") as f:
    res_list = pickle.load(f)
    
igs_filename = dataset + "_igs.pkl"
with open(igs_filename, "rb") as f:
    igs_res_list = pickle.load(f)
    
iqn = res_list[0]
igs = igs_res_list[0]
sliqn = res_list[1]
sliqn_sr1 = res_list[2]
sliqn_srk = res_list[3]

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(iqn[:125], '-', label='IQN', linewidth=2)
plt.plot(igs[:125],  '--', label='IGS', linewidth=2)
plt.plot(sliqn[:125], '-.', label='SLIQN', linewidth=2)
#plt.plot(iqn_sr1[:150], '--', label='iqn_sr1', linewidth=2)
plt.plot(sliqn_sr1[:125], ':', label='LISR1', linewidth=2)
plt.plot(sliqn_srk[:125], '--', label='LISR-k', linewidth=2)
#plt.plot(grsr1[:200], "-.", label="grsr1", linewidth=2)

ax.grid()
ax.legend()
ax.set_yscale('log')  
# plt.xscale('log') 
#plt.ylim(bottom=1e-12)
#plt.xlim(right=120)
ax.set_ylabel('Normalized Error')
ax.set_xlabel('No of effective passes')
#ax.set_title('General Function Minimization')
plt.tight_layout()
plt_name = "sliqn_"+ dataset + ".pdf"
plt.savefig(plt_name, format='pdf', bbox_inches='tight', dpi=300)