import pickle
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("agg")

xi = 12
num_of_instances = 1000
d = 50

with open("quadratic" + str(xi) + "_" + str(num_of_instances) + "_" + str(d) +".pkl", "rb") as f:
    res_list = pickle.load(f)
    
with open("igs_quadratic" + str(xi) + "_" + str(num_of_instances) + "_" + str(d) +".pkl", "rb") as f:
    igs_res_list = pickle.load(f)
    
iqn = res_list[0]
igs = igs_res_list[0]
sliqn = res_list[1]
sliqn_sr1 = res_list[2]
sliqn_srk = res_list[3]

font = {'size'   : 14}

plt.rc('font', **font)

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(iqn[:60], '-', label='IQN', linewidth=2)
plt.plot(igs[:60], '--', label='IGS', linewidth=2)
plt.plot(sliqn[:60], '-.', label='SLIQN', linewidth=2)
#plt.plot(iqn_sr1[:400], '--', label='iqn_sr1', linewidth=2)
plt.plot(sliqn_sr1[:60], ':', label='LISR-1', linewidth=2)
plt.plot(sliqn_srk[:60], '--', label='LISR-$k$', linewidth=2)
#plt.plot(grsr1[:200], "-.", label="grsr1", linewidth=2)

ax.grid()
ax.legend()
ax.set_yscale('log')  
# plt.xscale('log') 
# plt.ylim(top=1e2) 
ax.set_ylabel('Normalized Error')
ax.set_xlabel('No. of effective passes')
#ax.set_title('Quadratic Function Minimization')
plt.tight_layout()
plt_name = "quadratic" + str(xi) + "_" + str(num_of_instances) + "_" + str(d) +".pdf"
plt.savefig(plt_name, format='pdf', bbox_inches='tight', dpi=300)
