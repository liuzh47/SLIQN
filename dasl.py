from mpi4py import MPI
import numpy as np
from scipy.io import loadmat
import sklearn.datasets
import matplotlib.pyplot as plt
import time
import matplotlib
import scipy.linalg
from scipy.sparse import csr_matrix, linalg

matplotlib.use('agg')
np.random.seed(22556)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

## Loistic Regression.
class Logistic:
    def __init__(self, X, Y, reg):
        self.X = X
        self.Y = Y
        self.μ = reg
        self.d = X.shape[1]
        self.N = X.shape[0]
        self.L = linalg.svds(X, k=1)[1] ** 2 / 4 + reg
        self.M = np.mean(np.linalg.norm(self.X, axis=1) ** 3) / self.μ ** (3/2)
        self.kappa = self.L / self.μ
        print("Logistic regression oracle created")
        print("\td = %d, L = %.2f; μ = %.2f; M = %.2f"%(self.d, self.L, self.μ, self.M))
        print("\tκ = {}".format(self.kappa))

    def f(self, w):
        pred = self.Y * (self.X @ w)
        pos = np.sum(np.log(1+np.exp(-pred[pred>=0])))
        neg = np.sum(np.log(1+np.exp(pred[pred<0]))-pred[pred<0])
        return (pos + neg) / self.N + 0.5 * self.μ * (w.T @ w)[0, 0]
    
    def grad(self, w):
        pred = self.Y * (self.X @ w)
        p = 0.5 * (1 + np.tanh(-0.5 * pred))
        return -self.X.T @ (self.Y * p) / self.N + self.μ * w
    
    def hes_vec(self, w, v):
        pred = self.Y * (self.X @ w)
        p = 0.5 * (1 + np.tanh(-0.5 * pred))
        return self.X.T @ (self.X @ v * p * (1-p)) / self.N + self.μ * v
    
    def hes(self, w):
        pred = self.Y * (self.X @ w)
        p = 0.5 * (1 + np.tanh(-0.5 * pred))
        return self.X.T @ (self.X * p * (1-p)) / self.N + self.μ * np.eye(self.d)
    
    def hes_diag(self, w):
        pred = self.Y * (self.X @ w)
        p = 0.5 * (1 + np.tanh(-0.5 * pred))
        return np.sum(self.X ** 2 * p * (1-p), axis=0) / self.N + self.μ * np.ones(self.d)

def prepare_dataset(dataset):
    X, Y = sklearn.datasets.load_svmlight_file('./data/libsvm/'+dataset+'.txt')
    X = np.array(X.todense())
    if len(Y.shape) == 1:
        Y = Y.reshape([-1, 1])
    if np.min(Y) != -1:
        Y = 2 * Y - 1
    return X, Y


def newton_sol(w, epoch):
    gw = oracle.grad(w)
    res = [np.linalg.norm(gw)]
    warmup_ws = []
    
    for i in range(epoch):
        w = w - np.linalg.pinv(oracle.hes(w)) @ oracle.grad(w)
        gw = oracle.grad(w)
#         res.append(np.sqrt(gw.T @ np.linalg.pinv(oracle.hes(w)) @ gw)[0, 0])   
        res.append(np.linalg.norm(gw))  
        print(res[-1], oracle.f(w))
        warmup_ws.append(w)
    return res, w, warmup_ws[1]

update_tag = 11
weight_tag = 15

def iqn_sol(oracle, max_L, 
            w_opt, init_w, epochs=200):
    res = []
    d = oracle.d
 
    if rank > 0:
        g_old = np.zeros_like(init_w)
        w_old = np.zeros_like(init_w)
        u_old = np.zeros_like(init_w)
        w = np.copy(init_w)
        G = np.eye(d) * max_L
    else:
        w = np.copy(init_w)
        invG = np.eye(d) / max_L
        u = np.zeros_like(init_w)
        g = np.zeros_like(init_w)
        
    for epo in range(epochs):
        if rank > 0:
            g = oracle.grad(w)
            s = w - w_old
            yy = g - g_old
            q = G @ s
            alpha = yy.T @ s
            beta = s.T @ G @ s
            G = G + yy @ yy.T / alpha - q @ q.T / beta
            u = G @ w
            u_diff = u - u_old
            w_old = w
            g_old = g
            u_old = u
            data = {"u_diff": u_diff, "yy":yy, "q":q, 
                    "alpha":alpha, "beta":beta, 
                    "client_id": rank}
            comm.send(data, dest=0, tag=update_tag)
            w = comm.recv(source=0, tag=weight_tag)
        else:
            for step in range(size - 1):
                data = comm.recv(source=MPI.ANY_SOURCE, tag=update_tag)
                u_diff = data["u_diff"]
                yy = data["yy"]
                q = data["q"]
                alpha = data["alpha"]
                beta = data["beta"]

                denom = size - 1
                u = u + u_diff / denom
                g = g + yy / denom
                
                v = invG @ yy
                U = invG - v @ v.T / (denom * alpha + v.T @ yy)
                z = U @ q
                invG = U + z @ z.T / (denom * beta - q.T @ z)
                w = invG @ (u - g)
                comm.send(w, dest=data["client_id"], tag=weight_tag)
        comm.bcast(0, root=0)
        if rank == 0:
            res.append(np.linalg.norm(w - w_opt))
    
    return res
    
def sliqn_sol(oracle, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    res = []
    d = oracle.d
    kappa = oracle.kappa
 
    if rank > 0:
        g_old = np.zeros_like(init_w)
        w_old = np.zeros_like(init_w)
        u_old = np.zeros_like(init_w)
        w = np.copy(init_w)
        G = np.eye(d) * max_L
    else:
        w = np.copy(init_w)
        invG = np.eye(d) / max_L
        u = np.zeros_like(init_w)
        g = np.zeros_like(init_w)
        
    for epo in range(epochs):
        gamma_k = max_M * np.sqrt(max_L) * np.linalg.norm(init_w - w_opt) * (1 - 1 /(d * kappa))** epo
        if rank > 0:
            g = oracle.grad(w)
            s = w - w_old
            yy = g - g_old
            scale_yy = (1 + gamma_k) * yy
            scale_G = (1 + gamma_k) ** 2 * G
            q = scale_G @ s
            alpha = scale_yy.T @ s
            beta = s.T @ scale_G @ s
            G_1 = scale_G + scale_yy @ scale_yy.T / alpha - q @ q.T / beta
            
            ind = np.argmax(np.diag(G_1) / oracle.hes_diag(w))
            gv = np.zeros([d, 1]) 
            gv[ind] = 1
            base_Hessian = oracle.hes(w)
            alpha_1 = gv.T @ base_Hessian @ gv
            beta_1 = gv.T @ G_1 @ gv
            y_1 = base_Hessian @ gv
            q_1 = G_1 @ gv
            G = G_1 + y_1 @ y_1.T / alpha_1 - q_1 @ q_1.T / beta_1
            
            u = G @ w
            u_diff = u - u_old
            w_old = w
            g_old = g
            u_old = u
            data = {"u_diff": u_diff, "yy":yy, "scale_yy":scale_yy, "q":q, 
                    "alpha":alpha, "beta":beta, "y_1":y_1, "q_1":q_1,
                    "alpha_1":alpha_1, "beta_1":beta_1, "client_id": rank}
            comm.send(data, dest=0, tag=update_tag)
            w = comm.recv(source=0, tag=weight_tag)
        else:
            invG = invG / (1 + gamma_k) ** 2
            for step in range(size - 1):
                data = comm.recv(source=MPI.ANY_SOURCE, tag=update_tag)
                u_diff = data["u_diff"]
                yy = data["yy"]
                scale_yy = data["scale_yy"]
                q = data["q"]
                alpha = data["alpha"]
                beta = data["beta"]
                y_1 = data["y_1"]
                q_1 = data["q_1"]
                alpha_1 = data["alpha_1"]
                beta_1 = data["beta_1"]

                denom = size - 1
                u = u + u_diff / denom
                g = g + yy / denom
                
                v = invG @ y_1
                U = invG - v @ v.T / (denom * alpha_1 + v.T @ y_1)
                z = U @ q_1
                U_1 = U + z @ z.T / (denom * beta_1 - q_1.T @ z)
                v_1 = U_1 @ scale_yy
                U_2 = U_1 - v_1 @ v_1.T / (denom * alpha + v_1.T @ scale_yy)
                z_1 = U_2 @ q
                invG = U_2 + z_1 @ z_1.T / (denom * beta - q.T @ z_1)
                
                w = invG @ (u - g)
                comm.send(w, dest=data["client_id"], tag=weight_tag)
        comm.bcast(0, root=0)
        if rank == 0:
            res.append(np.linalg.norm(w - w_opt))
    
    return res

def sliqn_sr1_sol(oracle, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    res = []
    d = oracle.d
    kappa = oracle.kappa
 
    if rank > 0:
        g_old = np.zeros_like(init_w)
        w_old = np.zeros_like(init_w)
        u_old = np.zeros_like(init_w)
        w = np.copy(init_w)
        G = np.eye(d) * max_L
    else:
        w = np.copy(init_w)
        invG = np.eye(d) / max_L
        u = np.zeros_like(init_w)
        g = np.zeros_like(init_w)
        
    for epo in range(epochs):
        gamma_k = max_M * np.sqrt(max_L) * np.linalg.norm(init_w - w_opt) * (1 - 1 /d)** epo
        if rank > 0:
            g = oracle.grad(w)
            s = w - w_old
            yy = g - g_old
            scale_yy = (1 + gamma_k) * yy
            scale_G = (1 + gamma_k) ** 2 * G
            q = scale_G @ s - scale_yy
            alpha = q.T @ s
            G_1 = scale_G - q @ q.T / (alpha + 1e-30)
            ind = np.argmax(np.diag(G_1) - oracle.hes_diag(w))
            gv = np.zeros([d, 1]) 
            gv[ind] = 1
            base_Hessian = oracle.hes(w)
            
            q_1 = (G_1 - base_Hessian) @ gv
            alpha_1 = q_1.T @ gv
            G = G_1  - q_1 @ q_1.T / (alpha_1 + 1e-30)
            
            u = G @ w
            u_diff = u - u_old
            w_old = w
            g_old = g
            u_old = u
            data = {"u_diff": u_diff, "yy":yy,  "q":q, 
                    "alpha":alpha, "q_1":q_1,
                    "alpha_1":alpha_1, "client_id": rank}
            comm.send(data, dest=0, tag=update_tag)
            w = comm.recv(source=0, tag=weight_tag)
        else:
            invG = invG / (1 + gamma_k) ** 2
            for step in range(size - 1):
                data = comm.recv(source=MPI.ANY_SOURCE, tag=update_tag)
                u_diff = data["u_diff"]
                yy = data["yy"]
                q = data["q"]
                alpha = data["alpha"]
                q_1 = data["q_1"]
                alpha_1 = data["alpha_1"]

                denom = size - 1
                u = u + u_diff / denom
                g = g + yy / denom
                
                v = invG @ q_1
                U = invG + v @ v.T / (denom * alpha_1 - v.T @ q_1)
                z = U @ q
                invG = U + z @ z.T / (denom * alpha - z.T @ q)
                
                w = invG @ (u - g)
                comm.send(w, dest=data["client_id"], tag=weight_tag)
        comm.bcast(0, root=0)
        if rank == 0:
            res.append(np.linalg.norm(w - w_opt))
    
    return res


dataset = 'w8a' ## 'w8a', 'a9a', 'w6a', 'mushrooms', 'ijcnn1'
X, Y = prepare_dataset(dataset)
n = X.shape[0]
batch_size = n // (size - 1)
n = batch_size * (size - 1)
X = X[:n, :]
Y = Y[:n, :]
reg = 0.01
reg = 4e-1
oracle = Logistic(X, Y, reg)
print(size-1)

d = oracle.d
w = np.random.randn(d, 1) / 10
res, w_opt,  warmup_w = newton_sol(w, 20)

if rank > 0:
    oracle = Logistic(X[(rank - 1)*batch_size:batch_size+(rank-1)*batch_size,:], 
            Y[(rank - 1)*batch_size:batch_size+(rank - 1)*batch_size, :], reg)

#init_w = np.random.randn(d, 1) / 10
init_w = warmup_w
max_L = 0.1
max_M = 0.03

iqn = iqn_sol(oracle, max_L, w_opt, init_w, epochs=500)

max_L = 0.1
max_M = 1e-8
sliqn = sliqn_sol(oracle, max_L, max_M, w_opt, init_w, corr=False, epochs=500)
    
max_L = 0.1
max_M = 1e-8
sliqn_sr1 = sliqn_sr1_sol(oracle, max_L, max_M, w_opt, init_w, corr=False, epochs=500)
if rank == 0:  
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plt.plot(iqn[:500], '-', label='iqn', linewidth=2)
    plt.plot(sliqn[:500], '-.', label='sliqn', linewidth=2)
    plt.plot(sliqn_sr1[:500], '--', label='sliqn_sr1', linewidth=2)
    ax.grid()
    ax.legend()
    ax.set_yscale('log')  
    # plt.xscale('log')  
    ax.set_ylabel('$\lambda_f(x_k)$')
    ax.set_xlabel('Epochs, $n=500, \kappa=%d$'%(oracle.kappa))
    ax.set_title('General Function Minimization')
    plt.tight_layout()  
    plt_name = 'dist-qn_' + dataset + ".pdf"
    plt.savefig(plt_name, format='pdf', bbox_inches='tight', dpi=300)
# passing MPI datatypes explicitly



# run the code with "mpiexec -n 32 python xxx.py"