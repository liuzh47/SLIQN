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
    for i in range(epoch):
        w = w - np.linalg.pinv(oracle.hes(w)) @ oracle.grad(w)
        gw = oracle.grad(w)
#         res.append(np.sqrt(gw.T @ np.linalg.pinv(oracle.hes(w)) @ gw)[0, 0])   
        res.append(np.linalg.norm(gw))  
        print(res[-1], oracle.f(w))
    return res, w

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
        w = init_w
        G = np.eye(d) * max_L
    else:
        w = init_w
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
                if epo == 0:
                    denom = step + 1
                else:
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
    

dataset = 'a6a' ## 'w8a', 'a6a', 'w6a'
X, Y = prepare_dataset(dataset)
n = X.shape[0]
batch_size = n // (size - 1)
n = batch_size * (size - 1)
X = X[:n, :]
Y = Y[:n, :]
reg = 0.01
reg = 3e-1
oracle = Logistic(X, Y, reg)
print(size-1)

d = oracle.d
w = np.random.randn(d, 1) / 10
res, w_opt = newton_sol(w, 20)

if rank > 0:
    oracle = Logistic(X[(rank - 1)*batch_size:batch_size+(rank-1)*batch_size,:], 
            Y[(rank - 1)*batch_size:batch_size+(rank - 1)*batch_size, :], reg)

init_w = np.random.randn(d, 1) / 10
max_L = 0.1
max_M = 0.03

iqn = iqn_sol(oracle, max_L, w_opt, init_w, epochs=500)
if rank == 0:
    print(iqn)

# passing MPI datatypes explicitly



# run the code with "mpiexec -n 32 python xxx.py"