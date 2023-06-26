#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.special
import copy
import sklearn.datasets
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, linalg
import time

import matplotlib
matplotlib.use("agg")
# In[2]:


SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 25

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
plt.rcParams['lines.linewidth'] = 2
# plt.rcParams['figure.figsize'] = (9, 5)


# In[3]:


d = 100
kappa = 2000
A = np.random.randn(d-1, d) * 10
A = A.T @ A
mu = np.linalg.eigh(A)[0][-1] / (kappa-0.99)
A += np.eye(d) * mu
invA = np.linalg.inv(A)
eigens = np.linalg.eigh(A)[0]
L, mu = eigens[-1], eigens[0]
kappa = L / mu
print('L', L, 'u', mu, 'kappa:', kappa)


# In[4]:


def gr1sr1(G, A):
    d, _ = A.shape
    ind = np.argmax(np.diag(G) / np.diag(A))
    u = np.zeros([d, 1])
    u[ind] = 1
    return u

def gr2sr1(G, A):
    d, _ = A.shape
    ind = np.argmax(np.diag(G-A))
    u = np.zeros([d, 1])
    u[ind] = 1
    return u

def matrix_app_sr1(A, method='greedy', ind='sigma'):
    d,_ = A.shape
    G = L * np.eye(d)
    if ind == 'tau':
        res = [np.trace(G-A)]
    if ind == 'sigma':
        res = [np.trace(G@invA)-d]
    for i in range(d-2):
        if method == 'random':
            u = np.random.randn(d, 1) 
        if method == 'greedy-v2':    
            u = gr2sr1(G, A)
        if method == 'greedy-v1':    
            u = gr1sr1(G, A)    
        G = G - (G - A) @ u @ u.T @ (G - A) / (u.T @ (G - A) @ u)
        if ind == 'tau':
            res.append(np.trace(G - A))
        if ind == 'sigma':    
            res.append(np.trace(G@invA)-d)
    return res

ind = 'sigma'
greedy_res2 = matrix_app_sr1(A, 'greedy-v2', ind=ind)
rand_res = matrix_app_sr1(A, 'random', ind=ind)
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(greedy_res2, '--', label='GrSR1v2', linewidth=2) 
ax.plot(rand_res, '-.', label='RaSR1', linewidth=2) 
ax.plot([greedy_res2[0]*(1-i/d) for i in range(d-1)], '-', label='Theory', linewidth=2)
ax.legend()
ax.set_ylabel(r'$tr(G_k-A)$')
ax.set_xlabel('Epochs, $n=100, \kappa=%d$'%(kappa))
ax.set_title('Matrix Approximation')
ax.set_yscale('log')
ax.grid()
plt.tight_layout()  
# plt.savefig('mat-app-sr1.pdf', format='pdf', bbox_inches='tight', dpi=300)


# In[5]:


def grbfgs(sqrG, invA):
    d, _ = A.shape
    ind = np.argmax(np.diag(sqrG @ invA @ sqrG))
    u = np.zeros([d, 1])
    u[ind] = 1
    return u

def matrix_app_bfgs(A, g, epochs=20*d, method='greedy'):
    d,_ = A.shape
    G = np.eye(d) * L
    sqr_invG = np.linalg.cholesky(np.linalg.pinv(G)).T
    
    invA = np.linalg.inv(A)
    res = [np.trace(G@invA)-d]
               
    for i in range(epochs):
        if method == 'random':
            u = np.random.randn(d, 1) 
        if method == 'greedy':    
            u = grbfgs(np.linalg.pinv(sqr_invG), invA)
        if method == 'random_v2':
            tu = np.random.randn(d, 1)
            u = sqr_invG.T @ tu    
            v = u / np.sqrt(u.T @ A @ u)
            _, tmp_r = scipy.linalg.qr_update(np.eye(d), sqr_invG, -sqr_invG @ (A @ v), v)
            _, sqr_invG = scipy.linalg.qr_insert(np.eye(d), tmp_r, v.T, 0)
            sqr_invG = sqr_invG[:-1, :]
            
        Au = A @ u
        Gu = G @ u        
        G = G - (Gu @ Gu.T) / (u.T @ Gu) + (Au @ Au.T) / (u.T @ Au) 
#         print(np.linalg.norm(sqr_invG.T @ G @ sqr_invG-np.eye(d)))
        res.append(np.trace(G@invA)-d)
    return res

epcohs = 20*d
g = np.random.randn(d, d)
rand_res1 = matrix_app_bfgs(A, g, epcohs, method='random')
rand_res2 = matrix_app_bfgs(A, g, epcohs, method='random_v2')

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.plot(rand_res1, '--', label='RaBFGSv1', linewidth=2)
ax.plot(rand_res2, '-.', label='RaBFGSv2', linewidth=2)
# plt.plot(greedy_res, label='Greedy')
ax.plot([rand_res1[0]*(1-1/d)**i for i in range(epcohs+1)], label='Theory', linewidth=2)
ax.legend()
ax.set_yscale('log')
ax.set_ylim([1e-6, 1e5])
ax.set_ylabel('$tr(G_kA^{-1})-n$')
ax.set_xlabel('Epochs, $n=100, \kappa=%d$'%(kappa))
ax.set_title('Matrix Approximation')
ax.grid()
plt.tight_layout()  
# plt.savefig('mat-app-bfgs-kappa-%d.pdf'%kappa, format='pdf', bbox_inches='tight', dpi=300)


# In[6]:


b = np.random.randn(d, 1)
x = np.random.randn(d, 1)
init_x = copy.deepcopy(x)

## GrSR1v1
G = np.eye(d) * L
invG = np.linalg.pinv(G)
x = init_x
gx = A @ x - b
grsr1v1 = [np.sqrt(gx.T @ invA @ gx)[0, 0]]
for i in range(d+1):
    x = x - invG @ gx
    
    ind = np.argmax(np.diag(G) / np.diag(A))
    u = np.zeros([d, 1])
    u[ind] = 1
    
    Au = A @ u
    Gu = G @ u  
    G = G - (Gu - Au) @ (Gu - Au).T / (u.T @ (Gu - Au) + 1e-30)
    
    v = invG @ Au
    invG = invG + (u - v) @ (u - v).T / (u.T @ A @ (u - v) + 1e-30) 
    gx = A @ x - b
    grsr1v1.append(np.sqrt(gx.T @ invA @ gx)[0, 0])
    
    
## GrSR1v2
G = np.eye(d) * L
invG = np.linalg.pinv(G)
x = init_x
gx = A @ x - b
grsr1v2 = [np.sqrt(gx.T @ invA @ gx)[0, 0]]
for i in range(d+1):
    x = x - invG @ gx
    
    ind = np.argmax(np.diag(G-A))
    u = np.zeros([d, 1])
    u[ind] = 1
    
    Au = A @ u
    Gu = G @ u  
    G = G - (Gu - Au) @ (Gu - Au).T / (u.T @ (Gu - Au) + 1e-30)    
    
    v = invG @ Au
    invG = invG + (u - v) @ (u - v).T / (u.T @ A @ (u - v) + 1e-30) 
    gx = A @ x - b
    grsr1v2.append(np.sqrt(gx.T @ invA @ gx)[0, 0])


## RaSR1
G = np.eye(d) * L
invG = np.linalg.pinv(G)
x = init_x
gx = A @ x - b
rasr1 = [np.sqrt(gx.T @ invA @ gx)[0, 0]]
for i in range(d+1):
    u = np.random.randn(d, 1)
    x = x - invG @ gx
    v = invG @ (A @ u)
    invG = invG + (u - v) @ (u - v).T / (u.T @ A @ (u - v)+1e-30) 
    gx = A @ x - b
    rasr1.append(np.sqrt(gx.T @ invA @ gx)[0, 0])


## GrBFGSv1
G = np.eye(d) * L
sqr_invG = np.linalg.cholesky(np.linalg.pinv(G)).T
x = init_x
gx = A @ x - b
grbfgsv1 = [np.sqrt(gx.T @ invA @ gx)[0, 0]]
for i in range(15*d):
    x = x - sqr_invG.T @ sqr_invG @ gx
    
    ind = np.argmax(np.diag(G) / np.diag(A))
    u = np.zeros([d, 1])
    u[ind] = 1 
    
    Au = A @ u
    Gu = G @ u
    G = G - (Gu @ Gu.T) / (u.T @ Gu) + (Au @ Au.T) / (u.T @ Au) 

    v = u / np.sqrt(u.T @ Au)
    _, tmp_r = scipy.linalg.qr_update(np.eye(d), sqr_invG, -sqr_invG @ (A @ v), v)
    _, sqr_invG = scipy.linalg.qr_insert(np.eye(d), tmp_r, v.T, 0)
    sqr_invG = sqr_invG[:-1, :]
    gx = A @ x - b
    grbfgsv1.append(np.sqrt(gx.T @ invA @ gx)[0, 0])
#     print(np.linalg.norm(sqr_invG @ G @ sqr_invG.T-np.eye(d)))    
    

## RaBFGSv1
G = np.eye(d) * L
sqr_invG = np.linalg.cholesky(np.linalg.pinv(G)).T
x = init_x
gx = A @ x - b
rabfgsv1 = [np.sqrt(gx.T @ invA @ gx)[0, 0]]
for i in range(15*d):
    x = x - sqr_invG.T @ sqr_invG @ gx
    u = np.random.randn(d, 1)  
    v = u / np.sqrt(u.T @ A @ u)
    _, tmp_r = scipy.linalg.qr_update(np.eye(d), sqr_invG, -sqr_invG @ (A @ v), v)
    _, sqr_invG = scipy.linalg.qr_insert(np.eye(d), tmp_r, v.T, 0)
    sqr_invG = sqr_invG[:-1, :]
    gx = A @ x - b
    rabfgsv1.append(np.sqrt(gx.T @ invA @ gx)[0, 0])


## RaBFGSv2
G = np.eye(d) * L
sqr_invG = np.linalg.cholesky(np.linalg.pinv(G)).T
x = init_x
gx = A @ x - b
rabfgsv2 = [np.sqrt(gx.T @ invA @ gx)[0, 0]]
for i in range(15*d):
    x = x - sqr_invG.T @ sqr_invG @ gx
    tu = np.random.randn(d, 1)
    u = sqr_invG.T @ tu    
    v = u / np.sqrt(u.T @ A @ u)
    _, tmp_r = scipy.linalg.qr_update(np.eye(d), sqr_invG, -sqr_invG @ (A @ v), v)
    _, sqr_invG = scipy.linalg.qr_insert(np.eye(d), tmp_r, v.T, 0)
    sqr_invG = sqr_invG[:-1, :]
    gx = A @ x - b
    rabfgsv2.append(np.sqrt(gx.T @ invA @ gx)[0, 0])


# In[7]:


fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# plt.plot(grsr1v1, '--', label='GrSR1v1')
# plt.plot(grsr1v2, '-', label='GrSR1v2')
plt.plot(rasr1, '-', label='RaSR1', linewidth=2)
plt.plot(grbfgsv1, '-.', label='GrBFGSv1', linewidth=2)
plt.plot(rabfgsv1, '--', label='RaBFGSv1', linewidth=2)
plt.plot(rabfgsv2, ':', label='RaBFGSv2', linewidth=2)
ax.grid()
ax.legend()
ax.set_yscale('log')  
# plt.xscale('log')  
ax.set_ylabel('$\lambda_f(x_k)$')
ax.set_xlabel('Epochs, $n=100, \kappa=%d$'%(kappa))
ax.set_title('Quadratic Minimization')
plt.tight_layout()  
# plt.savefig('quad-sr1-bfgs.pdf', format='pdf', bbox_inches='tight', dpi=300)


# In[8]:


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


# In[9]:


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

def rasr1_sol(w, G, epochs, corr=False):
    invG = np.linalg.pinv(G)
    gw = oracle.grad(w)
    res = [np.linalg.norm(gw)]
    for i in range(epochs):
        dw = - invG @ gw
        if corr:
            r = np.sqrt(dw.T @ oracle.hes_vec(w, dw))
            invG = invG / (1 + oracle.M * r) # proposed by authors
        w = w + dw
        u = np.random.randn(d, 1)
        v = invG @ oracle.hes_vec(w, u)
        invG = invG + (u - v) @ (u - v).T / (u.T @ oracle.hes_vec(w, u-v) + 1e-30) 
        gw = oracle.grad(w)
#         res.append(np.sqrt(gw.T @ np.linalg.pinv(oracle.hes(w)) @ gw)[0, 0])
        res.append(np.linalg.norm(gw))
    print(res[-1])    
    return res

def grsr1_sol(w, G, epochs, ours=True, corr=False):
    gw = oracle.grad(w)
    res = [np.linalg.norm(gw)]
    invG = np.linalg.pinv(G)
    for i in range(epochs):
        dw = - invG @ gw
        if corr:
            r = np.sqrt(dw.T @ oracle.hes_vec(w, dw))
            G = G * (1 + oracle.M * r)
            invG = invG / (1 + oracle.M * r)
        w = w + dw    
        if ours:  
            ind = np.argmax(np.diag(G) - oracle.hes_diag(w))
        else:    
            ind = np.argmax(np.diag(G) / oracle.hes_diag(w))
        u = np.zeros([d, 1])
        u[ind] = 1
    
        Gu = G @ u
        Au = oracle.hes_vec(w, u)
        G = G - (Gu - Au) @ (Gu - Au).T / (u.T @ (Gu - Au) + 1e-30)
        
        v = invG @ Au
        invG = invG + (u - v) @ (u - v).T / (u.T @ oracle.hes_vec(w, u-v) + 1e-30) 
        gw = oracle.grad(w)
        res.append(np.linalg.norm(gw))
    print(res[-1])
    return res


def iqn_sol(oracles, max_L, 
            w_opt, init_w, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(X.shape[1]) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [np.linalg.norm(ws[-1] - w_opt)]
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    invG = np.eye(d) / max_L
    for _ in range(epochs):
        for i in range(N):
            w = invG @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            
            stoc_Hessian = Gs[i] + yy@yy.T / (yy.T@s) - \
                           (Gs[i]@ s)@(s.T@Gs[i]) / (s.T @ Gs[i] @s)
            B = B + (stoc_Hessian - Gs[i]) / N
            u = u + (stoc_Hessian @ w - Gs[i] @ ws[i]) / N
            g = g + yy / N
            
            U = invG - (invG@ yy) @ (yy.T@ invG) / (N * yy.T@s + yy.T@ invG @ yy)
            invG = U + (U@(Gs[i]@s))@((s.T@Gs[i])@U) / (N* s.T@Gs[i]@s - (s.T@Gs[i])@U@(Gs[i]@s))
            
            Gs[i] = np.copy(stoc_Hessian)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt))
        
    return res
    
    
def iqs_sol(oracles, max_L, max_M,
            w_opt, init_w, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(X.shape[1]) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [np.linalg.norm(ws[-1] - w_opt)]
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    for _ in range(epochs):
        for i in range(N):
            w = np.linalg.pinv(B) @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            r = np.sqrt(s.T @ oracles[i].hes_vec(ws[i], s))
            scale = 1 + max_M * r
            scale = 1
            scale_Hessian = scale * Gs[i]
            ind = np.argmax(np.diag(scale_Hessian) / oracles[i].hes_diag(w))
            gv = np.zeros([d, 1])
            gv[ind] = 1
            base_Hessian = oracles[i].hes(w)
            
            stoc_Hessian = scale_Hessian + (base_Hessian@ gv)@(gv.T@base_Hessian) / (gv.T @ base_Hessian @gv) - \
                           (scale_Hessian@ gv)@(gv.T@scale_Hessian) / (gv.T @ scale_Hessian @gv)
            B = B + (stoc_Hessian - Gs[i]) / N
            u = u + (stoc_Hessian @ w - Gs[i] @ ws[i]) / N
            g = g + yy / N
                        
            Gs[i] = np.copy(stoc_Hessian)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt))
        
    return res
# In[10]:

def sliqn_sol(oracles, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(X.shape[1]) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [np.linalg.norm(ws[-1] - w_opt)]
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    for _ in range(epochs):
        for i in range(N):
            w = np.linalg.pinv(B) @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            r = np.sqrt(s.T @ oracles[i].hes_vec(ws[i], s))
            
            if corr:
                scale = 1. + max_M * r / 2.
            else:   
                scale = 1.
                
            scale_Hessian = scale * scale * Gs[i]
            scale_yy = scale * yy
            
            stoc_Hessian = scale_Hessian + scale_yy@scale_yy.T / (scale_yy.T@s) - \
                           (scale_Hessian @ s)@(s.T @ scale_Hessian) / (s.T @ Gs[i] @s)
            ind = np.argmax(np.diag(stoc_Hessian) / oracles[i].hes_diag(w))
            gv = np.zeros([d, 1])
            gv[ind] = 1
            base_Hessian = oracles[i].hes(w)
            
            stoc_Hessian = stoc_Hessian + (base_Hessian@ gv)@(gv.T@base_Hessian) / (gv.T @ base_Hessian @gv) - \
                           (stoc_Hessian@ gv)@(gv.T@stoc_Hessian) / (gv.T @ stoc_Hessian @gv)
            B = B + (stoc_Hessian - Gs[i]) / N
            u = u + (stoc_Hessian @ w - Gs[i] @ ws[i]) / N
            g = g + yy / N
                        
            Gs[i] = np.copy(stoc_Hessian)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt))
        
    return res

def iqn_sr1_sol(oracles, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(X.shape[1]) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [np.linalg.norm(ws[-1] - w_opt)]
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    for _ in range(epochs):
        for i in range(N):
            w = np.linalg.pinv(B) @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            r = np.sqrt(s.T @ oracles[i].hes_vec(ws[i], s))
            
            scale = 1.
                
            scale_Hessian = scale * scale * Gs[i]
            scale_yy = scale * yy
            
            vec_diff = scale_Hessian @ s - scale_yy
            stoc_Hessian = scale_Hessian - vec_diff @ vec_diff.T / (vec_diff.T @ s)
            
            B = B + (stoc_Hessian - Gs[i]) / N
            u = u + (stoc_Hessian @ w - Gs[i] @ ws[i]) / N
            g = g + yy / N
                        
            Gs[i] = np.copy(stoc_Hessian)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt))
        
    return res

def sliqn_sr1_sol(oracles, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(X.shape[1]) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [np.linalg.norm(ws[-1] - w_opt)]
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    for _ in range(epochs):
        for i in range(N):
            w = np.linalg.pinv(B) @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            r = np.sqrt(s.T @ oracles[i].hes_vec(ws[i], s))
            
            if corr:
                scale = 1. + max_M * r / 2.
            else:   
                scale = 1.
                
            scale_Hessian = scale * scale * Gs[i]
            scale_yy = scale * yy
            
            vec_diff = scale_Hessian @ s - scale_yy
            stoc_Hessian = scale_Hessian - vec_diff @ vec_diff.T / (vec_diff.T @ s)
            base_Hessian = oracles[i].hes(w)
            ind = np.argmax(np.diag(stoc_Hessian) - np.diag(base_Hessian))
            gv = np.zeros([d, 1])
            gv[ind] = 1
            
            Hessian_diff = stoc_Hessian - base_Hessian
            stoc_Hessian_2 = stoc_Hessian - Hessian_diff @ gv @ gv.T @ Hessian_diff / (gv.T @ Hessian_diff @ gv)
            B = B + (stoc_Hessian_2 - Gs[i]) / N
            u = u + (stoc_Hessian_2 @ w - Gs[i] @ ws[i]) / N
            g = g + yy / N
                        
            Gs[i] = np.copy(stoc_Hessian_2)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt))
        
    return res

def prepare_dataset(dataset):
    X, Y = sklearn.datasets.load_svmlight_file('./data/libsvm/'+dataset+'.txt')
    X = np.array(X.todense())
    if len(Y.shape) == 1:
        Y = Y.reshape([-1, 1])
    if np.min(Y) != -1:
        Y = 2 * Y - 1
    return X, Y
dataset = 'a6a' ## 'w8a', 'a6a', 'w6a'
X, Y = prepare_dataset(dataset)
reg = 0.01
reg = 1e-3
oracle = Logistic(X, Y, reg)
print(X.shape, Y.shape)


# In[11]:
batch_size = 1000
num_of_batches = int(X.shape[0] / batch_size)
data_size = batch_size * num_of_batches
X = X[:data_size, :]
Y = Y[:data_size]
oracle = Logistic(X, Y, reg)

d = oracle.d
G = np.eye(d) * oracle.L
w = np.random.randn(d, 1) / 10
res, w_opt = newton_sol(w, 20)


oracles = []

for i in range(int(X.shape[0] / batch_size)):
    oracles.append(Logistic(X[i*batch_size:batch_size+i*batch_size,:], 
        Y[i*batch_size:batch_size+i*batch_size, :], reg))

print(len(oracles))

Ls = [o.L for o in oracles]
Ms = [o.M for o in oracles]
max_L = 0.05
max_M = 0.03

init_w = np.random.randn(d, 1) / 10

iqn = iqn_sol(oracles, max_L, w_opt, init_w, epochs=500)
iqs = iqs_sol(oracles, max_L, max_M, w_opt, init_w, epochs=200)
sliqn = sliqn_sol(oracles, max_L, max_M, w_opt, init_w, corr=True, epochs=500)
iqn_sr1 = iqn_sr1_sol(oracles, max_L, max_M, w_opt, init_w, corr=True, epochs=1000)
max_M = 0.5
sliqn_sr1 = sliqn_sr1_sol(oracles, max_L, max_M, w_opt, init_w, corr=True, epochs=1000)
print(sliqn[300:500])
print(iqn_sr1[300:500])
print(sliqn_sr1[300:500])

