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
import heapq

import matplotlib
matplotlib.use("agg")
np.random.seed(22556)
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


# In[8]:


## Loistic Regression.
class Quadratic:
    def __init__(self, d, xi):
        self.d = d
        self.xi = xi
        t1 = np.random.uniform(1, 10**(xi/2), int(d/2))
        t2 = np.random.uniform(10**(-xi/2), 1, int(d/2))
        self.diag_elems = np.concatenate((t1, t2))
        self.A = np.diag(self.diag_elems)
        self.invA = np.diag(1 / self.diag_elems)
        self.b = np.random.uniform(0, 1000, d)
        self.b = np.expand_dims(self.b, axis=1)
        self.L = np.max(self.diag_elems)
        self.μ = np.min(self.diag_elems)
        self.kappa = self.L / self.μ
        print("Quadratic oracle created")
        print("\td = %d, L = %.2f; μ = %.2f;"%(self.d, self.L, self.μ))
        print("\tκ = {}".format(self.kappa))

    def f(self, w):
        return w.T @ self.A @ w / 2 + w.T @ self.b
    
    def grad(self, w):
        return self.A @ w + self.b
    
    def hes_vec(self, w, v):
        return self.A @ v
        
    def hes(self, w):
        return self.A
        
    def hesU(self, w, U):
        return self.A @ U
        
    def hes_diag(self, w):
        return self.diag_elems

# In[9]:

def grad_sol(w, epoch, lr, A, b):
    res = []
    ws = []
    for i in range(epoch):
        lr = 1 / (i + 500000)
        cur_grad = A @ w + b
        w = w - lr * cur_grad
        res.append(np.linalg.norm(cur_grad))
        ws.append(w)
    return res, ws

def newton_sol(oracle, w, epoch):
    warmup_ws = []
    gw = oracle.grad(w)
    res = [np.linalg.norm(gw)]
    for i in range(epoch):
        w = w - np.linalg.pinv(oracle.hes(w)) @ oracle.grad(w)
        gw = oracle.grad(w)
#         res.append(np.sqrt(gw.T @ np.linalg.pinv(oracle.hes(w)) @ gw)[0, 0])   
        res.append(np.linalg.norm(gw))  
        warmup_ws.append(w)
    return res, w, warmup_ws[1]
    
def BFGS_sol(oracle, L, init_x, epochs=2000):
    d = oracle.d
    G = np.eye(d) * L
    sqr_invG = np.linalg.cholesky(np.linalg.pinv(G)).T
    x = init_x
    gx = oracle.grad(x)
    grbfgsv1 = [np.sqrt(gx.T @ oracle.invA @ gx)[0, 0]]
    for i in range(epochs):
        x = x - sqr_invG.T @ sqr_invG @ gx
    
        ind = np.argmax(np.diag(G) / oracle.hes_diag(x))
        u = np.zeros([d, 1])
        u[ind] = 1 
    
        Au = oracle.hesU(x, u)
        Gu = G @ u
        G = G - (Gu @ Gu.T) / (u.T @ Gu) + (Au @ Au.T) / (u.T @ Au) 

        v = u / np.sqrt(u.T @ Au)
        _, tmp_r = scipy.linalg.qr_update(np.eye(d), sqr_invG, -sqr_invG @ (oracle.hesU(x, v)), v)
        _, sqr_invG = scipy.linalg.qr_insert(np.eye(d), tmp_r, v.T, 0)
        sqr_invG = sqr_invG[:-1, :]
        gx = oracle.grad(x)
        grbfgsv1.append(np.sqrt(gx.T @ oracle.invA @ gx)[0, 0])
    return grbfgsv1

def iqn_sol(oracles, max_L, 
            w_opt, init_w, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    ts = []
    d = oracles[0].d
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(d) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [1]
    init_norm = np.linalg.norm(init_w - w_opt)
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    invG = np.eye(d) / max_L
    init_time = time.time()
    for _ in range(epochs):
        for i in range(N):
            w = invG @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            
            stoc_Hessian = Gs[i] + yy@yy.T / (yy.T@s) - \
                           (Gs[i]@ s)@(s.T@Gs[i]) / (s.T @ Gs[i] @s + 1e-30)
            B = B + (stoc_Hessian - Gs[i]) / N
            u = u + (stoc_Hessian @ w - Gs[i] @ ws[i]) / N
            g = g + yy / N
            
            U = invG - (invG@ yy) @ (yy.T@ invG) / (N * yy.T@s + yy.T@ invG @ yy + 1e-30)
            invG = U + (U@(Gs[i]@s))@((s.T@Gs[i])@U) / (N* s.T@Gs[i]@s - (s.T@Gs[i])@U@(Gs[i]@s) + 1e-30)
            
            Gs[i] = np.copy(stoc_Hessian)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt) / init_norm)
        ts.append(time.time() - init_time)
        
    return res, ts
    
    
def iqs_sol(oracles, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    d = oracles[0].d
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(d) * max_L)
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
                scale = 1 + max_M * r
            else:
                scale = 1
            scale_Hessian = scale * Gs[i]
            ind = np.argmax(np.diag(scale_Hessian) / oracles[i].hes_diag(w))
            gv = np.zeros([d, 1])
            gv[ind] = 1
            base_Hessian = oracles[i].hes(w)
            
            stoc_Hessian = scale_Hessian + (base_Hessian@ gv)@(gv.T@base_Hessian) / (gv.T @ base_Hessian @gv + 1e-30) - \
                           (scale_Hessian@ gv)@(gv.T@scale_Hessian) / (gv.T @ scale_Hessian @gv + 1e-30)
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
    d = oracles[0].d
    kappa = oracles[0].kappa
    ts = []
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(d) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [1]
    init_norm = np.linalg.norm(init_w - w_opt)
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    invG = np.eye(d) / max_L
    init_time = time.time()
    for epo in range(epochs):
        gamma_k = max_M * np.sqrt(max_L) * np.linalg.norm(init_w - w_opt) * (1 - 1 /(d * kappa))** epo
        invG = invG / (1 + gamma_k)**2
        u = (1 + gamma_k)**2 * u
        for i in range(N):
            #w = np.linalg.pinv(B) @ (u - g)
            w = invG @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
                
            scale_Hessian = (1 + gamma_k)**2 * Gs[i]
            scale_yy = (1 + gamma_k) * yy
            
            stoc_Hessian = scale_Hessian + scale_yy@scale_yy.T / (scale_yy.T@s) - \
                           (scale_Hessian @ s)@(s.T @ scale_Hessian) / (s.T @ scale_Hessian @s)
            ind = np.argmax(np.diag(stoc_Hessian) / oracles[i].hes_diag(w))
            gv = np.zeros([d, 1])
            gv[ind] = 1
            base_Hessian = oracles[i].hes(w)
            
            stoc_Hessian_2 = stoc_Hessian + (base_Hessian@ gv)@(gv.T@base_Hessian) / (gv.T @ base_Hessian @gv) - \
                           (stoc_Hessian@ gv)@(gv.T@stoc_Hessian) / (gv.T @ stoc_Hessian @gv)
            
            invG = invG - invG @ (base_Hessian @ gv) @ (gv.T @ base_Hessian) @ invG / (N * gv.T @ base_Hessian @ gv + gv.T @ base_Hessian @ invG @ base_Hessian @ gv)
            invG = invG + invG @ (stoc_Hessian @ gv) @ (gv.T @ stoc_Hessian) @ invG / (N * gv.T @ stoc_Hessian @ gv - gv.T @ stoc_Hessian @ invG @ stoc_Hessian @ gv)
            invG = invG - invG @ scale_yy @ scale_yy.T @ invG / (N * scale_yy.T @ s + scale_yy.T @ invG @ scale_yy)
            invG = invG + invG @ (scale_Hessian @ s)@(s.T @ scale_Hessian) @ invG / (N * s.T @ Gs[i] @ s - s.T @ scale_Hessian @ invG @ scale_Hessian @ s)
            B = B + (stoc_Hessian_2 - scale_Hessian) / N
            u = u + (stoc_Hessian_2 @ w - scale_Hessian @ ws[i]) / N
            g = g + yy / N
            
            Gs[i] = np.copy(stoc_Hessian_2)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt) / init_norm)
        ts.append(time.time() - init_time)
        
    return res, ts

def iqn_sr1_sol(oracles, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    ts = []
    d = oracles[0].d
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(d) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [np.linalg.norm(ws[-1] - w_opt)]
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    invG = np.eye(d) / max_L
    init_time = time.time()
    for _ in range(epochs):
        for i in range(N):
            #w = np.linalg.pinv(B) @ (u - g)
            w = invG @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            r = np.sqrt(s.T @ oracles[i].hes_vec(ws[i], s))
            
            vec_diff = Gs[i] @ s - yy
            stoc_Hessian = Gs[i] - vec_diff @ vec_diff.T / (vec_diff.T @ s)
            invG = invG + invG @ vec_diff @ vec_diff.T @ invG / (N * vec_diff.T @ s - vec_diff.T @ invG @ vec_diff) 
            
            B = B + (stoc_Hessian - Gs[i]) / N
            u = u + (stoc_Hessian @ w - Gs[i] @ ws[i]) / N
            g = g + yy / N
                        
            Gs[i] = np.copy(stoc_Hessian)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt))
        ts.append(time.time() - init_time)
        
    return res, ts

def sliqn_sr1_sol(oracles, max_L, max_M,
            w_opt, init_w, corr=False, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    d = oracles[0].d
    ts = []
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(d) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [1]
    init_norm = np.linalg.norm(init_w - w_opt)
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    invG = np.eye(d) / max_L
    init_time = time.time()
    for epo in range(epochs):
        gamma_k = max_M * np.sqrt(max_L) * np.linalg.norm(init_w - w_opt) * (1 - 1 /d)** epo
        invG = invG / (1 + gamma_k)**2
        u = (1 + gamma_k)**2 * u
        for i in range(N):
            w = invG @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            
            scale_Hessian = (1 + gamma_k)**2 * Gs[i]   
            scale_yy = (1 + gamma_k) * yy
            
            vec_diff = scale_Hessian @ s - scale_yy
            #stoc_Hessian = scale_Hessian - vec_diff @ vec_diff.T / (vec_diff.T @ s + 1e-30)
            stoc_Hessian = scale_Hessian
            base_Hessian = oracles[i].hes(w)
            ind = np.argmax(np.diag(stoc_Hessian) - np.diag(base_Hessian))
            gv = np.zeros([d, 1])
            gv[ind] = 1
            
            Hessian_diff = stoc_Hessian - base_Hessian
            stoc_Hessian_2 = stoc_Hessian - Hessian_diff @ gv @ gv.T @ Hessian_diff / (gv.T @ Hessian_diff @ gv + 1e-30)
            
            v = invG @ oracles[i].hes_vec(w, gv)
            invG = invG + invG@ Hessian_diff @ gv @ gv.T @ Hessian_diff @ invG / (N * gv.T @ Hessian_diff @ gv - gv.T @ Hessian_diff @ invG @ Hessian_diff @ gv + 1e-30)
            #invG = invG + invG @ vec_diff @ vec_diff.T @ invG / (N * vec_diff.T @ s - vec_diff.T @ invG @ vec_diff + 1e-30)
            B = B + (stoc_Hessian_2 - scale_Hessian) / N
            u = u + (stoc_Hessian_2 @ w - scale_Hessian @ ws[i]) / N
            g = g + yy / N
                        
            Gs[i] = np.copy(stoc_Hessian_2)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt) / init_norm)
        ts.append(time.time() - init_time)
        
    return res, ts
    
def sliqn_srk_sol(oracles, max_L, max_M, tau,
            w_opt, init_w, corr=False, epochs=200):
    N = len(oracles)
    Gs = []
    ws = []
    grads = []
    d = oracles[0].d
    ts = []
    I = np.eye(d)
    
    g = np.zeros_like(init_w)
    for i in range(N):
        Gs.append(np.eye(d) * max_L)
        ws.append(np.copy(init_w))
        grads.append(oracles[i].grad(init_w))
        g = g + grads[-1]        
    res = [1]
    init_norm = np.linalg.norm(init_w - w_opt)
    w = np.copy(init_w)
    B = np.copy(Gs[0])
    u = Gs[0] @ ws[0]
    g = g / N
    invG = np.eye(d) / max_L
    init_time = time.time()
    for epo in range(epochs):
        gamma_k = max_M * np.sqrt(max_L) * np.linalg.norm(init_w - w_opt) * (1 - tau /d)** epo
        invG = invG / (1 + gamma_k)**2
        u = (1 + gamma_k)**2 * u
        for i in range(N):
            w = invG @ (u - g)
            cur_grad = oracles[i].grad(w)
            s = w - ws[i]
            yy = cur_grad - grads[i]
            
            scale_Hessian = (1 + gamma_k)**2 * Gs[i]   
            scale_yy = (1 + gamma_k) * yy
            
            vec_diff = scale_Hessian @ s - scale_yy
            #stoc_Hessian = scale_Hessian - vec_diff @ vec_diff.T / (vec_diff.T @ s + 1e-30)
            stoc_Hessian = scale_Hessian
            base_Hessian = oracles[i].hes(w)
            inds = heapq.nlargest(tau, range(d), list(np.diag(stoc_Hessian) - oracles[i].hes_diag(w)).__getitem__ )
            U = I[:, inds]
            GU = stoc_Hessian @ U
            AU = oracles[i].hesU(w, U)
            DU = GU - AU
            stoc_Hessian_2 = stoc_Hessian - DU @ np.linalg.pinv(U.T @ DU + 1e-30*np.eye(tau)) @ DU.T
            V = invG @ AU
            Delta = U - V
            invG = invG + invG @ DU @ ((np.linalg.pinv(N * U.T @ DU - DU.T@invG @ DU +1e-30*np.eye(tau)))@DU.T @ invG )
            #invG = invG + invG @ vec_diff @ vec_diff.T @ invG / (N * vec_diff.T @ s - vec_diff.T @ invG @ vec_diff + 1e-30)
            B = B + (stoc_Hessian_2 - scale_Hessian) / N
            u = u + (stoc_Hessian_2 @ w - scale_Hessian @ ws[i]) / N
            g = g + yy / N
                        
            Gs[i] = np.copy(stoc_Hessian_2)
            grads[i] = np.copy(cur_grad)
            ws[i] = np.copy(w)
            
        res.append(np.linalg.norm(ws[-1] - w_opt) / init_norm)
        ts.append(time.time() - init_time)
        
    return res, ts

def grsr1_sol(oracles, w, L, M, epochs, corr=True):
    oracle = oracles[0]
    d=oracle.d
    G=L*np.eye(d)
    invG=1/L*np.eye(d)
    gw = oracle.grad(w)
  #  invG = np.linalg.pinv(G)
    res = [np.linalg.norm(gw)]
    time_t=[0]
    ts=time.time()
    for i in range(epochs):
        dw = - invG @ gw
        if corr:
            r = np.sqrt(dw.T @ oracle.hesU(w, dw))
            G = G * (1 + M * r)
            invG = invG / (1 + M * r)


        w_ = w + dw    
        gw_ = oracle.grad(w_)

        if np.linalg.norm(gw_)<np.linalg.norm(gw):
            gw = gw_
            w=w_
            corr=True
        else:
            corr = False
        
        ind = np.argmax(np.diag(G) - oracle.hes_diag(w))

        u = np.zeros([d, 1])
        u[ind] = 1
    
        Gu = G @ u
        Au = oracle.hesU(w, u)
        G = G - (Gu - Au) @ (Gu - Au).T / (u.T @ (Gu - Au) + 1e-30)
        
        v = invG @ Au
        invG = invG + (u - v) @ (u - v).T / (u.T @ oracle.hesU(w, u-v) + 1e-30) 
        gw = oracle.grad(w)
        res.append(np.linalg.norm(gw))
        time_t.append(time.time()-ts)
        if i%50 == 0:
            print(res[-1])
    print(res[-1])
    return res,time_t

# function to convert to superscript
def get_super(x):
	normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
	super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
	res = x.maketrans(''.join(normal), ''.join(super_s))
	return x.translate(res)

# In[11]:
num_of_instances = 10
d = 100
xi = 12

w = np.random.randn(d, 1) / 10

oracles = []
A_avg = 0
b_avg = 0

for i in range(num_of_instances):
    oracles.append(Quadratic(d, xi))
    A_avg += oracles[-1].A
    b_avg += oracles[-1].b

A_avg /= num_of_instances
b_avg /= num_of_instances
w_opt = -np.linalg.pinv(A_avg) @ b_avg

print(len(oracles))

res, ws = grad_sol(w, 100000, 1e-6, A_avg, b_avg)

newton, _, warmup_w = newton_sol(oracles[0], w, 40)

max_L = 2e4
max_M = 0.03

#init_w = np.random.randn(d, 1) / 10
#init_w = ws[250]
init_w = w_opt + np.random.randn(d, 1) / 10

iqn, iqn_ts = iqn_sol(oracles, max_L, w_opt, init_w, epochs=10)


iqn, iqn_ts = iqn_sol(oracles, max_L, w_opt, init_w, epochs=400)
#iqs = iqs_sol(oracles, max_L, max_M, w_opt, init_w, corr=False, epochs=500)
max_L = 2e4
max_M = 0
sliqn, sliqn_ts = sliqn_sol(oracles, max_L, max_M, w_opt, init_w, corr=False, epochs=400)

#small kappa
#max_L = 1e2
max_L = 2e5
max_M = 0
sliqn_sr1, sliqn_sr1_ts = sliqn_sr1_sol(oracles, max_L, max_M, w_opt, init_w, corr=False, epochs=400)

#small kappa
#max_L = 4e3
max_L = 4e5
max_M = 0
tau = 5
#tau = 2
sliqn_srk, sliqn_srk_ts = sliqn_srk_sol(oracles, max_L, max_M, tau, w_opt, init_w, corr=False, epochs=400)

res_list = []
res_list.append(iqn)
res_list.append(sliqn)
res_list.append(sliqn_sr1)
res_list.append(sliqn_srk)

import pickle
with open("quadratic" + str(xi) +".pkl", "wb") as f:
    pickle.dump(res_list, f)

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
plt.plot(iqn[:150], '-', label='IQN', linewidth=2)
plt.plot(sliqn[:150], '-.', label='SLIQN', linewidth=2)
#plt.plot(iqn_sr1[:400], '--', label='iqn_sr1', linewidth=2)
plt.plot(sliqn_sr1[:150], ':', label='LISR1', linewidth=2)
plt.plot(sliqn_srk[:150], '--', label='LISR-k', linewidth=2)
#plt.plot(grsr1[:200], "-.", label="grsr1", linewidth=2)

ax.grid()
ax.legend()
ax.set_yscale('log')  
# plt.xscale('log') 
# plt.ylim(top=1e2) 
kappa_avg = np.max(A_avg) / np.min(np.diag(A_avg))
ax.set_ylabel('Normalized Error')
ax.set_xlabel('No of effective passes')
ax.set_title('Quadratic Function Minimization')
plt.tight_layout()
plt_name = "sliqn_quadratic.pdf"
plt.savefig(plt_name, format='pdf', bbox_inches='tight', dpi=300)
