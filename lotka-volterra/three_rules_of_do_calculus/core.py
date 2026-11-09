import numpy as np
from scipy.stats import ks_2samp, wilcoxon, f as fdist, norm

A, B, G, D = 0.481, 0.0248, 0.926, 0.0275
X0, Y0 = 34.9, 3.86
N = 600
OBS_SIGMA = 0.03
EXPS = {"E1": [0.01, 0.02, 0.05, 0.1], "E2": [0.05, 0.1, 0.2, 0.4], "E3": [0.1, 0.25, 0.5, 1.0]}
KINDS = ["noise", "const", "sin"]

def rhs(model, s, z, k):
    x, y = s
    fx, fy = A - B*y, -G + D*x
    if model == "E1":  return np.stack([(fx+k*z)*x, (fy+k*z)*y])
    if model == "E2":
        m = np.maximum(0.0, 1+k*z); return np.stack([fx*x*m, fy*y*m])
    if model == "E3":  return np.stack([fx*x+k*z, fy*y+k*z])
    return np.stack([fx*x, fy*y])          # base

def step_year(model, k, s, z, n=20):
    h = 1.0/n
    for _ in range(n):
        k1 = rhs(model, s, z, k); k2 = rhs(model, s+0.5*h*k1, z, k)
        k3 = rhs(model, s+0.5*h*k2, z, k); k4 = rhs(model, s+h*k3, z, k)
        s = np.clip(s + h/6*(k1+2*k2+2*k3+k4), 1e-3, 1e5)
    return s

def sample_world(seed, n=N, T=21):
    g = np.random.default_rng(seed)
    s0 = np.stack([X0*np.exp(0.15*g.standard_normal(n)), Y0*np.exp(0.15*g.standard_normal(n))])
    eps = g.standard_normal((n, T))
    return s0, eps

def zval(zmode, t, s, eps):
    kind = zmode[0]
    n = s.shape[1]
    if kind == "none":  return np.zeros(n)
    if kind == "noise": return eps[:, t]
    if kind == "const": return np.ones(n)
    if kind == "sin":   return np.full(n, np.sin(2*np.pi*t/10))
    if kind == "endo":
        rho = zmode[1]
        return rho*(np.log(s[1])-np.log(10.0))/0.8 + np.sqrt(1-rho**2)*eps[:, t]
    raise ValueError(kind)

def run(model, k, zmode, s0, eps, T=21):
    n = s0.shape[1]
    S = np.empty((2, n, T)); Z = np.empty((n, T)); s = s0.copy(); S[:, :, 0] = s
    for t in range(T-1):
        z = zval(zmode, t, s, eps); Z[:, t] = z
        s = step_year(model, k, s, z); S[:, :, t+1] = s
    Z[:, T-1] = zval(zmode, T-1, s, eps)
    return S, Z

def observe(S, seed):
    return S*np.exp(OBS_SIGMA*np.random.default_rng(seed).standard_normal(S.shape))

def holm(p, alpha=0.05):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); sig = np.zeros(m, bool)
    for r, i in enumerate(o):
        if p[i] < alpha/(m-r): sig[i] = True
        else: break
    return sig
