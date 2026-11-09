import pickle, time
from core import *

TS = [3, 7, 11, 15]; CS = [-1.0, 0.0, 1.0]; WIN = 0.3

def rule2_test(model, k, rho, s0, eps, seed=7):
    S, Z = run(model, k, ("endo", rho), s0, eps)
    n = S.shape[1]; g = np.random.default_rng(seed)
    tests = []
    for t in TS:
        for c in CS:
            mask = np.abs(Z[:, t]-c) < WIN
            if mask.sum() < 30: continue
            s_do = step_year(model, k, S[:, :, t].copy(), np.full(n, c))          # do(z_t = c) for all units
            for v in range(2):
                a = S[v, mask, t+1]*np.exp(OBS_SIGMA*g.standard_normal(mask.sum()))   # observe z_t ~ c
                b = s_do[v]*np.exp(OBS_SIGMA*g.standard_normal(n))
                r = ks_2samp(a, b)
                tests.append(dict(t=t, c=c, v=v, D=r.statistic, p=r.pvalue, n_obs=int(mask.sum()),
                                  W=float(np.mean(a)-np.mean(b))))
    sig = holm([x["p"] for x in tests])
    return dict(n_tests=len(tests), n_sig=int(sig.sum()), maxD=max(x["D"] for x in tests),
                meanD=float(np.mean([x["D"] for x in tests])), tests=tests)

def rule2_table():
    s0, eps = sample_world(50)
    rows = []
    for rho in [0.0, 0.8]:
        r = rule2_test("base", 0.0, rho, s0, eps); r.update(exp="ctrl", rho=rho, s=0.0, label=f"base/rho={rho}"); rows.append(r)
        for exp, levels in EXPS.items():
            for k in levels:
                r = rule2_test(exp, k, rho, s0, eps); r.update(exp=exp, rho=rho, s=k, label=f"{exp}/rho={rho}/{k}"); rows.append(r)
    return rows

if __name__ == "__main__":
    t0 = time.time(); rows = rule2_table(); pickle.dump(rows, open("rule2.pkl", "wb"))
    print("time %.1fs" % (time.time()-t0))
    for r in rows:
        print("%-20s KS: значимых %2d/%2d | maxD=%.2f meanD=%.2f" % (r["label"], r["n_sig"], r["n_tests"], r["maxD"], r["meanD"]))
