import pickle, time
from core import *

def basis(lx, ly):
    return np.column_stack([np.ones_like(lx), lx, ly, lx**2, ly**2, lx*ly])

def rule1_tests(S, Z, seed=5):
    O = observe(S, seed)
    lx = np.log(O[0, :, 1:20]).ravel(); ly = np.log(O[1, :, 1:20]).ravel()   # state at t=1..19
    z = Z[:, 1:20].ravel()
    tg = [np.log(O[0, :, 2:21]).ravel(), np.log(O[1, :, 2:21]).ravel()]       # state at t+1
    B0 = basis(lx, ly); B1 = np.column_stack([B0, z, z*lx, z*ly])
    n = len(z); out = dict(F=[], FZ=[], KS=[], KSD=[], dR2=[])
    lo, hi = z < np.quantile(z, 1/3), z > np.quantile(z, 2/3)
    for y in tg:
        b0, *_ = np.linalg.lstsq(B0, y, rcond=None); r0 = y-B0@b0
        b1, *_ = np.linalg.lstsq(B1, y, rcond=None); r1 = y-B1@b1
        rss0, rss1 = r0@r0, r1@r1; df2 = n-B1.shape[1]
        Fst = ((rss0-rss1)/3)/(rss1/df2); out["F"].append(fdist.sf(Fst, 3, df2)); out["dR2"].append((rss0-rss1)/rss0)
        # Fisher-Z partial correlation of z and y given the 6 controls
        zr = z - B0@np.linalg.lstsq(B0, z, rcond=None)[0]
        r = np.corrcoef(zr, r0)[0, 1]; out["FZ"].append(2*norm.sf(abs(np.sqrt(n-B0.shape[1]-3)*np.arctanh(r))))
        ks = ks_2samp(r0[lo], r0[hi]); out["KS"].append(ks.pvalue); out["KSD"].append(ks.statistic)
    return dict(F=min(out["F"])*2, FZ=min(out["FZ"])*2, KSp=min(out["KS"])*2, KSD=max(out["KSD"]), dR2=max(out["dR2"]))

def rule1_table():
    s0, eps = sample_world(40)
    rows = []
    for rho in [0.0, 0.8]:
        S, Z = run("base", 0.0, ("endo", rho), s0, eps)
        r = rule1_tests(S, Z); r.update(exp="ctrl", rho=rho, s=0.0, label=f"control rho={rho}"); rows.append(r)
        for exp, levels in EXPS.items():
            for k in levels:
                S, Z = run(exp, k, ("endo", rho), s0, eps)
                r = rule1_tests(S, Z); r.update(exp=exp, rho=rho, s=k, label=f"{exp}/rho={rho}/{k}"); rows.append(r)
    sig = holm([r["F"] for r in rows])
    for r, s in zip(rows, sig): r["F_sig"] = int(s)
    return rows

if __name__ == "__main__":
    t0 = time.time(); rows = rule1_table(); pickle.dump(rows, open("rule1.pkl", "wb"))
    print("time %.1fs" % (time.time()-t0))
    for r in rows:
        print("%-22s F-test: p=%.1e sig=%d dR2=%.4f | FisherZ p=%.1e | KS D=%.2f p=%.1e" %
              (r["label"], r["F"], r["F_sig"], r["dR2"], r["FZ"], r["KSD"], r["KSp"]))
