import pickle, time
from core import *

def rule3_table():
    sA0, eA = sample_world(10)          # world used for all do-samples
    sB0, eB = sample_world(20)          # independent no-intervention world
    sC0, eC = sample_world(30)          # control: another independent no-intervention world
    SbaseA, _ = run("base", 0.0, ("none",), sA0, eA)
    SbaseB, _ = run("base", 0.0, ("none",), sB0, eB); OB = observe(SbaseB, 1)
    SbaseC, _ = run("base", 0.0, ("none",), sC0, eC); OC = observe(SbaseC, 2)
    OA_base = observe(SbaseA, 3)
    rows = []
    def evaluate(label, O, OA_ref):
        # time-resolved KS: 20 years x 2 species, Holm
        p, d = [], []
        for v in range(2):
            for t in range(1, 21):
                r = ks_2samp(O[v, :, t], OB[v, :, t]); p.append(r.pvalue); d.append(r.statistic)
        sig = holm(p)
        # old-style pooled KS (time ignored)
        pooled = [ks_2samp(O[v].ravel(), OB[v].ravel()) for v in range(2)]
        # paired signed-rank per year (same initial conditions), Holm
        pw = []
        for v in range(2):
            for t in range(1, 21):
                diff = O[v, :, t]-OA_ref[v, :, t]
                pw.append(wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0)
        sigw = holm(pw)
        return dict(label=label, ks_nsig=int(sig.sum()), ks_maxD=float(max(d)),
                    pooled_D=float(max(r.statistic for r in pooled)), pooled_sig=int(min(r.pvalue for r in pooled) < 0.025),
                    wil_nsig=int(sigw.sum()))
    ctrl = evaluate("control", OC, OA_base); ctrl.update(exp="ctrl", kind="-", s=0.0); rows.append(ctrl)
    for exp, levels in EXPS.items():
        for kind in KINDS:
            for k in levels:
                S, _ = run(exp, k, (kind,), sA0, eA)
                r = evaluate(f"{exp}/{kind}/{k}", observe(S, 4), OA_base); r.update(exp=exp, kind=kind, s=k); rows.append(r)
    return rows

if __name__ == "__main__":
    t0 = time.time(); rows = rule3_table(); pickle.dump(rows, open("rule3.pkl", "wb"))
    print("time %.1fs" % (time.time()-t0))
    print("%-18s %s  %s   %s   %s" % ("config", "KS по годам (знач. из 40) / maxD", "KS общий (D, 1/0)", "парный Вилкоксон (из 40)", ""))
    for r in rows:
        print("%-18s KS:%2d/40 D=%.2f | pooled D=%.2f sig=%d | Wilcoxon:%2d/40" % (r["label"], r["ks_nsig"], r["ks_maxD"], r["pooled_D"], r["pooled_sig"], r["wil_nsig"]))
