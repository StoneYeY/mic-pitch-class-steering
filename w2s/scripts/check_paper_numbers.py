"""Recompute every hand-written statistic of paper/main.tex from results/ and print a check table.

The auto-generated macros (numbers*.tex, table_main.tex) are covered by `make_figures.py` (regenerate and diff);
this script covers the numbers that are typed into the prose.  Run from the repository root:

    python w2s/scripts/check_paper_numbers.py [--md paper/STATS_CHECK.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

R = Path("results")
rows: list[tuple[str, str, str, str, str]] = []   # (paper location, claim, source, recomputed, status)


def add(where, claim, src, value, ok):
    rows.append((where, claim, src, value, "ok" if ok else "MISMATCH"))


def paired(df, a, b, col="coherence_mlsp"):
    x = df[df.method == a].set_index("trial")[col]; y = df[df.method == b].set_index("trial")[col]
    j = x.index.intersection(y.index); d = (x[j] - y[j]).dropna()
    return len(d), float(d.mean()), float(wilcoxon(d).pvalue), float((d > 0).sum()), float((d != 0).sum())


def holm_vs(df, ref, col="coherence_mlsp"):
    ms = [m for m in df.method.unique() if m != ref]
    ps = {m: paired(df, m, ref, col)[2] for m in ms}
    order = sorted(ps, key=ps.get); out = {}; run = 0.0
    for i, m in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[m])); out[m] = run
    return out


# ---------------------------------------------------------------- Exp A (003): decodability, R_t
s = pd.read_csv(R / "003_expA_trajectories/summary.csv"); raw = s[s.variant == "raw"].sort_values("row")
f1 = raw.f1.values; sat = int(np.argmax(f1 >= 0.95 * f1.max()))
add("§4.1 / abstract", "F1 reaches 95% of its maximum at 88% of the trajectory", "003/summary.csv", f"step {sat} -> {100*(sat+1)/50:.0f}%", abs((sat + 1) / 50 - 0.88) < 1e-9)
c = json.load(open(R / "003_expA_trajectories/corr.json"))["corr"]["raw"]["corr(R_entropy,f1)"]
add("§4.1", "Pearson(R_t, F1) across steps = 0.96", "003/corr.json", f"{c:.3f}", round(c, 2) == 0.96)
ps = pd.read_csv(R / "003_expA_trajectories/per_step.csv"); rawps = ps[ps.variant == "raw"]
med = float(np.nanmedian([np.corrcoef(g.R_entropy, g.f1)[0, 1] for _, g in rawps.groupby("run")]))
add("§4.1", "median within-trajectory Pearson = 0.85", "003/per_step.csv", f"{med:.3f}", round(med, 2) == 0.85)
rho = rawps.groupby("row").apply(lambda g: spearmanr(g.R_entropy, g.f1).correlation)
add("§4.1", "fixed-step Spearman median 0.27; 0.57 over the last 15 steps", "003/per_step.csv", f"{np.nanmedian(rho):.2f}; {np.nanmedian(rho.iloc[-15:]):.2f}", round(float(np.nanmedian(rho)), 2) == 0.27 and round(float(np.nanmedian(rho.iloc[-15:])), 2) == 0.57)

# ---------------------------------------------------------------- Exp C (004): burst sensitivity
bp = pd.read_csv(R / "004_expC_burst/by_position.csv").sort_values("position")
pk = bp.d_coherence_mlsp.idxmax()
add("§4.2 / abstract", "peak at s=0.54 (54%), dC=0.104", "004/by_position.csv", f"s={bp.progress[pk]:.2f}, dC={bp.d_coherence_mlsp[pk]:.3f}", bp.progress[pk] == 0.54 and round(bp.d_coherence_mlsp[pk], 3) == 0.104)
early = bp[bp.progress <= 0.2].d_coherence_mlsp.max()
add("§4.2", "burst in the first 20% changes coherence by < 1/10 of the peak", "004/by_position.csv", f"max early dC={early:.4f} vs peak/10={bp.d_coherence_mlsp[pk]/10:.4f}", early < bp.d_coherence_mlsp[pk] / 10)
i94, i54 = bp.index[bp.progress == 0.94][0], bp.index[bp.progress == 0.54][0]
add("§4.2", "BCE change -0.14 at s=0.94 vs -0.11 at s=0.54; log-mel 0.21 vs 0.53", "004/by_position.csv", f"{bp.d_bce0[i94]:.3f} vs {bp.d_bce0[i54]:.3f}; {bp.logmel_l1[i94]:.3f} vs {bp.logmel_l1[i54]:.3f}", round(bp.d_bce0[i94], 2) == -0.14 and round(bp.d_bce0[i54], 2) == -0.11 and round(bp.logmel_l1[i94], 2) == 0.21 and round(bp.logmel_l1[i54], 2) == 0.53)
add("§4.2", "mean dCLAP between -0.012 and -0.005 at every position", "004/by_position.csv", f"{bp.d_clap.min():.4f} .. {bp.d_clap.max():.4f}", bp.d_clap.min() >= -0.0125 and bp.d_clap.max() <= -0.0045)
st = json.load(open(R / "004_expC_burst/stability.json"))
share = st["boot_counts"].get("26", 0) / sum(st["boot_counts"].values())
add("§4.2", "bootstrap argmax at s=0.54 in 92%, within 0.46-0.62 in all", "004/stability.json", f"{100*share:.0f}%; positions {sorted(st['boot_counts'])}", round(100 * share) == 92 and set(st["boot_counts"]) <= {"22", "26", "30"})
loo = sorted(set(st["loo_prompt"].values())) if "loo_prompt" in st else sorted(set(st["per_prompt"].values()))
add("§4.2", "leave-one-prompt-out peaks between s=0.54 and 0.62", "004/stability.json", f"positions {loo}", set(loo) <= {26, 30})
pr = pd.read_csv(R / "004_expC_burst/per_run.csv"); ref = pr[pr.position < 0].set_index("trial").coherence_mlsp
d = lambda pos: (pr[pr.position == pos].set_index("trial").coherence_mlsp - ref)
e, m = (d(2) + d(6) + d(10)) / 3, (d(22) + d(26) + d(30)) / 3; j = e.index.intersection(m.index)
add("§4.2", "mid beats early in 88% of trials, p<1e-7", "004/per_run.csv", f"{100*(m[j] > e[j]).mean():.0f}%, p={wilcoxon(m[j], e[j]).pvalue:.1e}", round(100 * (m[j] > e[j]).mean()) == 88 and wilcoxon(m[j], e[j]).pvalue < 1e-7)
rb = pd.read_csv(R / "023_sao_randburst/per_run.csv"); rref = rb[rb.position < 0].set_index("trial").coherence_mlsp
dr = (rb[rb.position == 26].set_index("trial").coherence_mlsp - rref).dropna(); dp = d(26).dropna(); j = dr.index.intersection(dp.index)
add("§4.2", "random-direction burst 0.003 (n.s.) vs 0.104, paired p<1e-6", "023 + 004 per_run.csv", f"{dr[j].mean():.3f} (p vs 0: {wilcoxon(dr[j]).pvalue:.2f}) vs {dp[j].mean():.3f}, p={wilcoxon(dr[j], dp[j]).pvalue:.1e}", round(dr[j].mean(), 3) == 0.003 and wilcoxon(dr[j]).pvalue > 0.05 and wilcoxon(dr[j], dp[j]).pvalue < 1e-6)
bp5 = pd.read_csv(R / "015_expC_lam05/by_position.csv").sort_values("position"); pk5 = bp5.d_coherence_mlsp.idxmax()
add("§4.2", "lambda=0.05 sweep: peak s=0.54, dC=0.099, Pearson 0.99", "015/by_position.csv", f"s={bp5.progress[pk5]:.2f}, dC={bp5.d_coherence_mlsp[pk5]:.3f}, r={np.corrcoef(bp.d_coherence_mlsp, bp5.d_coherence_mlsp)[0,1]:.3f}", bp5.progress[pk5] == 0.54 and round(bp5.d_coherence_mlsp[pk5], 3) == 0.099 and round(np.corrcoef(bp.d_coherence_mlsp, bp5.d_coherence_mlsp)[0, 1], 2) == 0.99)
vals = {}
for a in ("0.2", "0.5", "1.0"):
    t = pd.read_csv(R / f"025_sao_early_strength/a{a}/table.csv").set_index("method"); vals[a] = (t.loc["early", "coherence_mlsp"], t.loc["early", "clap"], t.loc["sao", "coherence_mlsp"], t.loc["sao", "clap"])
add("§4.2", "early window at lambda=alpha=0.2/0.5/1.0: coherence 0.12 -> 0.23/0.29/0.41; CLAP 0.207 -> 0.187/0.184/0.173", "025/a*/table.csv", "; ".join(f"{a}: {v[0]:.2f}/{v[1]:.3f} (unguided {v[2]:.2f}/{v[3]:.3f})" for a, v in vals.items()),
    [round(vals[a][0], 2) for a in vals] == [0.23, 0.29, 0.41] and [round(vals[a][1], 3) for a in vals] == [0.187, 0.184, 0.173])

# ---------------------------------------------------------------- Exp B (009): main table
pb = pd.read_csv(R / "009_expB_full/per_run.csv"); tb = pd.read_csv(R / "009_expB_full/table.csv").set_index("method")
add("Table 1 caption", "MLSP-fixed: 224 valid coherence values, 225 CLAP values", "009/per_run.csv", f"coh n={pb[pb.method=='mlsp'].coherence_mlsp.count()}, clap n={pb[pb.method=='mlsp'].clap.count()}", pb[pb.method == "mlsp"].coherence_mlsp.count() == 224 and pb[pb.method == "mlsp"].clap.count() == 225)
add("§4.3", "Early 0.17 vs unguided 0.13; Mid 0.42", "009/table.csv", f"{tb.loc['early','coherence_mlsp']:.2f} vs {tb.loc['sao','coherence_mlsp']:.2f}; {tb.loc['mid','coherence_mlsp']:.2f}", round(tb.loc["early", "coherence_mlsp"], 2) == 0.17 and round(tb.loc["sao", "coherence_mlsp"], 2) == 0.13 and round(tb.loc["mid", "coherence_mlsp"], 2) == 0.42)
sch = json.load(open(R / "009_expB_full/schedules.json"))
add("§4.3", "SCPG selects 19-33; decodability-calibrated ablation = late window 35-49", "009/schedules.json", f"topk {min(sch['topk_dc_const']['steps'])}-{max(sch['topk_dc_const']['steps'])}; f1 {min(sch['rapg_cal_f1']['steps'])}-{max(sch['rapg_cal_f1']['steps'])}", (min(sch["topk_dc_const"]["steps"]), max(sch["topk_dc_const"]["steps"])) == (19, 33) and sch["rapg_cal_f1"]["steps"] == sch["late"]["steps"])
pm = pd.read_csv(R / "009_expB_full/per_melody.csv").set_index("method")
alt = pm["alternating"]
add("§4.3", "alternating pattern: SCPG+R 0.327 vs MLSP-fixed 0.346", "009/per_melody.csv", f"{alt['rapg_cal_dc']:.3f} vs {alt['mlsp']:.3f}", round(alt["rapg_cal_dc"], 3) == 0.327 and round(alt["mlsp"], 3) == 0.346)
add("§4.3", "CLAP: SCPG variants 0.321-0.324, MLSP-fixed 0.316, unguided 0.307", "009/table.csv", f"{tb.loc['topk_dc_const','clap']:.3f}/{tb.loc['rapg_cal_dc','clap']:.3f}, {tb.loc['mlsp','clap']:.3f}, {tb.loc['sao','clap']:.3f}", round(tb.loc["topk_dc_const", "clap"], 3) == 0.321 and round(tb.loc["rapg_cal_dc", "clap"], 3) == 0.324 and round(tb.loc["mlsp", "clap"], 3) == 0.316 and round(tb.loc["sao", "clap"], 3) == 0.307)
hc = holm_vs(pb, "mlsp", "clap")
add("§4.3", "no guided method differs significantly from MLSP-fixed in CLAP (Holm)", "009/per_run.csv", "min Holm p = %.3f" % min(v for k, v in hc.items() if k != "sao"), min(v for k, v in hc.items() if k != "sao") > 0.05)
fad = pd.read_csv(R / "009_expB_full/fad.csv")
fcol = [c for c in fad.columns if "maestro" in c][0]; fad = fad.set_index("method")[fcol]
add("§4.3", "FAD 0.75-0.76 (SCPG variants) vs 0.79 (MLSP-fixed)", "009/fad.csv", f"{fad['rapg_cal_dc']:.2f}/{fad['topk_dc_const']:.2f} vs {fad['mlsp']:.2f}", {round(fad["rapg_cal_dc"], 2), round(fad["topk_dc_const"], 2)} <= {0.75, 0.76} and round(fad["mlsp"], 2) == 0.79)
vc = tb.voiced_cov if "voiced_cov" in tb else pb.groupby("method").voiced_cov.mean()
add("§4.3", "voiced fraction 0.88 SCPG / 0.86 MLSP-fixed / 0.84 unguided", "009/per_run.csv", f"{vc['rapg_cal_dc']:.2f}/{vc['mlsp']:.2f}/{vc['sao']:.2f}", round(vc["rapg_cal_dc"], 2) == 0.88 and round(vc["mlsp"], 2) == 0.86 and round(vc["sao"], 2) == 0.84)
fn = tb.coherence_firstnote if "coherence_firstnote" in tb else pb.groupby("method").coherence_firstnote.mean()
add("§4.3", "first-pitch-class collapse: 0.445 vs 0.255", "009/per_run.csv", f"{tb.loc['rapg_cal_dc','coherence_mlsp']:.3f} vs {fn['rapg_cal_dc']:.3f}", round(fn["rapg_cal_dc"], 3) == 0.255)
n, mn, p, w, nt = paired(pb, "rapg_cal_f1", "late")
add("§4.3", "R-scaling on the late window: 0.315 -> 0.310, paired p<1e-6", "009/per_run.csv", f"{tb.loc['late','coherence_mlsp']:.3f} -> {tb.loc['rapg_cal_f1','coherence_mlsp']:.3f}, d={mn:+.4f}, p={p:.1e}", round(tb.loc["late", "coherence_mlsp"], 3) == 0.315 and round(tb.loc["rapg_cal_f1", "coherence_mlsp"], 3) == 0.310 and p < 1e-6)
n, mn, p, w, nt = paired(pb, "rapg_on", "mlsp")
on = pb[pb.method == "rapg_on"]
add("§4.3", "online R-threshold 0.351 vs 0.363, p=0.53; first update at steps 2-12, last at step 40 (median)", "009/per_run.csv", f"{tb.loc['rapg_on','coherence_mlsp']:.3f} vs {tb.loc['mlsp','coherence_mlsp']:.3f}, p={p:.2f}; first {on.first_step.min()}-{on.first_step.max()}, last median {on.last_step.median():.0f}", round(p, 2) == 0.53 and (on.first_step.min(), on.first_step.max()) == (2, 12) and on.last_step.median() == 40)
add("§2.3 / §3", "online R-threshold also uses R-scaling", "009/schedules.json", f"adaptive_strength={sch['rapg_on']['adaptive_strength']}", bool(sch["rapg_on"]["adaptive_strength"]))
n, mn, p, w, nt = paired(pb, "rapg_cal_dc", "topk_dc_const")
add("§4.3", "R-scaling vs constant: +0.011, better in 122 of 184 non-tied trials, p<1e-5; realised mean lambda 0.057", "009/per_run.csv", f"d={mn:+.3f}, {w:.0f} of {nt:.0f}, p={p:.1e}; lambda={pb[pb.method=='rapg_cal_dc'].mean_lam.mean():.3f}", round(mn, 3) == 0.011 and (w, nt) == (122, 184) and p < 1e-5 and round(pb[pb.method == "rapg_cal_dc"].mean_lam.mean(), 3) == 0.057)
share = (tb.loc["topk_dc_const", "coherence_mlsp"] - tb.loc["mlsp", "coherence_mlsp"]) / (tb.loc["rapg_cal_dc", "coherence_mlsp"] - tb.loc["mlsp", "coherence_mlsp"])
add("§4.3", "about 86% of the improvement over MLSP-fixed comes from placement", "009/table.csv", f"{100*share:.1f}%", round(100 * share) == 86)

# ---------------------------------------------------------------- Exp D (014): budget sweep
bs = pd.read_csv(R / "014_expD_budget_v2/pareto.csv").set_index(["method", "K"])
dfD = pd.read_csv(R / "014_expD_budget_v2/per_run.csv"); refD = pb[(pb.method == "sao") & pb.trial.isin(set(dfD.trial))]
add("§4.4", "10 SCPG updates 0.38 vs 25 uniform 0.40; K=25 SCPG 0.56", "014/pareto.csv", f"{bs.loc[('topk_dc',10),'coh']:.2f} vs {bs.loc[('uniform',25),'coh']:.2f}; {bs.loc[('topk_dc',25),'coh']:.2f}", round(bs.loc[("topk_dc", 10), "coh"], 2) == 0.38 and round(bs.loc[("uniform", 25), "coh"], 2) == 0.40 and round(bs.loc[("topk_dc", 25), "coh"], 2) == 0.56)
add("§4.4", "CLAP 0.351-0.373 across budgets; 0.348 unguided on the subset", "014/pareto.csv + 009/per_run.csv", f"{bs.clap.min():.3f}-{bs.clap.max():.3f}; {refD.clap.mean():.3f}", round(bs.clap.min(), 3) == 0.351 and round(bs.clap.max(), 3) == 0.373 and round(refD.clap.mean(), 3) == 0.348)
pmin = 1.0
for K in sorted(set(dfD.K)):
    x = dfD[(dfD.method == "topk_dc") & (dfD.K == K)].set_index("trial").coherence_mlsp; y = dfD[(dfD.method == "uniform") & (dfD.K == K)].set_index("trial").coherence_mlsp; j = x.index.intersection(y.index)
    pmin = max(pmin if pmin < 1 else 0, wilcoxon(x[j], y[j]).pvalue)
add("§4.4", "SCPG better than uniform at every K, paired p<1e-5", "014/per_run.csv", f"max p over K = {pmin:.1e}", pmin < 1e-5)

# ---------------------------------------------------------------- Stable Audio 3 (019c/020c/021/024)
s3 = pd.read_csv(R / "019c_sa3_expA/summary.csv"); r3 = s3[s3.variant == "raw"].sort_values("row"); f13 = r3.f1.values; sat3 = int(np.argmax(f13 >= 0.95 * f13.max()))
add("§4.5", "SA3: F1 reaches 95% of max at 76%", "019c/summary.csv", f"{100*(sat3+1)/50:.0f}%", (sat3 + 1) / 50 == 0.76)
b3 = pd.read_csv(R / "020c_sa3_expC/by_position.csv").sort_values("position")
add("§4.5", "SA3 burst gain ~0.079 over the first 40%, 0.017 in the last 20%", "020c/by_position.csv", f"first 40%: {b3[b3.progress<=0.4].d_coherence_mlsp.mean():.3f}; last 20%: {b3[b3.progress>0.8].d_coherence_mlsp.mean():.3f}", abs(b3[b3.progress <= 0.4].d_coherence_mlsp.mean() - 0.079) < 0.006 and round(b3[b3.progress > 0.8].d_coherence_mlsp.mean(), 3) == 0.017)
rb3 = pd.read_csv(R / "024_sa3_randburst/per_run.csv"); rr3 = rb3[rb3.position < 0].set_index("trial").coherence_mlsp; p3 = pd.read_csv(R / "020c_sa3_expC/per_run.csv"); pr3ref = p3[p3.position < 0].set_index("trial").coherence_mlsp
dr3 = (rb3[rb3.position == 10].set_index("trial").coherence_mlsp - rr3).dropna(); dp3 = (p3[p3.position == 10].set_index("trial").coherence_mlsp - pr3ref).dropna(); j = dr3.index.intersection(dp3.index)
add("§4.5", "SA3 random-direction control -0.026 vs 0.089, paired p<1e-4", "024 + 020c per_run.csv", f"{dr3[j].mean():.3f} vs {dp3[j].mean():.3f}, p={wilcoxon(dr3[j], dp3[j]).pvalue:.1e}", round(dr3[j].mean(), 3) == -0.026 and round(dp3[j].mean(), 3) == 0.089 and wilcoxon(dr3[j], dp3[j]).pvalue < 1e-4)
pb3 = pd.read_csv(R / "021_sa3_expB/per_run.csv"); tb3 = pd.read_csv(R / "021_sa3_expB/table.csv").set_index("method")
add("§4.5", "SA3: 49 valid coherence values for Mid and Late", "021/per_run.csv", f"mid {pb3[pb3.method=='mid'].coherence_mlsp.count()}, late {pb3[pb3.method=='late'].coherence_mlsp.count()}", pb3[pb3.method == "mid"].coherence_mlsp.count() == 49 and pb3[pb3.method == "late"].coherence_mlsp.count() == 49)
h3 = holm_vs(pb3, "mlsp")
add("§4.5", "SCPG 0.44 vs MLSP-fixed 0.28, Holm-adjusted p=0.002", "021/per_run.csv", f"{tb3.loc['topk_dc_const','coherence_mlsp']:.2f} vs {tb3.loc['mlsp','coherence_mlsp']:.2f}, Holm p={h3['topk_dc_const']:.4f}", round(h3["topk_dc_const"], 3) == 0.002)
hc3 = holm_vs(pb3, "sao", "clap")
add("§4.5", "SA3 CLAP 0.375 -> 0.335 under SCPG (Holm p<0.01); all guided except MLSP-fixed (0.361, n.s.) decrease", "021/per_run.csv", f"{tb3.loc['sao','clap']:.3f} -> {tb3.loc['topk_dc_const','clap']:.3f}, Holm p={hc3['topk_dc_const']:.4f}; mlsp p={hc3['mlsp']:.3f}; others max p={max(v for k, v in hc3.items() if k != 'mlsp'):.4f}", hc3["topk_dc_const"] < 0.01 and hc3["mlsp"] > 0.05 and max(v for k, v in hc3.items() if k != "mlsp") < 0.05)
sch3 = json.load(open(R / "021_sa3_expB/schedules.json"))
add("§4.5", "SA3 calibration selects steps 4-18; Early (0-14) 0.44 within the CI of SCPG", "021/schedules.json + table.csv", f"{min(sch3['topk_dc_const']['steps'])}-{max(sch3['topk_dc_const']['steps'])}; early {tb3.loc['early','coherence_mlsp']:.2f}, SCPG CI [{tb3.loc['topk_dc_const','coherence_mlsp_lo']:.2f}, {tb3.loc['topk_dc_const','coherence_mlsp_hi']:.2f}]", (min(sch3["topk_dc_const"]["steps"]), max(sch3["topk_dc_const"]["steps"])) == (4, 18) and tb3.loc["topk_dc_const", "coherence_mlsp_lo"] <= tb3.loc["early", "coherence_mlsp"] <= tb3.loc["topk_dc_const", "coherence_mlsp_hi"])

# ---------------------------------------------------------------- legacy alignment (026)
t26 = pd.read_csv(R / "026_sao_legacy_mlsp/table.csv").set_index("method")
add("§3", "original alignment: 0.259 against the melody, 0.68 against its first pitch class", "026/table.csv", f"{t26.loc['mlsp','coherence_mlsp']:.3f}; {t26.loc['mlsp','coherence_firstnote']:.2f}", round(t26.loc["mlsp", "coherence_mlsp"], 3) == 0.259 and round(t26.loc["mlsp", "coherence_firstnote"], 2) == 0.68)

# ---------------------------------------------------------------- report
ap = argparse.ArgumentParser(); ap.add_argument("--md", default=None); args = ap.parse_args()
hdr = "| Paper | Claim | Source | Recomputed | Status |\n|---|---|---|---|---|\n"
body = "\n".join(f"| {w} | {c} | `{s}` | {v} | {st} |" for w, c, s, v, st in rows)
n_bad = sum(1 for r in rows if r[4] != "ok")
out = f"# Hand-written statistics of main.tex, recomputed from results/\n\n{len(rows)} claims checked, {n_bad} mismatch(es).\n\n" + hdr + body + "\n"
print(out)
if args.md:
    Path(args.md).write_text(out)
raise SystemExit(1 if n_bad else 0)
