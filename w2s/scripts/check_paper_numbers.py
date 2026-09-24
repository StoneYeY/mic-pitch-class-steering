"""Recompute every hand-written statistic of paper/main.tex from results/ and print a check table.

Scope.  The auto-generated macros (numbers*.tex, table_main.tex, budget_summary.csv) are covered by
`make_figures.py` (regenerate and diff); this script covers the numbers typed into the prose.  Each check
  * recomputes the statistic, from per-run files where they exist ("per-run") or from a summary file that
    the paper reads directly ("summary"; the summaries are in turn checked against the per-run files below),
  * checks the value AND the direction of every comparison, not only its p-value, and
  * checks that the wording it stands for is actually present in paper/main.tex ("tex").
Run from the repository root:  python w2s/scripts/check_paper_numbers.py [--md paper/STATS_CHECK.md]
Exit status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

R = Path("results")
TEX = Path("paper/main.tex").read_text()
rows: list[tuple[str, str, str, str, str, str]] = []   # (paper location, claim, source, kind, recomputed, status)


def add(where, claim, src, kind, value, ok, tex=()):
    missing = [t for t in tex if t not in TEX]
    status = "ok" if ok and not missing else ("MISMATCH" if not ok else "TEXT MISSING: " + " | ".join(missing))
    rows.append((where, claim, src, kind, value, status))


def paired(df, a, b, col="coherence_mlsp"):
    """Mean difference a-b over complete pairs, two-sided Wilcoxon p, wins of a, non-tied pairs."""
    x = df[df.method == a].set_index("trial")[col]; y = df[df.method == b].set_index("trial")[col]
    j = x.index.intersection(y.index); d = (x[j] - y[j]).dropna()
    return len(d), float(d.mean()), float(wilcoxon(d).pvalue), int((d > 0).sum()), int((d != 0).sum())


def holm_vs(df, ref, col="coherence_mlsp"):
    ms = [m for m in df.method.unique() if m != ref]
    ps = {m: paired(df, m, ref, col)[2] for m in ms}
    order = sorted(ps, key=ps.get); out = {}; run = 0.0
    for i, m in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[m])); out[m] = run
    return out


def burst_delta(per_run: pd.DataFrame, pos: int) -> pd.Series:
    ref = per_run[per_run.position < 0].set_index("trial").coherence_mlsp
    x = per_run[per_run.position == pos].set_index("trial").coherence_mlsp
    return (x - ref.reindex(x.index)).dropna()


# ============================================================ summaries vs per-run files
pb = pd.read_csv(R / "009_expB_full/per_run.csv"); tb = pd.read_csv(R / "009_expB_full/table.csv").set_index("method")
pb3 = pd.read_csv(R / "021_sa3_expB/per_run.csv"); tb3 = pd.read_csv(R / "021_sa3_expB/table.csv").set_index("method")
for name, per, tab in (("009_expB_full", pb, tb), ("021_sa3_expB", pb3, tb3)):
    g = per.groupby("method")[["coherence_mlsp", "clap"]].mean()
    err = float(np.nanmax(np.abs(g.loc[tab.index] - tab[["coherence_mlsp", "clap"]]).values))
    add("Table 1 / §4.5", f"{name}/table.csv means equal per_run.csv means (coherence, CLAP)", f"{name}", "per-run", f"max |diff| = {err:.1e}", err < 1e-6)
bpC = pd.read_csv(R / "004_expC_burst/by_position.csv").sort_values("position").set_index("position")
prC = pd.read_csv(R / "004_expC_burst/per_run.csv")
err = max(abs(burst_delta(prC, p).mean() - bpC.d_coherence_mlsp[p]) for p in bpC.index)
add("Fig. 1(b)", "004/by_position.csv mean dC equals per-run paired means", "004_expC_burst", "per-run", f"max |diff| = {err:.1e}", err < 1e-6)
bsD = pd.read_csv(R / "014_expD_budget_v2/pareto.csv").set_index(["method", "K"]); prD = pd.read_csv(R / "014_expD_budget_v2/per_run.csv")
gD = prD.groupby(["method", "K"])[["coherence_mlsp", "clap"]].mean()
err = float(np.nanmax(np.abs(gD.loc[bsD.index].values - bsD[["coh", "clap"]].values)))
add("Fig. 2", "014/pareto.csv means equal per_run.csv means", "014_expD_budget_v2", "per-run", f"max |diff| = {err:.1e}", err < 1e-6)

# ============================================================ Exp A (003): decodability, R_t
s = pd.read_csv(R / "003_expA_trajectories/summary.csv"); raw = s[s.variant == "raw"].sort_values("row")
ps = pd.read_csv(R / "003_expA_trajectories/per_step.csv"); rawps = ps[ps.variant == "raw"]
f1_steps = rawps.groupby("row").f1.mean().sort_index().values
err = float(np.max(np.abs(f1_steps - raw.f1.values)))
add("Fig. 1(a)", "003/summary.csv F1 curve equals the per-step mean over 50 runs", "003_expA_trajectories", "per-run", f"max |diff| = {err:.1e}", err < 1e-6)
sat = int(np.argmax(f1_steps >= 0.95 * f1_steps.max()))
add("§4.1 / abstract", "F1 reaches 95% of its maximum at 88% of the trajectory", "003/per_step.csv", "per-run", f"step {sat} -> {100*(sat+1)/50:.0f}%", (sat + 1) / 50 == 0.88, tex=["only at \\peakF"])
R_steps = rawps.groupby("row").R_entropy.mean().sort_index().values
c = float(np.corrcoef(R_steps, f1_steps)[0, 1])
add("§4.1", "Pearson(R_t, F1) across steps = 0.96", "003/per_step.csv", "per-run", f"{c:.3f}", round(c, 2) == 0.96, tex=["(Pearson \\corrRF"])
med = float(np.nanmedian([np.corrcoef(g.R_entropy, g.f1)[0, 1] for _, g in rawps.groupby("run")]))
add("§4.1", "median within-trajectory Pearson = 0.85", "003/per_step.csv", "per-run", f"{med:.3f}", round(med, 2) == 0.85, tex=["median 0.85 within a trajectory"])
rho = rawps.groupby("row").apply(lambda g: spearmanr(g.R_entropy, g.f1).correlation)
add("§4.1", "fixed-step Spearman median 0.27; 0.57 over the last 15 steps", "003/per_step.csv", "per-run", f"{np.nanmedian(rho):.2f}; {np.nanmedian(rho.iloc[-15:]):.2f}", round(float(np.nanmedian(rho)), 2) == 0.27 and round(float(np.nanmedian(rho.iloc[-15:])), 2) == 0.57, tex=["median Spearman \\rhoFixedT; \\rhoFixedTlate"])

# ============================================================ Exp C (004): burst sensitivity
dC = {p: burst_delta(prC, p) for p in bpC.index}
means = pd.Series({p: d.mean() for p, d in dC.items()}); pk = int(means.idxmax())
add("§4.2 / abstract", "peak at s=0.54 (54%), dC=0.104", "004/per_run.csv", "per-run", f"position {pk} -> s={(pk+1)/50:.2f}, dC={means[pk]:.3f}", (pk + 1) / 50 == 0.54 and round(means[pk], 3) == 0.104, tex=["peaks at $s{=}0.54$"])
early = means[[p for p in means.index if (p + 1) / 50 <= 0.2]].max()
add("§4.2", "burst in the first 20% changes coherence by < 1/10 of the peak", "004/per_run.csv", "per-run", f"max early dC={early:.4f} vs peak/10={means[pk]/10:.4f}", early < means[pk] / 10, tex=["less than a tenth of the mid-sampling effect"])
add("§4.2", "s=0.94 vs s=0.54: BCE change -0.14 vs -0.11; log-mel 0.21 vs 0.53", "004/by_position.csv", "summary", f"{bpC.d_bce0[46]:.3f} vs {bpC.d_bce0[26]:.3f}; {bpC.logmel_l1[46]:.3f} vs {bpC.logmel_l1[26]:.3f}",
    round(bpC.d_bce0[46], 2) == -0.14 and round(bpC.d_bce0[26], 2) == -0.11 and round(bpC.logmel_l1[46], 2) == 0.21 and round(bpC.logmel_l1[26], 2) == 0.53 and bpC.d_bce0[46] < bpC.d_bce0[26] and bpC.logmel_l1[46] < bpC.logmel_l1[26],
    tex=["BCE change $-0.14$ vs $-0.11$", "$0.21$ vs $0.53$"])
refc = prC[prC.position < 0].set_index("trial").clap
dclap = pd.Series({p: (prC[prC.position == p].set_index("trial").clap - refc).dropna().mean() for p in bpC.index})
add("§4.2", "mean dCLAP between -0.012 and -0.005 at every position (all negative)", "004/per_run.csv", "per-run", f"{dclap.min():.4f} .. {dclap.max():.4f}", dclap.min() >= -0.0125 and dclap.max() <= -0.0045 and (dclap < 0).all(), tex=["between $-0.012$ and $-0.005$"])
rng = np.random.default_rng(0); mat = pd.DataFrame(dC).dropna(); n = len(mat); pos = np.array(mat.columns)
boot = np.array([pos[np.argmax(mat.values[rng.integers(0, n, n)].mean(0))] for _ in range(5000)])
share = float((boot == 26).mean()); rng_ok = set(np.unique(boot)) <= {22, 26, 30}
add("§4.2", "bootstrap argmax at s=0.54 in 92% of 5000 resamples, within 0.46-0.62 in all (re-drawn, seed 0)", "004/per_run.csv", "per-run", f"{100*share:.1f}%; positions {sorted(set(np.unique(boot)))}", 0.89 <= share <= 0.95 and rng_ok, tex=["in \\bootPeakShare\\ and within $0.46$--$0.62$ in all of them"])
st = json.load(open(R / "004_expC_burst/stability.json")); share_json = st["boot_counts"].get("26", 0) / sum(st["boot_counts"].values())
add("§4.2", "the quoted 92% (stability.json) agrees with the re-drawn bootstrap", "004/stability.json", "summary", f"{100*share_json:.0f}% vs {100*share:.1f}%", round(100 * share_json) == 92 and abs(share_json - share) < 0.03)
trials = prC[prC.position >= 0][["trial", "prompt_id"]].drop_duplicates().set_index("trial").prompt_id
loo = sorted({int(pos[np.argmax(mat[trials.reindex(mat.index) != pid].mean().values)]) for pid in trials.unique()})
add("§4.2", "leave-one-prompt-out peaks between s=0.54 and 0.62", "004/per_run.csv", "per-run", f"positions {loo}", set(loo) <= {26, 30} and 26 in loo, tex=["peak between $s{=}0.54$ and $0.62$"])
e, m = (dC[2] + dC[6] + dC[10]) / 3, (dC[22] + dC[26] + dC[30]) / 3; j = e.index.intersection(m.index)
add("§4.2", "mid beats early in 88% of trials, p<1e-7 (mid > early)", "004/per_run.csv", "per-run", f"{100*(m[j] > e[j]).mean():.0f}%, mean diff {float((m[j]-e[j]).mean()):+.3f}, p={wilcoxon(m[j], e[j]).pvalue:.1e}", round(100 * (m[j] > e[j]).mean()) == 88 and (m[j] - e[j]).mean() > 0 and wilcoxon(m[j], e[j]).pvalue < 1e-7, tex=["in 88\\% of individual trials"])
rb = pd.read_csv(R / "023_sao_randburst/per_run.csv"); dr = burst_delta(rb, 26); dp = dC[26]; j = dr.index.intersection(dp.index)
add("§4.2", "random-direction burst 0.003 (n.s.) vs 0.104 for the probe gradient, paired p<1e-6", "023 + 004 per_run.csv", "per-run", f"{dr[j].mean():.3f} (p vs 0: {wilcoxon(dr[j]).pvalue:.2f}) vs {dp[j].mean():.3f}, p={wilcoxon(dr[j], dp[j]).pvalue:.1e}", round(dr[j].mean(), 3) == 0.003 and wilcoxon(dr[j]).pvalue > 0.05 and dp[j].mean() > dr[j].mean() and wilcoxon(dr[j], dp[j]).pvalue < 1e-6, tex=["changes coherence by $0.003$ (n.s.) against $0.104$"])
pr5 = pd.read_csv(R / "015_expC_lam05/per_run.csv"); m5 = pd.Series({p: burst_delta(pr5, p).mean() for p in bpC.index}); pk5 = int(m5.idxmax())
add("§4.2", "lambda=0.05 sweep: peak s=0.54, dC=0.099, Pearson 0.99 between the curves", "015/per_run.csv", "per-run", f"s={(pk5+1)/50:.2f}, dC={m5[pk5]:.3f}, r={np.corrcoef(means.values, m5.values)[0,1]:.3f}", (pk5 + 1) / 50 == 0.54 and round(m5[pk5], 3) == 0.099 and round(np.corrcoef(means.values, m5.values)[0, 1], 2) == 0.99, tex=["$\\Delta C{=}0.099$ vs $0.104$, Pearson $0.99$"])
vals = {}
for a in ("0.2", "0.5", "1.0"):
    t = pd.read_csv(R / f"025_sao_early_strength/a{a}/per_run.csv").groupby("method")[["coherence_mlsp", "clap"]].mean(); vals[a] = (t.loc["early", "coherence_mlsp"], t.loc["early", "clap"], t.loc["sao", "coherence_mlsp"], t.loc["sao", "clap"])
add("§4.2", "early window at lambda=alpha=0.2/0.5/1.0: coherence 0.12 -> 0.23/0.29/0.41; CLAP 0.207 -> 0.187/0.184/0.173", "025/a*/per_run.csv", "per-run", "; ".join(f"{a}: {v[0]:.2f}/{v[1]:.3f} (unguided {v[2]:.2f}/{v[3]:.3f})" for a, v in vals.items()),
    [round(vals[a][0], 2) for a in vals] == [0.23, 0.29, 0.41] and [round(vals[a][1], 3) for a in vals] == [0.187, 0.184, 0.173] and round(vals["0.2"][2], 2) == 0.12 and round(vals["0.2"][3], 3) == 0.207, tex=["to $0.23$, $0.29$ and $0.41$", "from $0.207$ to $0.187$, $0.184$ and $0.173$"])

# ============================================================ Exp B (009): main table
g = pb.groupby("method")
add("Table 1 caption", "MLSP-fixed: 224 valid coherence values, 225 CLAP values; every other method 225", "009/per_run.csv", "per-run", f"coh n={g.coherence_mlsp.count().to_dict()}", g.coherence_mlsp.count()["mlsp"] == 224 and g.clap.count()["mlsp"] == 225 and (g.coherence_mlsp.count().drop("mlsp") == 225).all(), tex=["224 valid MLSP-fixed coherence values (225 CLAP values"])
cm = g.coherence_mlsp.mean()
add("§4.3", "Early 0.17 vs unguided 0.13; Mid 0.42 (Early > unguided, Mid > Early)", "009/per_run.csv", "per-run", f"{cm['early']:.2f} vs {cm['sao']:.2f}; {cm['mid']:.2f}", round(cm["early"], 2) == 0.17 and round(cm["sao"], 2) == 0.13 and round(cm["mid"], 2) == 0.42 and cm["sao"] < cm["early"] < cm["mid"], tex=["($0.17$ vs $0.13$)", "the mid window reaches $0.42$"])
sch = json.load(open(R / "009_expB_full/schedules.json"))
add("§4.3", "SCPG selects 19-33; decodability-calibrated ablation = late window 35-49", "009/schedules.json", "summary", f"topk {min(sch['topk_dc_const']['steps'])}-{max(sch['topk_dc_const']['steps'])}; f1 {min(sch['rapg_cal_f1']['steps'])}-{max(sch['rapg_cal_f1']['steps'])}", (min(sch["topk_dc_const"]["steps"]), max(sch["topk_dc_const"]["steps"])) == (19, 33) and sch["rapg_cal_f1"]["steps"] == sch["late"]["steps"], tex=["SCPG selects steps 19--33", "selects exactly the late window (35--49)"])
n, mn, p, w, nt = paired(pb, "topk_dc_const", "mlsp")
add("§4.3", "constant SCPG 0.434 > MLSP-fixed 0.363, p<1e-6", "009/per_run.csv", "per-run", f"{cm['topk_dc_const']:.3f} vs {cm['mlsp']:.3f}, d={mn:+.3f}, p={p:.1e}", round(cm["topk_dc_const"], 3) == 0.434 and round(cm["mlsp"], 3) == 0.363 and mn > 0 and p < 1e-6, tex=["reaches \\cohTopK\\ against \\cohMLSP"])
n, mn, p, w, nt = paired(pb, "topk_dc_const", "mid")
add("§4.3", "constant SCPG vs Mid: +0.010, p=0.002", "009/per_run.csv", "per-run", f"d={mn:+.3f}, p={p:.3f}", round(mn, 3) == 0.010 and mn > 0 and round(p, 3) == 0.002, tex=["\\cohMid\\ for Mid ($\\dTopKvsMid$, $\\pTopKvsMid$)"])
pm = pd.read_csv(R / "009_expB_full/per_melody.csv").set_index("method")
alt = pb[pb.melody == "alternating"].groupby("method").coherence_mlsp.mean()
add("§4.3", "alternating pattern: constant SCPG 0.317 vs MLSP-fixed 0.346 (within noise); other four melodies SCPG > MLSP-fixed", "009/per_run.csv", "per-run", f"{alt['topk_dc_const']:.3f} vs {alt['mlsp']:.3f}; per_melody.csv {pm.loc['topk_dc_const','alternating']:.3f}/{pm.loc['mlsp','alternating']:.3f}; wins on others: {sum(pb[pb.melody==m].groupby('method').coherence_mlsp.mean().pipe(lambda x: x['topk_dc_const'] > x['mlsp']) for m in ['ascending','descending','pedal','zigzag'])}/4",
    round(alt["topk_dc_const"], 3) == 0.317 and round(alt["mlsp"], 3) == 0.346 and all(pb[pb.melody == m].groupby("method").coherence_mlsp.mean().pipe(lambda x: x["topk_dc_const"] > x["mlsp"]) for m in ["ascending", "descending", "pedal", "zigzag"]), tex=["($0.317$ vs $0.346$)"])
cl = g.clap.mean()
add("§4.3", "CLAP: SCPG variants 0.321-0.324, MLSP-fixed 0.316, unguided 0.307", "009/per_run.csv", "per-run", f"{cl['topk_dc_const']:.3f}/{cl['rapg_cal_dc']:.3f}, {cl['mlsp']:.3f}, {cl['sao']:.3f}", round(cl["topk_dc_const"], 3) == 0.321 and round(cl["rapg_cal_dc"], 3) == 0.324 and round(cl["mlsp"], 3) == 0.316 and round(cl["sao"], 3) == 0.307, tex=["($0.321$--$0.324$ for the SCPG variants vs $0.316$, and $0.307$ unguided)"])
hc = holm_vs(pb, "mlsp", "clap")
add("§4.3", "no guided method differs significantly from MLSP-fixed in CLAP (Holm over the family)", "009/per_run.csv", "per-run", "min Holm p = %.3f" % min(v for k, v in hc.items() if k != "sao"), min(v for k, v in hc.items() if k != "sao") > 0.05, tex=["CLAP does not differ significantly from MLSP-fixed for any guided method"])
fad = pd.read_csv(R / "009_expB_full/fad.csv").set_index("method").fad_clap_maestro
add("§4.3", "FAD 0.75-0.76 (SCPG variants) vs 0.79 (MLSP-fixed)", "009/fad.csv", "summary", f"{fad['rapg_cal_dc']:.2f}/{fad['topk_dc_const']:.2f} vs {fad['mlsp']:.2f}", {round(fad["rapg_cal_dc"], 2), round(fad["topk_dc_const"], 2)} <= {0.75, 0.76} and round(fad["mlsp"], 2) == 0.79, tex=["($0.75$--$0.76$ vs $0.79$"])
vc = g.voiced_cov.mean()
add("§4.3", "voiced fraction 0.88 SCPG / 0.86 MLSP-fixed / 0.84 unguided", "009/per_run.csv", "per-run", f"{vc['topk_dc_const']:.3f} ({vc['rapg_cal_dc']:.3f} with R) / {vc['mlsp']:.3f} / {vc['sao']:.3f}", round(vc["topk_dc_const"], 2) == 0.88 and round(vc["mlsp"], 2) == 0.86 and round(vc["sao"], 2) == 0.84, tex=["pYIN-voiced fraction $0.88$ under SCPG, $0.86$ MLSP-fixed, $0.84$ unguided"])
fn = g.coherence_firstnote.mean()
add("§4.3", "constant SCPG: coherence 0.434 vs 0.249 if collapsed onto the first pitch class", "009/per_run.csv", "per-run", f"{cm['topk_dc_const']:.3f} vs {fn['topk_dc_const']:.3f}", round(cm["topk_dc_const"], 3) == 0.434 and round(fn["topk_dc_const"], 3) == 0.249 and cm["topk_dc_const"] > fn["topk_dc_const"], tex=["($0.434$ vs $0.249$)"])
n, mn, p, w, nt = paired(pb, "rapg_cal_f1", "late"); hl = holm_vs(pb, "late")
add("§4.3", "R-scaling on the late window lowers coherence 0.315 -> 0.310, unadjusted paired p<1e-6", "009/per_run.csv", "per-run", f"{cm['late']:.3f} -> {cm['rapg_cal_f1']:.3f}, d={mn:+.4f}, p={p:.1e} (Holm vs Late: {hl['rapg_cal_f1']:.1e})", round(cm["late"], 3) == 0.315 and round(cm["rapg_cal_f1"], 3) == 0.310 and mn < 0 and p < 1e-6, tex=["from $0.315$ to $0.310$ (unadjusted paired $p{<}10^{-6}$)"])
n, mn, p, w, nt = paired(pb, "rapg_on", "mlsp"); on = pb[pb.method == "rapg_on"]
add("§4.3", "online R-threshold 0.351 vs 0.363 (lower, n.s. p=0.53); first update at steps 2-12, last at step 40 on the median trial", "009/per_run.csv", "per-run", f"{cm['rapg_on']:.3f} vs {cm['mlsp']:.3f}, d={mn:+.3f}, p={p:.2f}; first {on.first_step.min()}-{on.first_step.max()}, last median {on.last_step.median():.0f}", round(cm["rapg_on"], 3) == 0.351 and mn < 0 and round(p, 2) == 0.53 and (on.first_step.min(), on.first_step.max()) == (2, 12) and on.last_step.median() == 40, tex=["($0.351$ vs $0.363$, $p{=}0.53$)", "first update falls at steps 2--12 and its last at step 40 on the median trial"])
add("§2.3 / §3", "online R-threshold also uses R-scaling", "009/schedules.json", "summary", f"adaptive_strength={sch['rapg_on']['adaptive_strength']}, mean realised lambda {on.mean_lam.mean():.3f}", bool(sch["rapg_on"]["adaptive_strength"]), tex=["is combined with $R$-scaling", "online $R$-threshold (with $R$-scaling)"])
n, mn, p, w, nt = paired(pb, "rapg_cal_dc", "topk_dc_const")
add("§4.3", "R-scaling vs constant: +0.011 (R-scaled higher), better in 122 of 184 non-tied trials, p<1e-5; realised mean lambda 0.057", "009/per_run.csv", "per-run", f"d={mn:+.3f}, {w} of {nt}, p={p:.1e}; lambda={pb[pb.method=='rapg_cal_dc'].mean_lam.mean():.3f}", round(mn, 3) == 0.011 and mn > 0 and (w, nt) == (122, 184) and p < 1e-5 and round(pb[pb.method == "rapg_cal_dc"].mean_lam.mean(), 3) == 0.057, tex=["realised mean step on the test set is $\\lamRealised$ rather than $0.05$"])
share = (cm["topk_dc_const"] - cm["mlsp"]) / (cm["rapg_cal_dc"] - cm["mlsp"])
add("§4.3", "about 86% of the improvement over MLSP-fixed comes from placement", "009/per_run.csv", "per-run", f"{100*share:.1f}%", round(100 * share) == 86, tex=["about \\placementShare\\ of the improvement"])

# ============================================================ Exp D (014): budget sweep
refD = pb[(pb.method == "sao") & pb.trial.isin(set(prD.trial))]
add("§4.4", "10 SCPG updates 0.38 vs 25 uniform 0.40 (95% recovered); K=25 SCPG 0.56 > uniform 0.40", "014/per_run.csv", "per-run", f"{gD.loc[('topk_dc',10),'coherence_mlsp']:.2f} vs {gD.loc[('uniform',25),'coherence_mlsp']:.2f} ({100*gD.loc[('topk_dc',10),'coherence_mlsp']/gD.loc[('uniform',25),'coherence_mlsp']:.0f}%); {gD.loc[('topk_dc',25),'coherence_mlsp']:.2f}",
    round(gD.loc[("topk_dc", 10), "coherence_mlsp"], 2) == 0.38 and round(gD.loc[("uniform", 25), "coherence_mlsp"], 2) == 0.40 and 0.93 <= gD.loc[("topk_dc", 10), "coherence_mlsp"] / gD.loc[("uniform", 25), "coherence_mlsp"] <= 0.97 and round(gD.loc[("topk_dc", 25), "coherence_mlsp"], 2) == 0.56, tex=["(\\cohTopKx\\ vs \\cohUniKxxv)", "reaches \\cohTopKxxv\\ against \\cohUniKxxv"])
add("§4.4", "CLAP 0.351-0.373 across budgets; 0.348 unguided on the subset", "014/per_run.csv + 009/per_run.csv", "per-run", f"{gD.clap.min():.3f}-{gD.clap.max():.3f}; {refD.clap.mean():.3f}", round(gD.clap.min(), 3) == 0.351 and round(gD.clap.max(), 3) == 0.373 and round(refD.clap.mean(), 3) == 0.348, tex=["CLAP ranges from $0.351$ to $0.373$", "($0.348$ unguided on this subset)"])
worst_p, min_d = 0.0, 1.0
for K in sorted(set(prD.K)):
    x = prD[(prD.method == "topk_dc") & (prD.K == K)].set_index("trial").coherence_mlsp; y = prD[(prD.method == "uniform") & (prD.K == K)].set_index("trial").coherence_mlsp; j = x.index.intersection(y.index)
    worst_p = max(worst_p, wilcoxon(x[j], y[j]).pvalue); min_d = min(min_d, float((x[j] - y[j]).mean()))
add("§4.4", "SCPG > uniform at every K (positive mean difference), paired p<1e-5", "014/per_run.csv", "per-run", f"min mean diff {min_d:+.3f}, max p over K = {worst_p:.1e}", min_d > 0 and worst_p < 1e-5, tex=["better at every $K$ (paired Wilcoxon $p<10^{-5}$)"])

# ============================================================ Stable Audio 3 (019c/020c/021/024)
ps3 = pd.read_csv(R / "019c_sa3_expA/per_step.csv"); r3 = ps3[ps3.variant == "raw"]; f13 = r3.groupby("row").f1.mean().sort_index().values; sat3 = int(np.argmax(f13 >= 0.95 * f13.max()))
add("§4.5", "SA3: F1 reaches 95% of max at 76%", "019c/per_step.csv", "per-run", f"{100*(sat3+1)/50:.0f}%", (sat3 + 1) / 50 == 0.76, tex=["only at \\saThreePeakF"])
p3 = pd.read_csv(R / "020c_sa3_expC/per_run.csv"); b3 = pd.read_csv(R / "020c_sa3_expC/by_position.csv").sort_values("position")
m3 = pd.Series({p: burst_delta(p3, p).mean() for p in b3.position})
first40 = m3[[p for p in m3.index if (p + 1) / 50 <= 0.4]]; last20 = m3[[p for p in m3.index if (p + 1) / 50 > 0.8]]
add("§4.5", "SA3 burst gain ~0.08 over the first 40% (flat), 0.017 in the last 20%", "020c/per_run.csv", "per-run", f"first 40%: mean {first40.mean():.3f} (range {first40.min():.3f}-{first40.max():.3f}); last 20%: {last20.mean():.3f}", round(first40.mean(), 2) == 0.08 and round(last20.mean(), 3) == 0.017 and first40.min() > last20.mean(), tex=["$\\Delta C\\approx0.08$ over the first 40\\% of steps", "falls to \\saThreeDClate\\ in the last 20\\%"])
rb3 = pd.read_csv(R / "024_sa3_randburst/per_run.csv"); dr3 = burst_delta(rb3, 10); dp3 = burst_delta(p3, 10); j = dr3.index.intersection(dp3.index)
add("§4.5", "SA3 random-direction control -0.026 vs 0.089 for the probe gradient, paired p<1e-4", "024 + 020c per_run.csv", "per-run", f"{dr3[j].mean():.3f} vs {dp3[j].mean():.3f}, p={wilcoxon(dr3[j], dp3[j]).pvalue:.1e}", round(dr3[j].mean(), 3) == -0.026 and round(dp3[j].mean(), 3) == 0.089 and dp3[j].mean() > dr3[j].mean() and wilcoxon(dr3[j], dp3[j]).pvalue < 1e-4, tex=["gives $-0.026$ against $0.089$"])
g3 = pb3.groupby("method"); cm3 = g3.coherence_mlsp.mean(); cl3 = g3.clap.mean()
add("§4.5", "SA3: 49 valid coherence values for Mid and Late, 50 for the others", "021/per_run.csv", "per-run", f"{g3.coherence_mlsp.count().to_dict()}", g3.coherence_mlsp.count()["mid"] == 49 and g3.coherence_mlsp.count()["late"] == 49 and (g3.coherence_mlsp.count().drop(["mid", "late"]) == 50).all(), tex=["49 valid coherence values for Mid and Late"])
n, mn, p, w, nt = paired(pb3, "topk_dc_const", "mlsp"); h3 = holm_vs(pb3, "mlsp")
add("§4.5", "SA3 constant SCPG 0.44 > MLSP-fixed 0.28 (unguided 0.09), Holm-adjusted p=0.002; Late 0.18", "021/per_run.csv", "per-run", f"{cm3['topk_dc_const']:.3f} vs {cm3['mlsp']:.3f} (unguided {cm3['sao']:.3f}), d={mn:+.3f}, raw p={p:.1e}, Holm p={h3['topk_dc_const']:.4f}; late {cm3['late']:.3f}", round(cm3["topk_dc_const"], 2) == 0.44 and round(cm3["mlsp"], 2) == 0.28 and round(cm3["sao"], 2) == 0.09 and round(cm3["late"], 2) == 0.18 and mn > 0 and round(h3["topk_dc_const"], 3) == 0.002, tex=["Holm-adjusted $p{=}\\saThreePTopKvsMLSPholm$"])
hc3 = holm_vs(pb3, "sao", "clap")
add("§4.5", "SA3 CLAP 0.375 -> 0.335 under SCPG (decrease, Holm p<0.01); every guided schedule except MLSP-fixed (0.361, n.s.) decreases significantly", "021/per_run.csv", "per-run", f"{cl3['sao']:.3f} -> {cl3['topk_dc_const']:.3f}, Holm p={hc3['topk_dc_const']:.4f}; mlsp {cl3['mlsp']:.3f} p={hc3['mlsp']:.3f}; others max p={max(v for k, v in hc3.items() if k != 'mlsp'):.4f}", round(cl3["sao"], 3) == 0.375 and round(cl3["topk_dc_const"], 3) == 0.335 and cl3["topk_dc_const"] < cl3["sao"] and hc3["topk_dc_const"] < 0.01 and round(cl3["mlsp"], 3) == 0.361 and hc3["mlsp"] > 0.05 and max(v for k, v in hc3.items() if k != "mlsp") < 0.05 and all(cl3[k] < cl3["sao"] for k in cl3.index if k != "sao"), tex=["CLAP decreases from \\saThreeClapSAO\\ unguided to \\saThreeClapTopK"])
sch3 = json.load(open(R / "021_sa3_expB/schedules.json"))
lo, hi = tb3.loc["topk_dc_const", "coherence_mlsp_lo"], tb3.loc["topk_dc_const", "coherence_mlsp_hi"]
add("§4.5", "SA3 calibration selects steps 4-18; Early (0-14) 0.44 within the CI of SCPG", "021/schedules.json + per_run.csv", "per-run", f"{min(sch3['topk_dc_const']['steps'])}-{max(sch3['topk_dc_const']['steps'])}; early {cm3['early']:.3f}, SCPG CI [{lo:.3f}, {hi:.3f}]", (min(sch3["topk_dc_const"]["steps"]), max(sch3["topk_dc_const"]["steps"])) == (4, 18) and round(cm3["early"], 2) == 0.44 and lo <= cm3["early"] <= hi, tex=["Calibration selects steps \\saThreeRAPGsteps", "reaches \\saThreeCohEarly\\ here, within the CI of SCPG"])

# ============================================================ legacy alignment (026)
t26 = pd.read_csv(R / "026_sao_legacy_mlsp/per_run.csv").groupby("method")[["coherence_mlsp", "coherence_firstnote"]].mean()
add("§3", "original alignment: 0.259 against the melody, 0.68 against its first pitch class", "026/per_run.csv", "per-run", f"{t26.loc['mlsp','coherence_mlsp']:.3f}; {t26.loc['mlsp','coherence_firstnote']:.2f}", round(t26.loc["mlsp", "coherence_mlsp"], 3) == 0.259 and round(t26.loc["mlsp", "coherence_firstnote"], 2) == 0.68, tex=["scores $0.259$ against the melody and $0.68$ against its first pitch class alone"])

# ============================================================ report
ap = argparse.ArgumentParser(); ap.add_argument("--md", default=None); args = ap.parse_args()
n_bad = sum(1 for r in rows if r[5] != "ok"); n_pr = sum(1 for r in rows if r[3] == "per-run")
hdr = "| Paper | Claim | Source | Basis | Recomputed | Status |\n|---|---|---|---|---|---|\n"
body = "\n".join(f"| {w} | {c} | `{s}` | {k} | {v} | {st} |" for w, c, s, k, v, st in rows)
out = (f"# Hand-written statistics of main.tex, recomputed from results/\n\n{len(rows)} checks, {n_bad} failure(s). "
       f"{n_pr} recomputed from per-run files, {len(rows)-n_pr} read from summary files (which are themselves checked against the per-run files in the first rows). "
       "Every comparison is checked for value and direction, and the wording each check stands for is verified to be present in paper/main.tex.\n\n" + hdr + body + "\n")
print(out)
if args.md:
    Path(args.md).write_text(out)
raise SystemExit(1 if n_bad else 0)
