# Hand-written statistics of main.tex, recomputed from results/

47 checks, 0 failure(s). 42 recomputed from per-run files, 5 read from summary files (which are themselves checked against the per-run files in the first rows). Every comparison is checked for value and direction, and the wording each check stands for is verified to be present in paper/main.tex.

| Paper | Claim | Source | Basis | Recomputed | Status |
|---|---|---|---|---|---|
| Table 1 / §4.5 | 009_expB_full/table.csv means equal per_run.csv means (coherence, CLAP) | `009_expB_full` | per-run | max |diff| = 5.6e-17 | ok |
| Table 1 / §4.5 | 021_sa3_expB/table.csv means equal per_run.csv means (coherence, CLAP) | `021_sa3_expB` | per-run | max |diff| = 1.1e-16 | ok |
| Fig. 1(b) | 004/by_position.csv mean dC equals per-run paired means | `004_expC_burst` | per-run | max |diff| = 1.0e-16 | ok |
| Fig. 2 | 014/pareto.csv means equal per_run.csv means | `014_expD_budget_v2` | per-run | max |diff| = 5.6e-17 | ok |
| Fig. 1(a) | 003/summary.csv F1 curve equals the per-step mean over 50 runs | `003_expA_trajectories` | per-run | max |diff| = 6.9e-17 | ok |
| §4.1 / abstract | F1 reaches 95% of its maximum at 88% of the trajectory | `003/per_step.csv` | per-run | step 43 -> 88% | ok |
| §4.1 | Pearson(R_t, F1) across steps = 0.96 | `003/per_step.csv` | per-run | 0.959 | ok |
| §4.1 | median within-trajectory Pearson = 0.85 | `003/per_step.csv` | per-run | 0.846 | ok |
| §4.1 | fixed-step Spearman median 0.27; 0.57 over the last 15 steps | `003/per_step.csv` | per-run | 0.27; 0.57 | ok |
| §4.2 / abstract | peak at s=0.54 (54%), dC=0.104 | `004/per_run.csv` | per-run | position 26 -> s=0.54, dC=0.104 | ok |
| §4.2 | burst in the first 20% changes coherence by < 1/10 of the peak | `004/per_run.csv` | per-run | max early dC=0.0036 vs peak/10=0.0104 | ok |
| §4.2 | s=0.94 vs s=0.54: BCE change -0.14 vs -0.11; log-mel 0.21 vs 0.53 | `004/by_position.csv` | summary | -0.138 vs -0.109; 0.211 vs 0.532 | ok |
| §4.2 | mean dCLAP between -0.012 and -0.005 at every position (all negative) | `004/per_run.csv` | per-run | -0.0118 .. -0.0050 | ok |
| §4.2 | bootstrap argmax at s=0.54 in 92% of 5000 resamples, within 0.46-0.62 in all (re-drawn, seed 0) | `004/per_run.csv` | per-run | 91.8%; positions [22, 26, 30] | ok |
| §4.2 | the quoted 92% (stability.json) agrees with the re-drawn bootstrap | `004/stability.json` | summary | 92% vs 91.8% | ok |
| §4.2 | leave-one-prompt-out peaks between s=0.54 and 0.62 | `004/per_run.csv` | per-run | positions [26, 30] | ok |
| §4.2 | mid beats early in 88% of trials, p<1e-7 (mid > early) | `004/per_run.csv` | per-run | 88%, mean diff +0.084, p=1.6e-08 | ok |
| §4.2 | random-direction burst 0.003 (n.s.) vs 0.104 for the probe gradient, paired p<1e-6 | `023 + 004 per_run.csv` | per-run | 0.003 (p vs 0: 0.39) vs 0.104, p=4.9e-07 | ok |
| §4.2 | lambda=0.05 sweep: peak s=0.54, dC=0.099, Pearson 0.99 between the curves | `015/per_run.csv` | per-run | s=0.54, dC=0.099, r=0.990 | ok |
| §4.2 | early window at lambda=alpha=0.2/0.5/1.0: coherence 0.12 -> 0.23/0.29/0.41; CLAP 0.207 -> 0.187/0.184/0.173 | `025/a*/per_run.csv` | per-run | 0.2: 0.23/0.187 (unguided 0.12/0.207); 0.5: 0.29/0.184 (unguided 0.12/0.207); 1.0: 0.41/0.173 (unguided 0.12/0.207) | ok |
| Table 1 caption | MLSP-fixed: 224 valid coherence values, 225 CLAP values; every other method 225 | `009/per_run.csv` | per-run | coh n={'early': 225, 'late': 225, 'mid': 225, 'mlsp': 224, 'rapg_cal_dc': 225, 'rapg_cal_f1': 225, 'rapg_on': 225, 'sao': 225, 'topk_dc_const': 225, 'uniform': 225} | ok |
| §4.3 | Early 0.17 vs unguided 0.13; Mid 0.42 (Early > unguided, Mid > Early) | `009/per_run.csv` | per-run | 0.17 vs 0.13; 0.42 | ok |
| §4.3 | SCPG selects 19-33; decodability-calibrated ablation = late window 35-49 | `009/schedules.json` | summary | topk 19-33; f1 35-49 | ok |
| §4.3 | constant SCPG 0.434 > MLSP-fixed 0.363, p<1e-6 | `009/per_run.csv` | per-run | 0.434 vs 0.363, d=+0.072, p=9.8e-08 | ok |
| §4.3 | constant SCPG vs Mid: +0.010, p=0.002 | `009/per_run.csv` | per-run | d=+0.010, p=0.002 | ok |
| §4.3 | alternating pattern: constant SCPG 0.317 vs MLSP-fixed 0.346 (within noise); other four melodies SCPG > MLSP-fixed | `009/per_run.csv` | per-run | 0.317 vs 0.346; per_melody.csv 0.317/0.346; wins on others: 4/4 | ok |
| §4.3 | CLAP: SCPG variants 0.321-0.324, MLSP-fixed 0.316, unguided 0.307 | `009/per_run.csv` | per-run | 0.321/0.324, 0.316, 0.307 | ok |
| §4.3 | no guided method differs significantly from MLSP-fixed in CLAP (Holm over the family) | `009/per_run.csv` | per-run | min Holm p = 0.968 | ok |
| §4.3 | FAD 0.75-0.76 (SCPG variants) vs 0.79 (MLSP-fixed) | `009/fad.csv` | summary | 0.75/0.76 vs 0.79 | ok |
| §4.3 | voiced fraction 0.88 SCPG / 0.86 MLSP-fixed / 0.84 unguided | `009/per_run.csv` | per-run | 0.879 (0.883 with R) / 0.856 / 0.842 | ok |
| §4.3 | constant SCPG: coherence 0.434 vs 0.249 if collapsed onto the first pitch class | `009/per_run.csv` | per-run | 0.434 vs 0.249 | ok |
| §4.3 | R-scaling on the late window lowers coherence 0.315 -> 0.310, unadjusted paired p<1e-6 | `009/per_run.csv` | per-run | 0.315 -> 0.310, d=-0.0056, p=6.2e-07 (Holm vs Late: 2.5e-06) | ok |
| §4.3 | online R-threshold 0.351 vs 0.363 (lower, n.s. p=0.53); first update at steps 2-12, last at step 40 on the median trial | `009/per_run.csv` | per-run | 0.351 vs 0.363, d=-0.013, p=0.53; first 2-12, last median 40 | ok |
| §2.3 / §3 | online R-threshold also uses R-scaling | `009/schedules.json` | summary | adaptive_strength=True, mean realised lambda 0.044 | ok |
| §4.3 | R-scaling vs constant: +0.011 (R-scaled higher), better in 122 of 184 non-tied trials, p<1e-5; realised mean lambda 0.057 | `009/per_run.csv` | per-run | d=+0.011, 122 of 184, p=1.4e-06; lambda=0.057 | ok |
| §4.3 | about 86% of the improvement over MLSP-fixed comes from placement | `009/per_run.csv` | per-run | 86.1% | ok |
| §4.4 | 10 SCPG updates 0.38 vs 25 uniform 0.40 (95% recovered); K=25 SCPG 0.56 > uniform 0.40 | `014/per_run.csv` | per-run | 0.38 vs 0.40 (95%); 0.56 | ok |
| §4.4 | CLAP 0.351-0.373 across budgets; 0.348 unguided on the subset | `014/per_run.csv + 009/per_run.csv` | per-run | 0.351-0.373; 0.348 | ok |
| §4.4 | SCPG > uniform at every K (positive mean difference), paired p<1e-5 | `014/per_run.csv` | per-run | min mean diff +0.096, max p over K = 5.4e-06 | ok |
| §4.5 | SA3: F1 reaches 95% of max at 76% | `019c/per_step.csv` | per-run | 76% | ok |
| §4.5 | SA3 burst gain ~0.08 over the first 40% (flat), 0.017 in the last 20% | `020c/per_run.csv` | per-run | first 40%: mean 0.081 (range 0.074-0.089); last 20%: 0.017 | ok |
| §4.5 | SA3 random-direction control -0.026 vs 0.089 for the probe gradient, paired p<1e-4 | `024 + 020c per_run.csv` | per-run | -0.026 vs 0.089, p=2.7e-05 | ok |
| §4.5 | SA3: 49 valid coherence values for Mid and Late, 50 for the others | `021/per_run.csv` | per-run | {'early': 50, 'late': 49, 'mid': 49, 'mlsp': 50, 'rapg_cal_dc': 50, 'sao': 50, 'topk_dc_const': 50, 'uniform': 50} | ok |
| §4.5 | SA3 constant SCPG 0.44 > MLSP-fixed 0.28 (unguided 0.09), Holm-adjusted p=0.002; Late 0.18 | `021/per_run.csv` | per-run | 0.442 vs 0.277 (unguided 0.094), d=+0.165, raw p=6.6e-04, Holm p=0.0020; late 0.181 | ok |
| §4.5 | SA3 CLAP 0.375 -> 0.335 under SCPG (decrease, Holm p<0.01); every guided schedule except MLSP-fixed (0.361, n.s.) decreases significantly | `021/per_run.csv` | per-run | 0.375 -> 0.335, Holm p=0.0069; mlsp 0.361 p=0.086; others max p=0.0069 | ok |
| §4.5 | SA3 calibration selects steps 4-18; Early (0-14) 0.44 within the CI of SCPG | `021/schedules.json + per_run.csv` | per-run | 4-18; early 0.440, SCPG CI [0.369, 0.514] | ok |
| §3 | original alignment: 0.259 against the melody, 0.68 against its first pitch class | `026/per_run.csv` | per-run | 0.259; 0.68 | ok |
