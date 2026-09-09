# "When to Steer" — 评审简报（供 code-review / paper-review agent 使用）

**日期：** 2026-09-09  **投稿：** ICASSP 2027（截止 2026-09-16，不延期；4 页正文 + 第 5 页只放参考文献；非双盲）
**仓库：** https://github.com/StoneYeY/mic-pitch-class-steering ，分支 **`icassp`**（`main` 是 MLSP 2026 原始代码，不要评那个）
**论文：** `paper/main.tex`（编译：`cd paper && latexmk -pdf main.tex`；PDF 已在 `paper/main.pdf`）
**Demo：** `docs/index.html`（自包含听评/展示页），生成脚本 `demo/build_demo.py` + `demo/template.html`

## 0. 你要做什么

请以"挑剔的 ICASSP 审稿人 + 代码审核者"的身份审这份工作，产出一份意见清单，按严重程度排序（blocker / major / minor / nit），每条给出：文件 + 行号（或论文章节）、问题、建议的修改。重点顺序：

1. **结论是否被证据支持**（过度 claim、统计用法、对照是否公平）
2. **实现是否和论文描述一致**（公式、schedule、指标）——第 6 节给了数字↔文件对照表，请抽查
3. **可复现性**（缺什么就跑不出来）
4. 文字/排版（最后）

不需要重跑 GPU 实验（每个 method 225 次生成，约 12 分钟/method）；所有结果 csv 都在 `results/` 里，无 GPU 也能用 `w2s/scripts/make_figures.py` 从 csv 重新生成所有数字和图，验证论文数字。

## 1. 一句话

Stable Audio Open (SAO) 1.0 上，用一个 125k 参数的 latent 空间 pitch-class probe 做梯度引导来控制旋律。已有方法把引导放在人为选定的步区间（LatCH：前 20% 步；MLSP：后 60% 步 20,22,…,48）。我们测了两条曲线——**probe 的可解码性 F1(t)** 和 **单步干预的因果 steering 敏感度 ΔC(t)**——发现二者峰值不在同一处（可解码性在 88% 处饱和、敏感度峰值在 54%），据此提出 **RAPG**（Reliability-Adaptive Probe Guidance）：固定预算 K 次更新，按校准好的敏感度 profile 选步，按 probe 可靠度 R_t 缩放强度。同预算下 melodic coherence 从 MLSP 的 0.363 升到 0.445（p<1e-7），CLAP/FAD 不变。

## 2. 核心主张与证据

| # | 主张（论文位置） | 证据 | 数字 |
|---|---|---|---|
| C1 | 可解码性 ≠ 可操控性：F1(t) 单调升到末段，ΔC(t) 峰在中段（§4.1, §4.2, Fig.1） | Exp A：50 条 dev 轨迹上 probe F1 对 pYIN 转录标签；Exp C：50 dev trial × 12 位置，3 步 burst（λ=0.10） | F1 在 step 43 首次达到峰值的 95%（(43+1)/50 = "88%"，argmax 在 step 46）；ΔC 峰在 position 26（(26+1)/50 = "54%"，ΔC=0.104）；前 20% 的 burst ΔC≈0.004，< 峰值 1/10 |
| C2 | 目标无关的可靠度 R_t 跟 F1(t) 高度相关，使在线 schedule 可行（§4.1） | `results/003_expA_trajectories/corr.json` | Pearson 0.96（跨步均值）；单条轨迹内中位数 0.85 |
| C3 | 同预算 K=15，窗口位置决定一切（§4.3 Table 1） | Exp B：15 prompt × 5 melody × 3 seed = 225 配对样本/method，10 个 method | Early 0.166 / Mid 0.424 / Late 0.315 / Uniform 0.283 / MLSP 0.363 |
| C4 | RAPG-cal(ΔC) 最好，RAPG-cal(F1) 只等于 Late（§4.3） | 同上，配对 Wilcoxon + Holm | 0.445 vs 0.363，p=2.5e-8；RAPG-cal(F1)=0.310 ≈ Late 0.315 |
| C5 | 音质无代价（§4.3） | CLAP、FAD(CLAP emb, 600 MAESTRO clips) | CLAP 0.324 vs 0.316（vs SAO Holm p=0.014，"略升"）；FAD 0.75（最低）vs 0.79 |
| C6 | 自适应强度贡献小，主要是"放哪"（§4.3 末） | Top-K(ΔC) 常数 λ vs RAPG-cal(ΔC) | 0.434 vs 0.445 |
| C7 | 预算扫描：放对位置的 10 步 ≈ 均匀铺的 25 步；CLAP 不随 K 掉（§4.4 Fig.2） | Exp D：配对 50 trial（10 prompt × 5 melody × seed 0），K∈{5,10,15,25} | Top-K 0.26/0.38/0.48/0.56 vs Uniform 0.16/0.24/0.30/0.40，每个 K 的 p<1e-5 |

## 3. 方法（以代码为准）

**采样：** diffusers `StableAudioPipeline`，EDM-DPM-solver multistep，50 步，CFG 4.0，fp16，`audio_end_in_s=5.0`。注意 SAO 永远去噪 1024 帧 latent（21.53 帧/秒 = 47.55 s），再裁成 5 s 音频 → 只有前 **108 帧**可听。`w2s/generate.py::W2SGenerator.run`。

**引导更新（MLSP 原规则，未改）：** 在 legacy callback（`scheduler.step` 之后、下一步之前）里，对当前 latent z 计算 probe 的 BCE loss（只在可听区域 `region=[0,108)`），梯度 g，
`z ← z − clamp(λ_t · rms(z)/rms(g) · g, ±α·rms(z))`，α=0.05。`generate.py` L94–122。

**目标：** `make_target(melody, 5.0)`：8 个音、120 BPM、每音 0.5 s，按真实 21.53 fps 摆到 latent 帧上（MLSP 原代码用错 fps=204.8，把 4 s 旋律摊到 38 s latent，等于只推向第一个音；论文 §3 有一句披露，见第 7 节 Q1）。

**可靠度（`w2s/reliability.py`）：** probe 是 12 路独立 sigmoid（multi-label），所以先按帧归一化 q_f = p_f / Σ_c p_{f,c}，`R_t = 1 − mean_f H(q_f)/log 12`，在可听 108 帧上平均。target-free。

**Schedule（`w2s/schedules.py`）：**
- 固定窗：Early 0–14 / Mid 18–32 / Late 35–49 / Uniform 每 ~3.5 步 / MLSP 20,22,…,48（全部 K=15，λ=0.05）
- **RAPG-cal(w)：** 离线用 dev 集 profile w(t) 取 Top-K 步；w=F1(t) → 选到 35–49（与 Late 完全相同）；w=ΔC(t)（Exp C 的 `by_position.csv` 插值+平滑，`interpolate_profile(smooth=3)`）→ 选到 **19–33**。强度 `λ_t = λ·R_t / r̄`，r̄ = dev 集上这些步的平均 R_t（ΔC 版 r̄=0.301，F1 版 0.456；`results/009_expB_full/schedules.json`）
- **Top-K(ΔC) const：** 同 19–33，常数 λ（分离"放哪"和"多强"）
- **RAPG-on：** 在线，R_t ≥ η 且预算未用完就引导；η 用 dev 集校准使平均更新数 ≈ K（η=0.310，r̄=0.409）

**指标（`w2s/evalclip.py`, `w2s/metrics.py`）：** melodic coherence = MLSP 原函数（pYIN @ 86 fps，目标活跃区内 voiced 帧的 pitch-class 命中率）；chroma cosine（CQT chroma vs 二值目标）；CLAP = `laion/clap-htsat-unfused` 的 text–audio cosine（transformers 5.7 下 `larger_clap_music` 权重塌陷成 NaN，故换；论文已改为写明 checkpoint）；FAD = 各 method 225 条音频 CLAP audio embedding vs 600 条 MAESTRO 参考 clip。统计：bootstrap 95% CI，配对 Wilcoxon，Holm 校正（`w2s/scripts/aggregate_expB.py`）。

## 4. 实验一览

| Exp | 脚本 / job | 数据 | 输出 |
|---|---|---|---|
| A 轨迹可靠度 | `w2s/scripts/expA_trajectories.py`, `jobs/queue/003_*` | 5 dev prompt × 10 seed = 50 条无引导轨迹，每步存 latent；probe 预测 vs 最终音频 pYIN 转录标签 | `results/003_expA_trajectories/summary.csv`（每步 F1、R_t，三种 variant：raw / rmsnorm / gauss_proxy）、`corr.json` |
| C 单步 burst | `expC_burst.py`, `004_*`（CLAP 由 `010_*` 补算） | 50 dev trial（5 prompt × 5 melody × 2 seed）× 12 位置 {2,6,…,46}，每位置引导 3 步 λ=0.10，与同 seed 无引导配对 | `results/004_expC_burst/by_position.csv`（ΔC、Δchroma、ΔCLAP、Δbce0、CI）、`summary.json` |
| B 主表 | `expB_windows.py` + `aggregate_expB.py`, `008_*`(pilot) `009_*`(full) | test：15 prompt × 5 melody × 3 seed，10 method | `results/009_expB_full/per_run.csv`（2250 行）、`table.csv`、`fad.csv`、`schedules.json`、`per_melody.csv` |
| D 预算扫描 | `expD_budget.py`, `014_*`（`011_*` 是作废的 v1，r̄ 写死 1.0 导致 RAPG 强度偏弱 3×，**不要用**） | 10 test prompt × 5 melody × seed 0，Uniform vs Top-K(ΔC)-const，K∈{5,10,15,25} | `results/014_expD_budget_v2/per_run.csv`、`pareto.csv` |

prompt/melody 定义在 `w2s/data.py`：5 个 dev prompt（只用于 Exp A/C 和校准）、15 个 test prompt（MLSP 的 9 个 + 6 个新的，全部含钢琴）、5 条旋律（ascending / descending / alternating / pedal / zigzag，8 音，120 BPM，音高 MIDI 53–68）。

## 5. 代码地图

```
w2s/
  data.py          prompts, MELODIES, pitch_class_target(), trial_grid()
  reliability.py   R_t（entropy / margin / maxprob）
  schedules.py     GuidanceSchedule, make_schedule(kind=...), topk_steps, interpolate_profile, calibrate_eta
  generate.py      W2SGenerator：callback 内做 probe 引导；RunResult(log per step: R, sigma, guided, lam, loss)
  mlsp_bridge.py   加载 SAO pipeline 与 MLSP probe checkpoint（checkpoints/probe_best.pt，不在 git 里）
  metrics.py       ClapScorer(_emb_tensor 取 pooler_output), frechet_distance, chroma
  evalclip.py      ClipEvaluator.evaluate(audio) → coherence_mlsp / coherence_w2s / chroma_cos / clap / voiced_cov …
  probe_eval.py    Exp A 的 probe-vs-transcription F1
  scripts/         expA_trajectories, expC_burst, expB_windows, aggregate_expB, expD_budget, make_figures, collect_demo, …
jobs/queue/*.sh    每个实验的入口（环境变量在脚本里）；jobs/done/*.log 是 GPU 上的原始日志
runner/            GPU 侧 git 轮询执行器（runner.sh）+ Mac 侧自动 push（mac_git_sync.sh）
paper/             main.tex, numbers.tex / numbers_budget.tex / table_main.tex（均由 make_figures.py 生成，不要手改）, fig_*.pdf
```

数据流：`jobs/queue/NNN.sh` → 脚本写 `results/NNN_*/…csv` → `make_figures.py --expA … --expB … --expD … --out paper/` → `numbers*.tex` + `table_main.tex` + 图 → `main.tex` 用 `\input` 引用宏（如 `\cohRAPG`）。

无 GPU 复现论文数字：
```
python w2s/scripts/make_figures.py --expA results/003_expA_trajectories --expC results/004_expC_burst \
       --expB results/009_expB_full --expD results/014_expD_budget_v2 --out paper/
cd paper && latexmk -pdf main.tex
```

## 6. 论文数字 ↔ 来源对照（请抽查）

| 论文里的数字 | 来源 |
|---|---|
| 0.126 / 0.363 / 0.445 / 0.434 等 Table 1 全部 | `results/009_expB_full/table.csv` 列 `coherence_mlsp, chroma_cos, clap, fad_clap_maestro, n_updates`；CI 列 `_lo/_hi`；`p_coherence_mlsp_vs_mlsp_holm` 决定 † |
| "p<10^-7"（RAPG-cal(ΔC) vs MLSP） | 同上 `p_coherence_mlsp_vs_mlsp` = 2.5e-8 |
| CLAP "if anything, slightly higher"（论文未给 p；进展日志里的 p=0.014） | `p_clap_vs_sao_holm`(rapg_cal_dc) = 0.0143（是 vs SAO；vs MLSP 不显著） |
| 0.445 vs 0.255、0.363 vs 0.263（"first-note" 对照） | `per_run.csv` 按 method 平均 `coherence_mlsp` / `coherence_firstnote` |
| 分旋律 0.372 vs 0.294（descending）、0.725 vs 0.607（pedal）、0.327 vs 0.346（alternating） | `results/009_expB_full/per_melody.csv` |
| RAPG-on 0.351, p=0.53 | `table.csv` 行 rapg_on |
| Pearson 0.96；单轨迹中位数 0.85 | `results/003_expA_trajectories/corr.json` → `corr.raw["corr(R_entropy,f1)"]`=0.959, `per_run_corr_Rentropy_f1.median`=0.846 |
| "88%"（F1 饱和位置）、"54%"（ΔC 峰） | `numbers.tex` 的 `\peakF`, `\peakDC`；计算在 `make_figures.py::table_and_numbers` L167–174：peakF = (首个 F1 ≥ 0.95·max 的步 43 + 1)/50；peakDC = (`summary.json.argmax_dcoh_pos`=26 + 1)/50 |
| Fig.2 与 §4.4 的 0.16…0.56 | `paper/budget_summary.csv`（由 `results/014_expD_budget_v2/per_run.csv` bootstrap 得到）；配对 p 值需自算（脚本未存），我算的是 5e-6 / 2e-6 / 7e-9 / 7e-8 |
| 3.5× | 0.445 / 0.126 |

## 7. 已知问题与我自己的疑虑（请重点审）

**Q1 MLSP bug 的披露方式。** §3 "Base model and probe" 段说明了原公开代码把目标摊到整段 latent、导致只推向第一个音，并说 MLSP-fixed 基线是修正后的实现。这是同一批作者的前作。措辞是否够清楚又不自伤？是否应该改成脚注/勘误？这是作者决策项，但请评估现在这段话审稿人会怎么读。

**Q2 RAPG-cal(F1) 与 Late 的步完全相同（35–49）。** 论文说它"只匹配 Late 基线"——这是设计使然（F1 单调升，Top-15 必然是最后 15 步）。它其实只测了"自适应强度在后窗有没有用"（0.310 vs 0.315，没用）。表述是否把这点说透了？

**Q3 RAPG-online 表现平平（0.351 < MLSP 0.363，n.s.）。** 因为 R_t 单调升，阈值 η 选中的基本是后段步，本质上是可解码性准则，而论文自己的论点就是可解码性不是该看的信号。§4.3 把它写成"没有 dev 集时的首选"，这个定位站得住吗？是否应更直白地承认在线变体继承了 F1 准则的缺陷、并把它当作 negative result？

**Q4 题目叫 Reliability-*Adaptive*，但最好的方法里"自适应"贡献只有 0.445 vs 0.434。** 论文在 §4.3 末承认了。题目/摘要是否过度强调了 adaptive？是否应把主贡献更明确地写成"敏感度校准的 schedule"？

**Q5 Exp C 的 burst 用 λ=0.10 × 3 步，主实验用 λ=0.05 × 15 步。** 敏感度 profile ΔC(t) 是在不同强度下测的，再拿去选主实验的步。合理性论证是否足够？（论文里没有讨论强度不匹配。）

**Q6 Exp D 只有 seed 0、50 个配对 trial、只比两种 schedule。** 结论"10 步 ≈ 25 步"用的是 0.378 vs 0.400（Top-K K=10 vs Uniform K=25），CI 有重叠。措辞"are worth as much as"是否过强？

**Q7 FAD 没有 CI，也没有显著性检验**，且每个 method 只有 225 条生成音频 vs 600 参考，FAD 在小样本下有偏（但各 method n 相同，可比）。论文说 RAPG-cal(ΔC) 的 FAD "最低"。要不要加 bootstrap CI 或弱化措辞？

**Q8 coherence 只在 voiced 帧上算**（MLSP 定义）。理论上把音频变得 unvoiced 可以"作弊"。我核了 `voiced_cov`：rapg_cal_dc 0.883、mid 0.878、mlsp 0.856、sao 0.842、late 0.780——中段引导反而更 voiced，所以不是作弊；但论文没提这个检查。要不要加一句？

**Q9 生成音频最后约 1 s 是静音**（SAO 在 `seconds_total=5` 下的固有行为，各条件一致；旋律只占 0–4 s，不影响指标）。论文说 "5-second outputs"，是否需要说明？

**Q10 Callback 时序语义。** diffusers legacy callback 在 `scheduler.step` 之后触发，所以"在 step t 引导"= 修改的是 step t 输出、作用于 step t+1 的输入；R_t 也是在 step t 的输出 latent 上算的。Fig.1 横轴 s=t/N 与 Exp C 的位置编号沿用了这个约定。请确认论文/图注中没有 off-by-one 的表述冲突。

**Q11 CLAP 的 "vs SAO" p 值。** 正文"if anything, slightly higher"引用的 p=0.014 是 vs SAO（Holm），vs MLSP 不显著。措辞是否会让读者误以为 vs MLSP 显著？

**Q12 Table 1 caption** 说 CI 半宽 ≤0.035（coherence）/ ≤0.017（CLAP），这是我从 `table.csv` 算的最大值；请核对。

**Q13 论文里的 demo 链接** `https://stoneyey.github.io/mic-pitch-class-steering/` 需要作者在 GitHub Settings→Pages 里启用（branch `icassp`, folder `/docs`），目前未启用。

**Q14 作者信息**：`\name` 里三位作者标为 equal contribution，邮箱在 `\address`——作者自己确认。camera-ready 时取消注释 `\copyrightnotice`。

**Q15 Reference 检查**：`paper/refs.bib` 20 条，其中 LatCH（Novack et al., arXiv 2603.04366）和 MLSP（Ye et al., MLSP 2026）是关键引用，请核对条目字段是否齐全、年份是否正确。

## 8. 已经修过、不用再报的

- MLSP 目标时间轴 bug（fps 204.8 → 21.53），见 Q1。
- Exp D v1（`011_*`）r̄ 写死 1.0 → 作废，v2 改为 Uniform vs Top-K(ΔC)-const。
- CLAP：`larger_clap_music` 在 transformers 5.7 下输出 NaN → 换 `clap-htsat-unfused`，并显式取 `pooler_output`；论文措辞已从"music checkpoint"改为写明 checkpoint。
- 表格 overfull → 去掉表内 CI，改为 caption 里给半宽上限。
- 正文曾引用不存在的 `coh₁` 列 → 改为在正文给数字。

## 9. 输出格式

```
[BLOCKER|MAJOR|MINOR|NIT] <文件:行 或 论文§x.y> — 问题一句话
  证据/推理：…
  建议：…
```
最后附一段总体判断：以 ICASSP 标准，这篇现在是 accept / borderline / reject，最能提高胜算的 3 个改动是什么。
