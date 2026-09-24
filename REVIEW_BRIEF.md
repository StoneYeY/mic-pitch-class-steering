# "When to Steer" — 评审简报（供 code-review / paper-review agent 使用）

**日期：** 2026-09-09（首版）→ 2026-09-23（v9，见 §10–§16）  **投稿：** ICASSP 2027（官方截止 **2026-09-23 23:59:59 AoE**；4 页正文 + 第 5 页只放参考文献/伦理声明；非双盲；每位作者需 ORCiD；任何章节不得完全由 LLM 生成）
**仓库：** https://github.com/StoneYeY/mic-pitch-class-steering ，分支 **`icassp`**（`main` 是 MLSP 2026 原始代码，不要评那个）
**论文：** `paper/main.tex`（编译：`cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`；最终 PDF `paper/When_to_Steer_ICASSP2027_final.pdf`，5 页 = 正文 4 + 参考文献/伦理 1）
**Demo：** https://stoneyey.github.io/mic-pitch-class-steering/ （结果 + 带标签音频；`docs/index.html` 自包含），生成脚本 `demo/build_demo.py` + `demo/template.html`

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

## 10. 评审回应（v2，2026-09-16）

外部评审结论：Borderline / Weak Accept；主要意见与处理如下（全部已进 `paper/main.tex`，PDF `paper/When_to_Steer_ICASSP2027_final.pdf`）。

| 评审意见 | 处理 |
|---|---|
| [MAJOR] 主贡献命名错位：收益主要来自 sensitivity-based placement，不是 reliability adaptation | 摘要、贡献列表、§4.3、§5 全部重写为 "decodability ≠ steerability → 按敏感度放置预算 → reliability 缩放是次要增益"；§4.3 明确写出"约 85% 的提升来自放置"。**题目未改**（保留 RAPG），作者可选改为 *Sensitivity-Calibrated Probe Guidance* |
| [MAJOR] Exp C 用 λ=0.10，主实验用 λ=0.05 | **新跑 job 015**（`results/015_expC_lam05/`，同 12 位置 × 50 dev trial，λ=0.05）：峰仍在 position 26（s=0.54），ΔC 0.099 vs 0.104，两条曲线 Pearson 0.99，Top-15 窗从 19–33 变 18–32。§4.2 加一句 |
| [MAJOR] "10 步 ≈ 25 步" 过强 | 改为 "recover 95% of the coherence of 25 uniform updates (0.38 vs 0.40)"；§4.4 与 §5 同步；加 "one seed, efficiency analysis" 限定 |
| [MAJOR] dev/test 分离要写明 | §3 首句改为 "All … profiles, schedule choices and thresholds are calibrated exclusively on five development prompts; the evaluation prompts are held out and never used for schedule selection"；Table 1 caption 加 "schedules are calibrated on the disjoint development set" |
| [MAJOR] ΔC 峰值稳定性 | 新脚本 `w2s/scripts/expC_stability.py` → `results/00*/stability.json`：5000 次 bootstrap 峰中位数 26、95% 区间 [22,30]、92% 在 26；逐 prompt {22,30,22,30,26}、逐旋律 {26,30,26,26,26}、留一法全 26；mid vs early 88% trial 胜、p=1.6e-8。§4.2 加两句 |
| [MINOR] "F1 saturates at 88%" 歧义 | 改为 "reaches 95% of its maximum only at 88% of the trajectory"（贡献列表 + §4.1） |
| [MINOR] observational vs interventional | §4.2 开头与贡献 2 加了这一区分 |
| [MINOR] RAPG-online 定位 | 改写为 negative result：不优于 MLSP（0.351 vs 0.363, p=0.53），因为 R_t 跟随可解码性，与核心论点一致；§5 删除 "letting a reliability threshold place it online" |
| [MINOR] voiced coverage | §4.3 加句：0.88 (RAPG) vs 0.86 (MLSP) vs 0.84 (SAO) |
| [MINOR] FAD "lowest" / CLAP "slightly higher" | 改为 "comparable (0.75 vs 0.79)" / "preserved … no guided method differs significantly from MLSP-fixed" |
| [MINOR] MLSP bug 披露措辞 | 改为中性："We correct a temporal-scaling mismatch in the public implementation of [8] …" |
| [MINOR] Top-K(F1) = Late | §4.3 明确："Because F1(t) increases monotonically, the top-K steps of RAPG-cal(F1) are exactly the late window (35–49), so its comparison with Late isolates reliability-based strength adaptation" |
| [MINOR] "determines everything" | 改为 "placement is the dominant factor" |
| [NIT] 步索引语义 | §2.1 定义 "Step t denotes the latent returned by the t-th scheduler update, to which guidance is applied before the next denoising step"；Fig.1 caption 加 "Positions index the latent after the corresponding scheduler update" |
| [NIT] 5 s / 4 s | §3 写 "eight-note target melodies … (4 s within each 5-s clip)" |
| 听评 | 未做；§5 加 "perceptual validation of the CLAP and Fréchet-distance quality proxies … left to future work" |

未改：题目；作者信息；`\copyrightnotice` 仍注释（camera-ready 再开）。

## 11. Stable Audio 3 迁移实验（v3，2026-09-16 晚）

**动机：** 评审/作者疑问"为什么用 SAO 1.0 而不是已发布数月的 SA3"。§3 已加说明；同时在 `stable-audio-3-medium-base`（2.3B，rectified flow，256 通道 latent @ 10.8 fps，50 步 Euler，CFG 7，默认 log-SNR-shift 时间表）上复现 Exp A/C 和一个小 Exp B。发布版 `stable-audio-3-medium` 是对抗式 post-training 的 8 步采样器，不适合做 schedule 分析，故用 base。

**实现：** `w2s/sa3.py`（`SA3Generator`，与 `W2SGenerator` 同接口；用 `generate_diffusion_cond_inpaint`，callback 里 in-place 修改 `x_i`；轨迹行 i = 更新后 latent x_{i+1}）、`w2s/backend.py`（`W2S_BACKEND=sa3`）、`w2s/scripts/sa3_encode_maestro.py`（MAESTRO v3 120 首 × 6 段 10 s → 608 train / 108 val）、`sa3_train_probe.py`（同 MLSP CNN 结构 248k 参数，噪声增广按 RF 插值 t∈{0,.1,…,.6}）。Job 016/017/017b/018b/019c/020c/021。GPU 环境 `stablenew`（需 `conda run -n stablenew`）。

**结果（`results/018b_sa3_probe`, `019c_sa3_expA`, `020c_sa3_expC`, `021_sa3_expB`）：**
- probe 验证集 micro-F1 0.59（t=0），随噪声下降（t=0.41→0.48，0.6→0.32）。
- Exp A：F1 沿真实轨迹单调升，在 76% 处才到峰值 95%；R_t 与 F1 相关 0.93（单轨迹中位数 0.89）。
- Exp C：ΔC 在前 40% 步平坦 ≈0.08（此处时间表 t≥0.95，probe 处于 chance），后 20% 掉到 0.017；corr(R_t, ΔC) = −0.77。**甜点区在轨迹起点——与 SAO 相反。** bootstrap 峰中位数 position 10，区间 [10,18]。
- Exp B（50 配对 trial，K=15）：SAO-unguided 0.09 / Early 0.44 / Mid 0.40 / Late 0.18 / Uniform 0.43 / MLSP 窗 0.28 / **RAPG-cal(ΔC)（步 4–18）0.47** / Top-K const 0.44；RAPG vs MLSP 窗 p=3.5e-5；voiced coverage 0.78 vs 0.66（无 gaming）。
- 论文：新增 §4.5 + Fig. 3（`fig_sa3.pdf`，含顶部噪声水平刻度），摘要/贡献 2/§3/§5 同步；结论改为"甜点区是模型特定的，固定窗不迁移，测 ΔC 再放预算的流程迁移"。正文仍 4 页 + 参考文献 1 页；数字由 `make_figures.py --sa3A/--sa3C/--sa3B` 生成（`numbers_sa3.tex`）。

**注意点：** SA3 的 Gaussian-proxy 变体（Exp A 的 `gauss_proxy`）在 RF 参数化下无意义（未用于论文）；Exp C 在 SA3 上未算 CLAP（`W2S_CLAP=0`）。

## 12. 模拟评审（3 位审稿人 + AC）与三个补充角度的回应（v4，2026-09-23）

模拟评审打分 3/4/2（AC：borderline，主要卡在 §4.5 的措辞、命名、随机方向对照缺失、早窗结论过强、旧对齐结果未量化）。三个补充角度：AI 味措辞、评测↔结论对应、demo 页与脚注承诺不符。处理如下；**§10 里"题目未改"已失效，题目和方法名已改**。

### 12.1 新增 GPU 作业（全部 exit 0，结果已在仓库）

| Job | 目的 | 结果 → 论文位置 |
|---|---|---|
| 022 `sa3_expB_clap` | SA3 Exp B 补 CLAP（`rescore_expB_clap.py`，从保存的 wav 重打分） | `results/021_sa3_expB/table.csv`：unguided 0.375 / SCPG 0.330 / MLSP 窗 0.361 / early 0.331 → §4.5 "the gain is not free… CLAP falls 0.375→0.335 (Holm p<0.001)" |
| 023 `sao_randburst` | SAO 峰值位置（step 26）随机方向对照，`W2S_RANDOM_GRAD=1`（同范数随机向量） | ΔC 0.003 (n.s.) vs 0.104 probe 梯度，paired p<1e-6 → §4.2 `\RANDNOTE` |
| 024 `sa3_randburst` | SA3 同上（position 10） | −0.026 vs 0.089，p<1e-4 → §4.5 `\RANDNOTESAThree` |
| 025 `sao_early_strength` | 早窗 (0–14) 在 λ=α∈{0.2,0.5,1.0} 下是否能 steer（dev 集，2 seeds） | coherence 0.12→0.23/0.29/0.41，CLAP 0.207→0.187/0.184/0.173 → §4.2 `\EARLYNOTE`："early window is inert only at this step size … at matched update norm the mid window is where a step counts most" |
| 026 `sao_legacy_mlsp` | 用 MLSP 原始（错位）对齐 `TARGET_FPS=204.8` 跑完整测试集 | 对真旋律 0.259，对其第一个音级 0.68 → §3 `\LEGACYNOTE`（"finding stands, but outputs followed the first note"） |
| 027 `sa3_demo` | SA3 demo 音频（prompts 0/2/4/6 × sao/mlsp/rapg_cal_dc） | `results/027_sa3_demo/audio/*.demo.wav` → demo 页 SA3 块 |
| 028 `sa3_expC_clap` | SA3 Exp C 补 ΔCLAP 列 | `results/020c_sa3_expC/by_position.csv` 的 `d_clap` → Fig. 3(b) 右轴 |

### 12.2 无 GPU 分析（`make_figures.py` "review macros" 块，全部由 CSV 自动生成）

- 配对检验补齐：Top-K(ΔC) vs MLSP 窗 p<1e-6；vs Mid +0.010, p=0.002；R-scaling vs Top-K +0.011，122/184 trial 胜，p<1e-5（`\pTopKvsMLSP` 等）。
- 放置贡献占比：(0.434−0.363)/(0.445−0.363)=**86%**（原文"about 85%"改为计算值）。
- **实际 λ 披露**：R-scaling 在测试集上的实际均值 λ=0.057（非 0.05）→ §4.3 明写"its gain cannot be separated from a 15% larger average step"。
- 固定 t 的跨样本 Spearman(R_t, F1)：中位 0.27（早）/0.57（后 15 步），86% 位置为正 → §4.1（回应"相关是趋势共享的假象"）。
- 峰稳定性：bootstrap 92% 在 s=0.54，留一 prompt 峰范围 0.54–0.62 → §4.2。
- 噪声水平换算（用于跨模型比较）：SAO step 18 ↔ σ≈27 (t≈0.96)，step 32 ↔ σ≈3.2 (t≈0.76)；SA3 的窗 t≥0.95。→ §4.5 把 "opposite end" 改为 "**adjacent in noise level**: SAO's steps 18–32 span σ≈27→3.2 (t≈0.96→0.76) while SA3's plateau is t≥0.95"。
- Holm family 明确：每个参照/指标 9 个比较。
- 校准成本：每模型一次 burst sweep，12 位置 × 50 dev trial + 参照 = 650 次生成（SAO 0.8 GPU-h，SA3 0.5）。

### 12.3 论文改动（`paper/main.tex`，5 页 = 正文 4 + 参考文献/伦理 1，0 overfull）

- **题目/命名**：*When to Steer: Sensitivity-Calibrated Probe Guidance for Controllable Music Diffusion*；方法名 **SCPG**（"RAPG" 全部退役）；表行 "SCPG: Top-K(ΔC)"、"SCPG + R-scaling"、"Top-K(F1) + R-scaling"、"R-threshold (online)"，`+/−` 标记 = 相对 MLSP 窗 Holm 校正后显著高/低。
- §4.5 八处逐句修正：opposite→adjacent；"2.3B" → "1.4B-parameter flow-matching transformer (2.3B with autoencoder and text encoder)"；"rises monotonically" → "essentially monotonically"；"useless on SAO" → "barely moves pitch-class content on SAO at this step size"；加随机方向对照；加 CLAP 代价；加"neither fixed window transfers, the burst sweep does"；FAD 未在 SA3 评估写明。
- AI 味清理：删掉 "Crucially/Notably/Importantly"、"paradigm"、"robust" 类空词，结论一段改为具体陈述；"determines everything" → "placement is the dominant factor"；限制条件压成一句 "Limits: pitch-class control, guidance on z_t, 50-step samplers, unvalidated quality proxies."
- 评测↔结论：贡献 3 与 §5 只声称 §4 里检验过的内容（placement 主要、R-scaling 小增益且与步长混杂、R-threshold 无增益、SA3 窗不同但流程迁移）；"10≈25" 已是 "recover 95%"。
- 相关工作补引：Stable Audio 3 (`evans2026stableaudio3`)、ARC post-training (`novack2025arc`)、SMITIN (`koo2024smitin`)、MusicGen、P2 weighting、h-space；MLSP 引用加 arXiv:2609.04516。
- 其它：Fig. 1 顶部加噪声水平轴；Fig. 3(b) 加 ΔCLAP 右轴；hyperref 元数据（pdftitle/pdfauthor）；伦理声明移到第 5 页并提及 SA3 + Stability AI Community License；作者单位 Zang = "Independent Researcher"（与 arXiv:2609.04516 一致，**仍需作者确认**）；旋律描述改为 "5 melodies of 7–8 quarter notes (3.5–4 s)"（alternating 是 7 音，原文"eight-note"不准确）。

### 12.4 Demo 页（`demo/template.html`、`demo/build_demo.py`、`docs/index.html`，GitHub Pages；v2 于 9/23 晚重做为"结果 demo"）

- 落地页现在是 **Results and audio examples**，不再是听评测试：标题 "When to Steer — Audio Examples"；顶部 "Results at a glance" = 一段结论 + 论文 Fig. 1 / Fig. 3（从 `paper/fig_*.pdf` 栅格化嵌入）+ 两张测试集均值表（SAO n=225 = Table 1；SA3 n=50 = §4.5），数字直接读自 `results/*/table.csv`。
- 每个 item 的带标签音频从 3 条扩到 **6 条**：Unguided；Early / Mid / Late 固定窗（job 029/030 从 Exp B 的 wav 收集，同一 seed）；MLSP 窗；SCPG + R-scaling。这样 "放置是主因"（Early 0.166 / Mid 0.424 / Late 0.315，同预算）在页面上可以直接听出来，而不只是看表。每条 clip 标出引导步、coherence / chroma / CLAP。
- 顶部链接行兑现脚注承诺：Code (branch icassp) / Per-run results (CSV) / Paper (PDF) / How this page was built；"How the examples were chosen"：6 对 prompt–melody 在听之前固定（test prompts 0,2,4,6,10,12，seed 0），无挑选。
- 盲测（A/B/C + MOS）保留但**不在页面上显示**，只能通过 `#blind` 打开；评分只存本地浏览器，页内注明不属于论文（§6 无人类被试）。评分者 ID / 进度条在结果视图中隐藏。
- SA3 对比块 4 item，同样 6 条；页脚写明两模型采样设置、窗口定义与 SCPG 步数（SAO 19–33、SA3 4–18）。
- claude.ai artifact（version 3）与 GitHub Pages 内容相同；后者评分只用 localStorage。

### 12.5 未做 / 作者侧待办

- 听评（perceptual study）未做，§5 明写为 future work。
- **作者信息**：Zang 单位待确认；三位作者 ORCiD（提交系统必填）；`\copyrightnotice` 仍注释（camera-ready 再开）。
- ICASSP LLM 政策：任何由助手起草的段落作者需自行改写；投稿截止 **2026-09-23 23:59:59 AoE**（= 9/24 07:59 EDT），可 Revise Submission 到同一时刻。

## 13. 写作审阅（第三轮，2026-09-23 晚）的回应（v5）

审阅重点是措辞与论证边界（AI 味、重复包装、结论强于证据、段落堆叠、图文呈现）。全文按其建议重写，正文仍 4 页 + 第 5 页参考文献/伦理，0 overfull；数字全部由 `make_figures.py` 从 CSV 生成，改写后逐条核对过。

| 审阅意见 | 处理 |
|---|---|
| 摘要把"F1 达峰值 95% 的位置"和"敏感度峰值"都写成 peak | 改为 "bursts change melodic coherence most at 54% of sampling, whereas probe F1 reaches 95% of its maximum only at 88%" |
| "audio realism unchanged" / "quality question is settled" / "no measurable quality cost" | 全部改为只陈述 CLAP（不显著）和 FAD（数值接近），§4.3 明写 "Neither metric measures perceived quality"，§5 写明需要听评；摘要补 SA3 的 CLAP 0.375→0.335 |
| "连续窗口 ⇒ 不遗漏交互" | 删除；改为 "Ranking steps by single-burst effects is a heuristic; the full-budget evaluation of Sec. 4.3 tests whether it yields an effective multi-step schedule" |
| R-scaling "only its distribution changes" | 限定为开发集均值匹配，测试集实际均值可能不同；0.445 退出摘要，§4.3 保留数值 + 15% 步长混杂 |
| LatCH 20% vs Early 0–14（30%） | Baselines 明确：Early 是 "budget-matched early schedule inspired by LatCH"，LatCH 自身窗口是前 20%（10/50），Early 为匹配预算取前 30%，在本文更新规则下测试早期放置；§4.5 改称 "the budget-matched early schedule (0–14)"；删除 "as LatCH may" |
| "zero overhead / no inference cost" | 改为 "Once calibrated, SCPG requires no online schedule search and uses the same number of guidance updates as the fixed-schedule baselines" |
| 机制解释写成事实 | 改为 "One possible explanation is … our experiments do not identify this mechanism" |
| 命名：actual accuracy / reliability / entropy not meaningful / at chance / Transfer | 分别改为 pYIN-derived labels、"target-free reliability proxy" + sigmoid 不构成类别分布的具体说明、"probe F1 remains low"、"Evaluation on Stable Audio 3"；coherence 定义处注明只是 pitch-class match rate |
| 同一发现反复包装（read/write/nudge/push、≠、sweet spot…） | 摘要留一次对照；结果只给测量；结论只说适用范围。全文已无 cheap / hand-picked / Curiously / nudge / push / sweet spot / not free / wrong quantity / stand out / settled / triples / handful |
| 摘要 240 词过密 | 重写为 178 词：问题→校准→54%/88%→0.363→0.434→SA3 与 CLAP 代价；删 125k、R-scaling、在线阈值 |
| 贡献列表 220 词 | 压成三条各 1–2 句（对照分析 / SCPG 及其可靠性消融 / 两模型评估含代价） |
| 方法顺序：先 R 后 SCPG | 改为 2.1 更新规则 → 2.2 敏感度校准与 SCPG（burst = 连续三步 {p,p+1,p+2}，λ=0.10；12 个位置线性插值到 50 步并 3 步滑动平均；Top-K，平局取早）→ 2.3 可靠性变体（R-scaling、在线阈值）作消融 |
| §4.2 / §4.3 / §4.5 长段 | §4.2 拆为主结果 + 稳健性；§4.3 按"主比较 / 指标边界 / R 消融"三段；§4.5 拆为设置 / 发现 / 结果与代价 |
| §3 "so the finding of [12] stands" 辩护语气 | 删除；只写错误、影响、统一修复，原对齐的 0.259 / 0.68 作括号内诊断 |
| 结论 "Limits: …" 像工作笔记 | 按示范改写，限制写成完整句（pitch-class、z_t、50 步配置、CLAP/FAD 为代理、需听评） |
| 图 1 版面 / 图 1 与图 3 组织不一致 / 符号 t 复用 / 表 1 位置 / MLSP 标签 / "0.54−−0.62" / 表注过密 | 图 1 改为三面板（(a) (b) + 带 Early/MLSP/SCPG 标签的步序条），3.05 in 高；图 3 改为与图 1 完全相同的版式与配色；噪声坐标改为 τ，t 只表示步；表 1 移到第 3 页 §4.3 旁；标签改 "MLSP-fixed (20:2:48)" 并在表注解释；区间改为 "between s=0.54 and 0.62"；表注精简并注明 MLSP-fixed n=224（一次生成失败） |
| 校对 agent 补充发现 | 步索引与进度定义对齐（t=0…N−1，s=(t+1)/N，图轴改为 "denoising progress s"）；"late bursts change the audio little" 补数据（final-latent BCE −0.14 vs −0.11；log-mel 0.21 vs 0.53）；"any 15 steps inside it are near-equivalent" 降为 "schedules inside it should be near-equivalent"；§4.5 "SAO's steps 18–32" 改为 "SAO's mid window (steps 18–32)"；摘要 "0.28" 注明是 prior schedule；"alternating arpeggio" 统一为 "alternating pattern" |

未采纳：图 1 仍为单栏（双栏会挤掉正文）；§4.2 稳健性段保留六项检验（各一句）。

## 14. 第四轮复审（数据一致性，2026-09-23 晚）的回应（v6，最终）

八条全部按复审意见做了局部修正，没有再改叙事：

1. §4.5 SA3 主比较 p 值：常量 SCPG 0.44 vs MLSP 0.28 改为 **Holm-adjusted p=0.002**（新增自动宏 `\saThreePTopKvsMLSPholm`，来自 `021_sa3_expB/table.csv` 的 `p_coherence_mlsp_vs_mlsp_holm`）。
2. §4.3 在线阈值：删除 "admits mostly late steps"；改为报告实测触发分布（首次更新在第 2–12 步，最后一次更新中位数在第 40 步），并在 §2.3 与 Baselines 注明在线变体同时使用 R-scaling。
3. §4.3 "R-scaling changes nothing" 改为 "slightly reduces mean coherence, from 0.315 to 0.310 (paired p<1e-6)"。
4. §4.2 CLAP 句改为 "lowers CLAP slightly at every position (mean ΔCLAP between −0.012 and −0.005 relative to the unguided output)"；§4.4 改为 "CLAP ranges from 0.351 to 0.373 across budgets and schedules (0.348 unguided on this subset), with overlapping CIs"。
5. §4.2 删除 "changing the audio the least"，只写 s=0.94 相对 s=0.54：BCE 下降更多（−0.14 vs −0.11）、log-mel 变化更小（0.21 vs 0.53）。
6. 表 1 注改为 "MLSP-fixed has 224 valid coherence values and 225 CLAP values, and paired tests use complete pairs"；§4.5 注明 SA3 的 Mid/Late 各 49 个有效 coherence 值。
7. §4.4 删除 "not simply additive"；§4.5 删除 "should be near-equivalent"，只写 Early (0–14) 也在平台内且其均值在 SCPG 的 CI 内。
8. 图 1 / 图 3 右边距从 0.87 改为 0.845，右轴标题（R_t、ΔCLAP）在 220 dpi 渲染下完整可见。

**统计核对表：** 新脚本 `w2s/scripts/check_paper_numbers.py` 从 `results/` 重算正文里全部手写统计量（39 条：峰值/饱和位置、相关系数、bootstrap/LOO、随机方向对照、早窗强度扫描、表 1 各项、R 消融、在线阈值触发步、预算扫描、SA3 全部数字、旧对齐诊断），输出 `paper/STATS_CHECK.md`；当前 **39/39 一致**。自动宏（numbers*.tex、table_main.tex、budget_summary.csv）继续由 `make_figures.py` 重新生成后逐字节比对。

## 15. 第五轮复审（核对脚本与收尾，2026-09-23 深夜）的回应（v7，定稿）

1. **核对脚本漏检**：`w2s/scripts/check_paper_numbers.py` 重写。现在 (a) 每条比较同时检查数值与方向（不只看 p 值）；(b) 手写统计量优先从逐运行文件重算（47 条中 42 条 per-run，5 条读汇总），并在开头加了"汇总表 = 逐运行均值"的一致性检查（009 / 021 table.csv、004 by_position.csv、014 pareto.csv、003 summary.csv）；(c) bootstrap 与 LOO 峰值改为从 per_run 重新抽样/重算（seed 0，5000 次），再与 stability.json 比对；(d) 每条检查附带它所代表的正文片段，脚本读取 `paper/main.tex` 核对片段确实存在（"TEXT MISSING" 状态）。复审给的两个对抗测试（SA3 汇总均值改 0.99；预算实验方法标签互换）现在分别触发 1 条和 3 条失败。真实数据：**47/47 通过**，输出 `paper/STATS_CHECK.md`。
2. **§4.3 变体混用**：alternating 改为常量 SCPG 的 0.317 vs 0.346；第一音级参照改为 0.434 vs 0.249（与该段讨论的常量 SCPG 一致；voiced fraction 0.88 两种变体相同）。
3. **晚窗 p 值口径**：写明 "unadjusted paired p<10⁻⁶"（对 Late 的 Holm 校正后 ≈2.5e-6，仍 <1e-5）。
4. **两句过强总结**：删除 "so late guidance changes what the probe reads … more than it changes the audio"；"at matched update norm" 改为 "under the common update rule and nominal guidance strength"。
5. **SA3 平台**：正文改为 ΔC≈0.08（前 40% 五个位置均值 0.081，前三个位置宏 0.079）。
6. **图 1 边距**：右边距 0.845→0.82、下边距 0.125→0.14；`make_figures.py` 加入标题/轴标签越界守卫（越界即打印 WARNING），并用 `pdftotext -bbox-layout` 复核：图 1 字形最右 245.1 pt / 页宽 248.4 pt，图 3 239.3 pt，无越界。

## 16. 最终投稿版同步（2026-09-23 深夜）

最终稿以作者确认的单作者版本为基础，摘要采用作者提供的 `When_to_Steer_ICASSP2027_final_7.pdf` 中的文字，其余论文源码不变。作者为 Yushi Ye，Carnegie Mellon University，yushiye@andrew.cmu.edu；按作者要求，PDF不显示 ORCID。保留作者确认的 AI 使用致谢，以及图1–3普通文字9.2点、表格和脚注9点的排版。

`paper/main.tex` 与 `paper/When_to_Steer_ICASSP2027_final.pdf` 对应，demo 的 Paper (PDF) 链接指向仓库中的该文件。最终PDF共5页：正文、图表与致谢在第1–4页，参考文献和伦理说明在第5页；编译无 overfull 或未解析引用。`check_paper_numbers.py` 47项通过。
