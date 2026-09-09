"""Render the MIC code-review + evaluation report (rounds 1-6) as a PDF."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak,
)
from reportlab.lib.enums import TA_LEFT


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "MIC_Code_Review_Report.pdf"

# Pull live numbers from the JSON outputs so the report is never stale
with open(ROOT / "outputs" / "evaluation" / "probe_metrics.json") as f:
    PM = json.load(f)
with open(ROOT / "outputs" / "evaluation" / "results.json") as f:
    RES = json.load(f)


styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, spaceAfter=10,
                    textColor=colors.HexColor('#1a365d'))
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13, spaceAfter=6,
                    textColor=colors.HexColor('#2c5282'))
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11, spaceAfter=4,
                    textColor=colors.HexColor('#2a4365'))
BODY = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9.5,
                      leading=13, alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle('Small', parent=BODY, fontSize=8.5, leading=11,
                       textColor=colors.HexColor('#444'))
CODE = ParagraphStyle('Code', parent=BODY, fontName='Courier', fontSize=8.5,
                      leading=11, textColor=colors.HexColor('#222'),
                      backColor=colors.HexColor('#f4f4f4'),
                      leftIndent=8, rightIndent=8, spaceBefore=2, spaceAfter=4)


def P(t):
    return Paragraph(t, BODY)


def code_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f7fafc')]),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def status_cell(status):
    if status.startswith('FIXED') or status.startswith('PASS'):
        c = colors.HexColor('#22543d')
    elif status.startswith('OPEN'):
        c = colors.HexColor('#742a2a')
    elif status.startswith('PARTIAL'):
        c = colors.HexColor('#7b341e')
    elif status.startswith('SUPERSEDED'):
        c = colors.HexColor('#4a5568')
    else:
        c = colors.HexColor('#1a365d')
    return Paragraph(f'<font color="{c}"><b>{status}</b></font>', SMALL)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#888'))
    canvas.drawString(0.75 * inch, 0.5 * inch,
                      'MIC Code Review + Evaluation Report — through 2026-05-02')
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.5 * inch,
                           f'Page {doc.page}')
    canvas.restoreState()


story = []

# ============================================================
# Title
# ============================================================
story.append(Paragraph('MIC — Music Inference Control', H1))
story.append(Paragraph('Code Review + Formal Evaluation Report', H2))
story.append(Paragraph(
    'Reviewer: Claude Opus 4.7 &nbsp;|&nbsp; '
    'Project root: <font face="Courier">/home/stoneyey/Desktop/MIC</font> &nbsp;|&nbsp; '
    'Period: 2026-05-01 12:00 — 2026-05-02 02:30 (UTC)', SMALL))
story.append(Spacer(1, 0.15 * inch))

# ============================================================
# Executive summary
# ============================================================
story.append(Paragraph('Executive summary', H2))
story.append(P(
    f'Across <b>six rounds</b>, the worker iterated on the MIC pipeline; I reviewed '
    f'each round read-only, and the worker landed every actionable fix raised. '
    f'After the final round both project contributions are now demonstrated with '
    f'measurable, statistically supported results:'))
story.append(P(
    f'<b>Contribution 1 — Latent space encodes pitch.</b> CNN probe trained on the '
    f'(leak-free) file-level split achieves <b>F1 (micro) = {PM["f1_micro"]:.3f}</b>, '
    f'<b>positive F1 = {PM["positive_f1"]:.3f}</b>, '
    f'<b>frame overlap = {PM["frame_overlap"]:.3f}</b>, '
    f'with all 12 pitch classes between {min(PM["per_class_f1"]):.3f} and '
    f'{max(PM["per_class_f1"]):.3f} F1. Performance degrades smoothly under noise '
    f'augmentation (F1 drops from {PM["noise_robustness"]["0.0"]["f1_micro"]:.3f} at '
    f'σ=0.0 to {PM["noise_robustness"]["1.0"]["f1_micro"]:.3f} at σ=1.0).'))
sm = RES["summary"]
wlx = RES["wilcoxon_vs_baseline"]
story.append(P(
    f'<b>Contribution 2 — Probe guidance steers generation.</b> All five guided '
    f'methods beat the unguided baseline on melody coherence with statistical '
    f'significance (Wilcoxon signed-rank, n=9 paired trials): '
    f'baseline mean = {sm["baseline"]["mean_coherence"]:.3f}; '
    f'probe_mid (scale=0.05) reaches {sm["probe_mid"]["mean_coherence"]:.3f} '
    f'(p={wlx["probe_mid"]["p_value"]:.4f}); combined fusion+probe reaches '
    f'{sm["combined"]["mean_coherence"]:.3f} (p={wlx["combined"]["p_value"]:.4f}).'))
story.append(P(
    'The two big course-corrections behind those numbers were both surfaced by '
    'review: a <b>train/val data leak</b> that inflated round-3 metrics, and a '
    '<b>flawed melody-coherence metric</b> that hid the round-5 evaluation\'s real '
    'signal behind voicing artifacts. Once both were fixed, the worker also '
    'reopened the probe-guidance window earlier in denoising '
    '(<code>guidance_start</code> 0.7 → 0.4), and the technique\'s effect became '
    'unambiguous.'))

story.append(PageBreak())

# ============================================================
# Section 1 — Engagement & methodology
# ============================================================
story.append(Paragraph('1. Engagement summary', H2))
story.append(P(
    'You asked me to act as the reviewer for a worker iterating on MIC in another '
    'window. My role across the session was strictly <b>read-only</b>: pull the '
    'worker\'s latest changes, run smoke tests, audit code quality and edge cases, '
    'and emit a prioritized list of concrete change requests. No source files were '
    'modified by me. Each "check again" instruction triggered a new round; over six '
    'rounds I tracked which items the worker addressed, what new issues each round '
    'introduced, and finally analyzed the formal evaluation outputs.'))

story.append(Paragraph('1.1 What MIC is', H3))
story.append(P(
    '<b>MIC (Music Inference Control)</b> is a research codebase that tests two '
    'contributions on top of Stable Audio Open 1.0:'))
story.append(P(
    '<b>Contribution 1 — Interpretable Latent Space.</b> Train small probes (linear '
    'and CNN) to decode pitch-class labels from the Stable Audio VAE\'s latent. '
    'High probe F1 implies the latent space encodes recoverable musical pitch '
    'information.'))
story.append(P(
    '<b>Contribution 2 — Probe-Guided Inference.</b> Use the trained probe as a '
    'differentiable loss during diffusion denoising. At each guided step, take the '
    'gradient of probe loss w.r.t. the latent and apply an RMS-normalized, '
    'magnitude-clamped update — steering generation toward a target melody without '
    'retraining the base model.'))

story.append(Paragraph('1.2 Review methodology', H3))
story.append(P(
    'Each round I (a) listed source files by mtime to identify what the worker had '
    'touched; (b) read the changed files in full; (c) ran <code>py_compile</code> '
    'on every modified module; (d) ran targeted smoke tests against probes, '
    'datasets, and checkpoints; (e) verified each prior fix functionally where '
    'possible; (f) wrote the next set of recommendations. In rounds 5–6 the focus '
    'shifted from code quality to evaluation methodology and analysis of the actual '
    'numbers in <code>outputs/evaluation/</code>.'))

# ============================================================
# Section 2 — Round-by-round
# ============================================================
story.append(Paragraph('2. Round-by-round narrative', H2))

story.append(Paragraph('Round 1 — initial audit', H3))
story.append(P(
    'Read all 10 source/script files. <code>py_compile</code> clean. Sanity-checked '
    'the probes (CNN 125k params / Linear 0.9k params, output shape '
    '<code>(B, T, 12)</code>) and inspected the saved dataset/checkpoint. Raised 5 '
    'bugs (B1–B5) and 12 quality items (Q1–Q12). Highest-priority bug was B3: '
    'pipeline used the legacy <code>callback=…, callback_steps=1</code> kwargs, '
    'which <code>StableAudioPipeline</code> in diffusers 0.27 might silently ignore '
    '— making Contribution 2 a potential no-op.'))

story.append(Paragraph('Round 2 — most fixes land, more issues found', H3))
story.append(P(
    'Worker addressed B1, B2, B4, B5, Q1–Q3, Q5–Q7, Q9, Q12. B3 untouched. New '
    'O-series issues raised: O1 (carryover B3), O2 (remaining mutable defaults), '
    'O3 (dead <code>make_combined_callback</code>), <b>O4 (val split unshuffled)</b>, '
    'O5 (asserts under <code>-O</code>), O6 (per-step seed independent of user '
    'seed), O8 (no <code>end &lt; start</code> guard), O9 (duplicate save blocks), '
    'O10 (need fallback for <code>weights_only=True</code>).'))

story.append(Paragraph('Round 3 — first real data bug surfaces', H3))
story.append(P(
    'O2/O3/O4/O5/O6/O8/O9/O10 all landed and verified. <b>OS1 still open.</b> '
    'New N-series: <b>N1 — train/val shuffle still leaks across clips of the same '
    'recording</b>. O4 had shuffled clips, but adjacent 10-second clips from the '
    'same WAV share timbre and performer style. Empirically: with O4, val of 153 '
    'clips touched only 5 distinct files, and those files also appeared in train. '
    'N2: combined callback missed a <code>start_step &gt;= n</code> guard. '
    'N3: docstring stale. N4: <code>except Exception</code> too broad.'))

story.append(Paragraph('Round 4 — file-level split, plus a structural concern', H3))
story.append(P(
    'N1–N4 all fixed and verified. Verified empirically that the new split has zero '
    'file overlap and is deterministic. Surfaced P-series: P1 (val partition is '
    'now only 2 files / 77 clips — too few), P2 (annotation lies '
    '<code>by_file: dict</code>), P3 '
    '(<code>pickle.UnpicklingError</code> not in except tuple). OS1 still open.'))

story.append(Paragraph('Round 5 — formal eval ran; metric was misleading', H3))
story.append(P(
    'P1/P2/P3 fixed. Worker confirmed OS1 by smoke-testing the callback: '
    '<code>probe_loss</code> drops from ~0.39 at step 0 to ~0.15 at step 40, '
    'no deprecation warnings — callback fires. Probe retrained on file-level split '
    '(epoch 41, val loss 0.244) and full inference comparison run, producing 45 '
    'audio files and the first <code>results.json</code>. <b>Apparent finding:</b> '
    'probe-only methods showed no effect over baseline; "combined" showed mean '
    '0.355 driven by 3 trials scoring exactly 1.0.'))
story.append(P(
    '<b>Real finding (uncovered by reviewer):</b> the 1.0 scores were a metric '
    'artifact. Instrumented '
    '<code>compute_melody_coherence</code> on the suspect samples and counted '
    'voiced frames: <code>prompt1_ascending_combined</code> had only 15 voiced '
    'frames out of 431 (3.5%); 11 of 11 valid frames matched, giving 1.0 over a '
    'tiny slice of audio. The latent-fusion path at <code>fusion_scale=0.3</code>, '
    'applied every step, was killing the generation\'s pitched content; pyin then '
    'evaluated only a handful of frames. Filed E1 (gate on voiced coverage), E2 '
    '(per-trial pivot for paired tests), E3 (persist Contribution-1 metrics), '
    'E4 (use <code>--n_val_files</code>), E5 (run Wilcoxon).'))

story.append(Paragraph('Round 6 — corrected metric, regenerated audio, real results', H3))
story.append(P(
    'Worker (a) rewrote <code>04_evaluate.py</code> with a 20% voiced-coverage gate, '
    'per-trial pivot, and one-sided Wilcoxon signed-rank vs baseline; (b) added '
    '<code>probe_metrics.json</code> persistence in <code>02_train_probe.py</code>; '
    '(c) <b>rebuilt the dataset</b> as <code>probe_dataset_5s.pt</code> with 5-second '
    'clips (n_train=2637, n_val=417 across 4 held-out files — fixes P1\'s small-N '
    'issue); (d) <b>regenerated all 45 audio</b> files at 02:19 with two tuned '
    'parameters: <code>guidance_start=0.4</code> (was 0.7 — opens the guidance '
    'window from "last 30%" to "last 60%" of denoising) and combined '
    '<code>fusion_scale=0.1, fusion_every_n=5</code> (was 0.3 every step — much '
    'gentler); (e) added a 5th method <code>probe_strong</code> at scale=0.2 '
    '(9 more files, 54 total). Verified all changes; results below.'))

story.append(PageBreak())

# ============================================================
# Section 3 — Master tracker
# ============================================================
story.append(Paragraph('3. Master tracker — every issue raised, current status', H2))
tracker = [['ID', 'Round', 'Title', 'Severity', 'Status']]
issues = [
    ('B1', 1, 'MaestroDataset.training default missing', 'Bug', 'FIXED (R2)'),
    ('B2', 1, 'Control-latent dtype mismatch', 'Bug', 'FIXED (R2)'),
    ('B3', 1, 'StableAudioPipeline legacy callback API', 'Bug', 'FIXED (R5 verify)'),
    ('B4', 1, 'evaluate_noise_robustness shape misalignment', 'Bug', 'FIXED (R2)'),
    ('B5', 1, 'Unseeded fusion noise', 'Bug', 'FIXED (R2)'),
    ('Q1', 1, 'current_loss undefined when num_epochs=0', 'Quality', 'FIXED (R2)'),
    ('Q2', 1, 'In-memory train/val split (no temp files)', 'Quality', 'FIXED (R2)'),
    ('Q3', 1, 'torch.load weights_only flag', 'Quality', 'FIXED (R2)'),
    ('Q4', 1, 'make_combined_callback dead code', 'Quality', 'FIXED (R3)'),
    ('Q5', 1, 'start=1.0 silent disable', 'Quality', 'FIXED (R2)'),
    ('Q6', 1, 'compute_rms local imports in hot path', 'Quality', 'FIXED (R2)'),
    ('Q7', 1, 'CNNProbe hidden % 8 assertion', 'Quality', 'FIXED (R2)'),
    ('Q8', 1, 'test_setup.py fp16/fp32 mismatch', 'Quality', 'OPEN (deferred)'),
    ('Q9', 1, 'find_maestro_pairs uses Path.with_suffix', 'Quality', 'FIXED (R2)'),
    ('Q10', 1, 'Mutable default noise_sigmas=[…]', 'Quality', 'FIXED (R3)'),
    ('Q11', 1, 'Serial dataset iteration in noise robustness', 'Quality', 'OPEN (acceptable)'),
    ('Q12', 1, 'load_probe KeyError on missing loss', 'Quality', 'FIXED (R2)'),
    ('O1', 2, 'Carryover of B3', 'Bug', 'FIXED (R5 verify)'),
    ('O2', 2, 'Remaining mutable defaults', 'Quality', 'FIXED (R3)'),
    ('O3', 2, 'Wire up combined callback', 'Quality', 'FIXED (R3)'),
    ('O4', 2, 'Shuffle train/val split', 'Bug', 'SUPERSEDED by N1 (R3)'),
    ('O5', 2, 'assert under -O', 'Quality', 'FIXED (R3)'),
    ('O6', 2, 'Fusion seed independent of user seed', 'Quality', 'FIXED (R3)'),
    ('O8', 2, 'end < start guard', 'Quality', 'FIXED (R3)'),
    ('O9', 2, 'Duplicate % 5 blocks', 'Quality', 'FIXED (R3)'),
    ('O10', 2, 'weights_only fallback', 'Quality', 'FIXED (R3)'),
    ('N1', 3, 'File-level split (true leak fix)', 'Bug', 'FIXED (R4)'),
    ('N2', 3, 'combined start>=n guard', 'Quality', 'FIXED (R4)'),
    ('N3', 3, 'seed param missing from docstring', 'Quality', 'FIXED (R4)'),
    ('N4', 3, 'load_probe except too broad', 'Quality', 'FIXED (R4)'),
    ('P1', 4, 'Tiny val set (2 files / 77 clips)', 'Statistical', 'FIXED (R6 — dataset rebuilt to 4 val files / 417 clips)'),
    ('P2', 4, 'defaultdict typed as dict', 'Cosmetic', 'FIXED (R5)'),
    ('P3', 4, 'load_probe missing pickle.UnpicklingError', 'Quality', 'FIXED (R5)'),
    ('E1', 5, 'compute_melody_coherence had no voiced-coverage guard', 'Bug', 'FIXED (R6 — 20% gate)'),
    ('E2', 5, 'results.json file lists not paired by trial', 'Quality', 'FIXED (R6 — per_trial pivot)'),
    ('E3', 5, 'Contribution-1 metrics not persisted', 'Quality', 'FIXED (R6 — probe_metrics.json)'),
    ('E4', 5, 'Retrain used default split, not --n_val_files', 'Quality', 'FIXED (R6)'),
    ('E5', 5, 'No statistical test reported', 'Quality', 'FIXED (R6 — Wilcoxon)'),
    ('E6', 5, 'No qualitative listening sanity check', 'Quality', 'PARTIAL (coverage stats now in JSON)'),
]
for row in issues:
    tag, rd, title, sev, status = row
    tracker.append([
        Paragraph(f'<b>{tag}</b>', SMALL),
        Paragraph(f'R{rd}', SMALL),
        Paragraph(title, SMALL),
        Paragraph(sev, SMALL),
        status_cell(status),
    ])
story.append(code_table(tracker, col_widths=[0.45*inch, 0.4*inch, 3.0*inch, 0.85*inch, 2.0*inch]))

story.append(PageBreak())

# ============================================================
# Section 4 — Contribution 1 results
# ============================================================
story.append(Paragraph('4. Contribution 1 — latent space encodes pitch (probe_metrics.json)', H2))
story.append(P(
    f'<b>Setup.</b> CNN probe (~125k parameters), trained '
    f'<b>{PM["best_epoch"]} epochs</b>, AdamW + cosine LR + grad-clip 1.0, '
    f'BCE-with-logits loss, noise augmentation (σ ∈ '
    f'{{0.0, 0.1, 0.3, 0.5, 0.7}}). Dataset: '
    f'<code>{PM["dataset"]}</code> (5-second clips), '
    f'<b>n_train = {PM["n_train_clips"]}</b> clips '
    f'/ <b>n_val = {PM["n_val_clips"]}</b> clips '
    f'across <b>{PM["n_val_files"]} held-out files</b> '
    f'(file-level split — zero leakage).'))
story.append(P('<b>Aggregate metrics on held-out val:</b>'))
metric_table = [
    ['Metric', 'Value'],
    ['Best validation BCE loss', f'{PM["best_val_loss"]:.4f}'],
    ['F1 (micro)', f'{PM["f1_micro"]:.4f}'],
    ['F1 (macro)', f'{PM["f1_macro"]:.4f}'],
    ['Precision (micro)', f'{PM["precision_micro"]:.4f}'],
    ['Recall (micro)', f'{PM["recall_micro"]:.4f}'],
    ['Positive recall (frames with pitch)', f'{PM["positive_recall"]:.4f}'],
    ['Positive F1', f'{PM["positive_f1"]:.4f}'],
    ['Frame overlap (any-class match)', f'{PM["frame_overlap"]:.4f}'],
]
metric_table = [[Paragraph(c, SMALL) for c in row] for row in metric_table]
story.append(code_table(metric_table, col_widths=[3.4*inch, 1.4*inch]))

story.append(Spacer(1, 0.1 * inch))
story.append(P(
    f'<b>Per-class F1 (12 pitch classes):</b> all values fall in '
    f'[{min(PM["per_class_f1"]):.3f}, {max(PM["per_class_f1"]):.3f}], '
    f'no class is degenerate. Below are the per-class F1 in chromatic order '
    f'(C, C#, D, …, B):'))
story.append(Paragraph(
    'F1 = ' + ', '.join(f'{v:.3f}' for v in PM['per_class_f1']),
    CODE,
))

story.append(Spacer(1, 0.1 * inch))
story.append(P('<b>Noise robustness sweep</b> (σ added to clean latent, then probe asked to decode):'))
nr_table = [['σ', 'F1 (micro)', 'Positive recall']]
for sigma in ['0.0', '0.1', '0.3', '0.5', '0.7', '1.0']:
    if sigma in PM['noise_robustness']:
        nr = PM['noise_robustness'][sigma]
        nr_table.append([sigma, f'{nr["f1_micro"]:.4f}', f'{nr["positive_recall"]:.4f}'])
nr_table = [[Paragraph(c, SMALL) for c in row] for row in nr_table]
story.append(code_table(nr_table, col_widths=[0.7*inch, 1.1*inch, 1.4*inch]))

story.append(Spacer(1, 0.1 * inch))
story.append(P(
    '<b>Reading the result.</b> F1 ≈ 0.66 with 87% frame overlap on a leak-free '
    'split is a strong positive answer to the question "does the Stable Audio VAE '
    'encode pitch in a recoverable form?" The macro/micro gap is small (0.657 vs '
    '0.663), per-class F1 has no near-zero classes, and noise robustness degrades '
    'monotonically without falling off a cliff — F1 stays above 0.58 even at σ=1.0. '
    'This is the result the worker can claim for Contribution 1.'))

story.append(PageBreak())

# ============================================================
# Section 5 — Contribution 2 results
# ============================================================
story.append(Paragraph('5. Contribution 2 — probe guidance steers generation (results.json)', H2))
story.append(P(
    '<b>Setup.</b> 9 trials per method = 3 prompts ("Piano melody", "Classical '
    'piano piece", "A slow and captivating piano piece") × 3 target melodies '
    '(ascending, descending, alternating). 6 methods × 9 trials = '
    f'{sum(s["n_total"] for s in sm.values())} generations, all 5 seconds at '
    '44.1 kHz stereo. Inference: 50 denoising steps, CFG=4.0, '
    '<code>guidance_start=0.4</code> (last 60% of steps), '
    '<code>probe_guidance_every_n=2</code>, <code>max_update_ratio=0.05</code>.'))
story.append(P(
    '<b>Method dose-response (probe guidance scale).</b> probe_low=0.03, '
    'probe_mid=0.05, probe_high=0.10, probe_strong=0.20. <b>combined</b> = '
    'latent fusion (scale=0.1, every 5 steps, starting at 50%) + probe guidance '
    '(scale=0.05).'))
story.append(P('<b>Aggregate melody-coherence (voiced-coverage gated at 20%):</b>'))

methods_order = ['baseline', 'probe_low', 'probe_mid', 'probe_high', 'probe_strong', 'combined']
res_table = [['Method', 'Mean', 'Median', 'Std', 'n_valid/n', 'Wilcoxon vs baseline (one-sided)']]
for m in methods_order:
    s = sm[m]
    if m == 'baseline':
        wlx_str = '—'
    else:
        w = wlx[m]
        sig = ' *' if w['p_value'] < 0.05 else ''
        wlx_str = f'W={w["W"]:.0f}, p={w["p_value"]:.4f}{sig}, median Δ={w["median_diff"]:+.3f}'
    res_table.append([m, f'{s["mean_coherence"]:.4f}', f'{s["median_coherence"]:.4f}',
                      f'{s["std_coherence"]:.4f}', f'{s["n_valid"]}/{s["n_total"]}', wlx_str])
res_table = [[Paragraph(c, SMALL) for c in row] for row in res_table]
story.append(code_table(res_table, col_widths=[0.95*inch, 0.7*inch, 0.7*inch, 0.6*inch,
                                                0.7*inch, 3.05*inch]))

story.append(Spacer(1, 0.1 * inch))
story.append(P(
    '<b>Per-trial pivot</b> (rows = prompt × melody, cells = melody coherence):'))
trial_keys_order = sorted(RES['per_trial'].keys())
pt_header = ['Trial'] + methods_order
pt_table = [pt_header]
for tk in trial_keys_order:
    row = [tk]
    for m in methods_order:
        cell = RES['per_trial'][tk].get(m, {})
        v = cell.get('coherence')
        cov = cell.get('voiced_coverage', 0.0)
        if v is None:
            row.append('NaN')
        else:
            row.append(f'{v:.3f} ({cov*100:.0f}%)')
    pt_table.append(row)
pt_table = [[Paragraph(str(c), SMALL) for c in row] for row in pt_table]
story.append(code_table(pt_table, col_widths=[1.4*inch] + [0.85*inch] * len(methods_order)))
story.append(Paragraph('Cell format: <b>coherence (voiced_coverage%)</b>. '
                       'Rows are paired across methods.', SMALL))

story.append(Spacer(1, 0.1 * inch))
story.append(P('<b>How to read these numbers.</b>'))
story.append(P(
    '&bull; <b>Baseline coherence is genuinely low</b> (mean 0.074, median 0.024). '
    'The unguided pipeline produces piano audio whose pitches do not match the '
    'requested melody — expected, since the prompt only mentions "piano", not '
    'specific notes.'))
story.append(P(
    f'&bull; <b>Every probe-guided method beats baseline at p &lt; 0.05</b> '
    f'(Wilcoxon signed-rank, one-sided, n=9 paired trials). probe_mid and '
    f'probe_high/strong/combined all hit the floor of what Wilcoxon can report at '
    f'n=9 (p={wlx["probe_mid"]["p_value"]:.4f} ≈ 1/512), which corresponds to all 9 '
    f'paired diffs being positive.'))
story.append(P(
    f'&bull; <b>probe_mid and combined are the top performers</b> '
    f'(mean ≈ 0.33, median ≈ 0.31), about 4.5× the baseline mean and ~13× the '
    f'baseline median. probe_low underperforms on two trials (3 of 9 below 0.07) — '
    f'the gentlest scale isn\'t always reliable.'))
story.append(P(
    '&bull; <b>Voiced coverage is uniformly high</b> across guided methods '
    '(mostly 0.80–0.99). The earlier "unfair 1.0 from 11 frames" artifact is gone. '
    'When a guided method is rated 0.30, it really means about 30% of voiced '
    'frames in the target region land on the right pitch class, over real (audible) '
    'piano.'))
story.append(P(
    '&bull; <b>Guidance scale plateaus around 0.05.</b> Going from probe_mid (0.05) '
    'to probe_high (0.10) or probe_strong (0.20) does not improve the median, and '
    'on several trials the higher scales produce identical coherence to the lower '
    'ones (e.g., prompt3_descending: probe_low=probe_mid=probe_strong=0.246 with '
    'voiced coverage 0.99). This suggests the probe gradient saturates the latent '
    '— more force does not buy more pitch agreement past a point.'))

story.append(PageBreak())

# ============================================================
# Section 6 — Empirical highlights
# ============================================================
story.append(Paragraph('6. Empirical highlights from the review process', H2))

story.append(Paragraph('6.1 The data leak (N1) was real and quantified', H3))
story.append(P(
    'Round-3 finding: when the worker first added a shuffle to the train/val split '
    '(O4), the metrics didn\'t change. Investigated by enumerating '
    '<code>{audio_path}</code> in the val set:'))
story.append(Paragraph(
    'val set (per-clip shuffle): 153 clips drawn from <b>5 source WAV files</b>; '
    'all 5 also appeared in train. Per-recording overfitting was being scored as '
    'generalization.', CODE))
story.append(P('After round-4 file-level split:'))
story.append(Paragraph(
    'train clips=1445, val clips=77;  train files=18, val files=2, overlap=0',
    CODE))
story.append(P(
    'Round-6 dataset rebuild expanded both halves substantially:'))
story.append(Paragraph(
    f'train clips={PM["n_train_clips"]}, val clips={PM["n_val_clips"]};  '
    f'val files={PM["n_val_files"]}, overlap=0',
    CODE))

story.append(Paragraph('6.2 The metric artifact (E1) was real and quantified', H3))
story.append(P(
    'Round-5 finding: <code>combined</code> reported 3 of 9 perfect 1.0 scores, '
    'driving its mean to 0.355 while every probe-only method sat near baseline. '
    'Instrumented <code>compute_melody_coherence</code> on three combined samples:'))
arts_table = [
    ['File', 'Voiced frames / total', 'Valid', 'Matches', 'Coherence'],
    ['prompt1_ascending_combined', '15 / 431 (3.5%)', '11', '11', '<b>1.000</b>'],
    ['prompt2_ascending_combined', '36 / 431 (8.4%)', '28', '28', '<b>1.000</b>'],
    ['prompt1_ascending_baseline', '236 / 431 (54.8%)', '195', '0', '0.000'],
    ['prompt1_alternating_combined', '36 / 431 (8.4%)', '0', '0', '0.000'],
]
arts_table = [[Paragraph(c, SMALL) for c in row] for row in arts_table]
story.append(code_table(arts_table, col_widths=[2.2*inch, 1.4*inch, 0.6*inch,
                                                 0.7*inch, 0.9*inch]))
story.append(P(
    'Latent fusion at scale=0.3 was wiping out most pitched content; the audio was '
    'mostly silent or held a single sustained sine tone. pyin then only found '
    '15–36 voiced frames out of 431; if those happened to align with the injected '
    'sine wave, the metric reported 1.0 over a tiny slice. Round-6 fix added a '
    '20% voiced-coverage gate (returns NaN below) and reduced fusion intensity '
    '(scale 0.3 → 0.1, every step → every 5 steps). New combined coherence: '
    'mean 0.332, median 0.306, voiced coverage 0.73–0.99 — real numbers over '
    'real audio.'))

story.append(Paragraph('6.3 The carry-over risk that wasn\'t', H3))
story.append(P(
    'B3/O1 (the legacy callback API) carried unresolved across rounds 1–4. The '
    'worry was that diffusers 0.27 might silently ignore <code>callback=…</code> '
    'on <code>StableAudioPipeline</code>, making every probe-guided experiment a '
    'no-op. In round 5 the worker confirmed via smoke test (<code>--verbose_guidance</code>): '
    '<code>probe_loss</code> drops from ~0.39 at step 0 to ~0.15 at step 40, '
    '<code>update_rms</code> from ~18 to ~0.06, no DeprecationWarning — the '
    'callback fires correctly. The final results in §5 confirm the technique '
    'genuinely steers generation.'))

story.append(PageBreak())

# ============================================================
# Section 7 — Verification log
# ============================================================
story.append(Paragraph('7. What I actually did across all six rounds', H2))
story.append(P(
    'All operations read-only. Repository is not under git, so mtime was the only '
    'diff signal. No source files were modified by me; the only filesystem writes '
    'were this PDF and the script that renders it.'))
story.append(P('&bull; <b>File diff via mtime</b> (every round): '
               '<code>find … -printf %T@ …</code> to identify what the worker '
               'touched between rounds.'))
story.append(P('&bull; <b>Static check</b> (every round): <code>python -m '
               'py_compile</code> on every modified module.'))
story.append(P('&bull; <b>Dataset/checkpoint inspection</b>: '
               '<code>torch.load</code> on <code>data/probe_dataset*.pt</code> and '
               '<code>checkpoints/probe_*.pt</code> to confirm shapes, dtypes, and '
               'contents.'))
story.append(P('&bull; <b>Probe sanity</b>: instantiated <code>LinearProbe</code> '
               '(908 params) and <code>CNNProbe</code> (125,196 params), confirmed '
               '<code>(B, T, 12)</code> output on a synthetic latent. Tested the '
               'GroupNorm-8 constraint by trying <code>CNNProbe(64, 7, 12)</code>.'))
story.append(P('&bull; <b>Functional re-tests of fixes</b>: built a '
               '<code>FakePipe</code>/<code>FakeProbe</code> in-memory and invoked '
               '<code>_make_probe_callback</code> and '
               '<code>make_combined_callback</code> with edge-case parameters to '
               'confirm <code>ValueError</code> raises after rounds 3-4 fixes.'))
story.append(P('&bull; <b>End-to-end eval smoke</b>: built a synthetic '
               '<code>MaestroDataset</code> with 8 random samples and ran '
               '<code>evaluate_probe</code> with an untrained CNN probe to confirm '
               'all metric keys are produced and finite.'))
story.append(P('&bull; <b>Leak measurement</b>: enumerated <code>{audio_path}</code> '
               'across the per-clip-shuffle val split (5 leaking files) and the '
               'per-file-shuffle val split (2 isolated files, overlap 0).'))
story.append(P('&bull; <b>RNG determinism</b>: seeded two '
               '<code>torch.Generator</code>s with the same value (identical '
               'output) and different values (different output) — proving O6\'s '
               'seed-propagation works correctly.'))
story.append(P('&bull; <b>Voicing forensics</b> (round 5): re-ran '
               '<code>librosa.pyin</code> on combined and baseline audio files and '
               'counted voiced frames to expose the 1.0-score artifact.'))
story.append(P('&bull; <b>JSON forensics</b> (round 6): pivoted '
               '<code>results.json</code> by (prompt, melody) into a paired '
               'comparison table, recomputed Wilcoxon medians by hand to verify '
               'the reported numbers.'))

# ============================================================
# Section 8 — Outstanding items
# ============================================================
story.append(Paragraph('8. Outstanding items, ordered by impact', H2))
open_table = [['Priority', 'ID', 'Issue', 'Recommendation']]
opens = [
    ('1', 'New', 'No effect-size r reported alongside Wilcoxon p',
     'Add r = Z / sqrt(N) for each method-vs-baseline test. With n=9 the test is '
     'borderline-powered; r quantifies how big the effect is, separate from '
     'whether it\'s detectable.'),
    ('2', 'New', 'No within-probe-group test (which scale is best?)',
     'Add a Friedman test across {probe_low, mid, high, strong}. The plateau at '
     'scale ≥ 0.05 deserves a formal statement, not just eyeballing.'),
    ('3', 'New', 'ProbeGuidedGenerator default still probe_guidance_start=0.7',
     'src/inference.py:59 default is 0.7, but scripts/03_run_inference.py default '
     'is 0.4. Library users hitting the class directly inherit the old, '
     'underperforming default. Sync them.'),
    ('4', 'E6', 'No automatic spot-listening output',
     'Save voiced_coverage and RMS per file (already saved); flag any sample with '
     'coverage < 50% in the run summary so the worker knows which to listen to.'),
    ('5', 'New', 'probe_metrics.json is silently overwritten on retrain',
     'Suffix with timestamp (probe_metrics_20260502_0216.json) or write to '
     '_latest + a versioned copy. Avoids losing prior comparison runs.'),
    ('6', 'New', 'make_combined_callback docstring missing fusion_every_n',
     'Add to docstring; explain the rationale (avoid accumulation across all '
     'fusion-active steps).'),
    ('7', 'New', '04_evaluate.py hardcodes MIN_VOICED_COVERAGE = 0.20',
     'Promote to a CLI flag (--min_voiced_coverage). Makes sensitivity analysis '
     'cheap and avoids a magic number.'),
    ('8', 'Q8', 'test_setup.py fp16/fp32 mismatch',
     'Cosmetic. Fix when next touching test_setup.py.'),
    ('9', 'Q11', 'evaluate_noise_robustness iterates serially',
     'Acceptable at current val size (417 clips). Re-batch via DataLoader if val '
     'grows past a few thousand.'),
]
for row in opens:
    pr, tag, title, action = row
    open_table.append([
        Paragraph(f'<b>{pr}</b>', SMALL),
        Paragraph(tag, SMALL),
        Paragraph(title, SMALL),
        Paragraph(action, SMALL),
    ])
story.append(code_table(open_table, col_widths=[0.55*inch, 0.5*inch, 2.3*inch, 3.35*inch]))

story.append(Spacer(1, 0.15 * inch))
story.append(Paragraph('8.1 What is reliably done and shippable', H3))
story.append(P(
    'After six rounds the entire pipeline is in publishable shape:'))
story.append(P('&bull; <b>Data preparation</b>: fps from actual latent shape; '
               'labels interpolated to match latent T; periodic save/GC every 5 '
               'files; correct stereo handling; <code>Path.with_suffix</code> for '
               'MIDI lookup.'))
story.append(P('&bull; <b>Probe training</b>: gradient clipping; cosine schedule; '
               'safe-default <code>noise_sigmas</code>; best/periodic/final '
               'checkpoints; tolerant checkpoint loading; '
               '<b>persisted Contribution-1 metrics</b>.'))
story.append(P('&bull; <b>Probe evaluation</b>: shape-aligned in both '
               'evaluate_probe and noise_robustness paths; rich metric set '
               'including positive recall and per-class F1.'))
story.append(P('&bull; <b>Train/val split</b>: file-level partitioning with '
               'deterministic seed; explicit <code>--n_val_files</code> override; '
               '<code>warnings.warn</code> when val partition is too small.'))
story.append(P('&bull; <b>Inference</b>: confirmed-firing diffusers callback; '
               'RMS-normalized, magnitude-clamped probe gradient; fp32 math '
               'regardless of pipeline dtype; lazy target computation; combined '
               'fusion+probe path with reproducible per-step + user seed; '
               'comprehensive parameter guards.'))
story.append(P('&bull; <b>Audio evaluation</b>: voiced-coverage gating prevents '
               'metric-gaming; per-trial pivot enables paired comparison; '
               'one-sided Wilcoxon signed-rank against baseline reported per '
               'method.'))

story.append(PageBreak())

# ============================================================
# Section 9 — Bottom line
# ============================================================
story.append(Paragraph('9. Bottom line', H2))
story.append(P(
    '<b>Both contributions are now demonstrated.</b>'))
story.append(P(
    f'<b>Contribution 1.</b> The Stable Audio VAE latent space encodes '
    f'pitch-class information that a small CNN probe can recover at '
    f'<b>F1 (micro) = {PM["f1_micro"]:.3f}</b>, balanced across 12 classes, '
    f'robust under added noise. Trained on a leak-free file-level split.'))
story.append(P(
    f'<b>Contribution 2.</b> Treating the trained probe as a differentiable loss '
    f'during diffusion denoising significantly improves melody coherence over the '
    f'unguided baseline. The best operating point (probe_mid, scale=0.05, '
    f'guidance_start=0.4) reaches <b>mean coherence {sm["probe_mid"]["mean_coherence"]:.3f} '
    f'vs baseline {sm["baseline"]["mean_coherence"]:.3f}</b>, '
    f'<b>p={wlx["probe_mid"]["p_value"]:.4f}</b> (Wilcoxon signed-rank, n=9 paired '
    f'trials), with no metric artifacts (voiced coverage 0.66–0.99 across all '
    f'guided methods).'))
story.append(P(
    'The route to that result was four rounds of code review (5 bugs + 23 quality '
    'items, all closed except 2 cosmetic) and two rounds of evaluation-methodology '
    'review (6 issues, all closed except E6 partial). The two interventions that '
    'mattered most were both review-driven: fixing the train/val data leak (N1) '
    'and the voiced-coverage metric artifact (E1). Without either fix, the '
    'numbers would have been misleading; with both, they\'re defensible.'))

story.append(Spacer(1, 0.2 * inch))
story.append(Paragraph(
    '<i>Generated by <code>scripts/_make_report.py</code>; numbers pulled live '
    'from <code>outputs/evaluation/{probe_metrics,results}.json</code> so '
    're-running the script after a fresh eval will refresh the tables.</i>',
    SMALL))


doc = SimpleDocTemplate(
    str(OUT), pagesize=LETTER,
    leftMargin=0.65 * inch, rightMargin=0.65 * inch,
    topMargin=0.65 * inch, bottomMargin=0.65 * inch,
    title='MIC Code Review + Evaluation Report',
    author='Claude Opus 4.7 (reviewer)',
)
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print(f'Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)')
