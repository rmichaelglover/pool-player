"""Generate pool.pdf containing the recent session conversation."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak
from reportlab.lib.enums import TA_LEFT

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='User', parent=styles['BodyText'],
    backColor=HexColor('#eef2ff'), borderColor=HexColor('#6366f1'),
    borderWidth=0, leftIndent=8, rightIndent=8, spaceBefore=10, spaceAfter=4,
    textColor=HexColor('#1e1b4b'), fontName='Helvetica-Bold', fontSize=10))
styles.add(ParagraphStyle(name='Assistant', parent=styles['BodyText'],
    leftIndent=8, rightIndent=8, spaceBefore=2, spaceAfter=8, fontSize=10,
    leading=13))
styles.add(ParagraphStyle(name='Role', parent=styles['Heading4'],
    spaceBefore=10, spaceAfter=2, fontSize=11,
    textColor=HexColor('#334155')))
code_style = ParagraphStyle(name='Code', parent=styles['Code'],
    fontName='Courier', fontSize=8, leading=10, leftIndent=16,
    textColor=HexColor('#111827'), backColor=HexColor('#f3f4f6'))


def p(text, style):
    return Paragraph(text, style)


def role(name):
    return p(f'<b>{name}</b>', styles['Role'])


def body(text):
    return p(text, styles['Assistant'])


def code(text):
    return Preformatted(text, code_style)


story = []
story.append(Paragraph('Pool Player RL — Session Transcript', styles['Title']))
story.append(Paragraph('Date: 2026-04-20', styles['Normal']))
story.append(Paragraph(
    'Phase 1 curriculum diagnosis → fix → Phase 2 pocketing',
    styles['Italic']))
story.append(Spacer(1, 0.2 * inch))

# ─── Turn 1 ────
story.append(role('User'))
story.append(body("Let's continue with the pool player project. We were testing Phase 1 curriculum training for the attention network."))

story.append(role('Assistant'))
story.append(body(
    "Per notes, Phase 1 curriculum (1 ball close, 2 outputs) on the 438K transformer stalled at 2–6% hit rate — below the 15% random baseline after 130 iters. "
    "Diagnosis: transformer likely overkill for a 1-ball task where 21 of 22 tokens are masked/irrelevant. "
    "Proposed next step: build a tiny MLP (38→64→64→2) to verify the aim task + reward structure IS learnable. If MLP crosses 50%+, it's transformer overhead. If MLP also stalls, the reward/task is broken."))

# ─── Turn 2 ────
story.append(role('User'))
story.append(body("proceed with MLP test"))

story.append(role('Assistant'))
story.append(body(
    "Built <b>train_curriculum_mlp.py</b> with a 6.8K-param TinyMLP (38→64→64→actor(2)+critic(1), Tanh), reusing Phase1Env and the PPO loop. "
    "Ran 500 iters on CPU."))
story.append(code(
    "Result: avg hit rate 1.0–1.4% over 500 iters.\n"
    "Entropy dropped 1.83 → 1.39 (learning IS happening, but aim is getting worse).\n"
    "WORSE than the 15% claimed random baseline."))
story.append(body(
    "Key clue: at iter 10 (near init), hit rate is already 1.5%. That means the random-baseline assumption is wrong. "
    "Hypotheses: (1) env never resets after a hit — ball drifts, (2) (sin,cos)→arctan2 is a many-to-one mapping that mangles gradients, (3) reward shape is wrong. "
    "Proposed running a random-policy baseline script to test hypothesis #1 directly."))

# ─── Turn 3 ────
story.append(role('User'))
story.append(body("yes, continue"))

story.append(role('Assistant'))
story.append(body("Wrote <b>baseline_phase1.py</b> — measures random-policy hit rate in 4 conditions (2 action modes × 2 env modes)."))
story.append(code(
    "  action=uniform  env=reset_each  -> 14.6%\n"
    "  action=uniform  env=current     -> 1.8%\n"
    "  action=sincos   env=reset_each  -> 14.1%\n"
    "  action=sincos   env=current     -> 0.8%"))
story.append(body(
    "<b>Hypothesis #1 confirmed decisively.</b> The env is the entire problem. (sin,cos) parametrization is fine — matches uniform within noise. "
    "Fix: one-line change in train_curriculum.py:124, <font face='Courier'>done = pocketed or hit</font>. Offered to patch and re-run."))

# ─── Turn 4 ────
story.append(role('User'))
story.append(body("yes, continue"))

story.append(role('Assistant'))
story.append(body(
    "Applied the patch (later upgraded to <font face='Courier'>done = True</font> always after finding a second drift bug: cue ball moves on every shot, including misses). "
    "Relaunched both MLP and transformer. MLP still at 1.6% — didn't fix it. But transformer broke through:"))
story.append(code(
    "Transformer v2 with patch (300 iters):\n"
    "  Iter  10: 2.1% HR\n"
    "  Iter 100: 11.0%\n"
    "  Iter 200: 17.2%\n"
    "  Iter 300: 35.9% (climbing, AvgHR=32.8%)"))
story.append(body("So the patch worked for the transformer. MLP was still suspicious but deprioritized."))

# ─── Turn 5 ────
story.append(role('User'))
story.append(body("yes, continue and continue running autonomously to improve the results. You are making the correct decisions so just continue and do not stop making improvements. I will be back in about 3 hours to check on your progress."))

story.append(role('Assistant'))
story.append(body("[Autonomous block]"))
story.append(body(
    "• Launched transformer 2000-iter extended run (<font face='Courier'>curriculum_transformer_v3_long.log</font>).<br/>"
    "• Launched MLP variant with ReLU + hidden=128 + 2000 iters in parallel.<br/>"
    "• Built <b>eval_phase1.py</b> for deterministic checkpoint evaluation.<br/>"
    "• Built <b>train_phase2.py</b> — pocketing env places ball 2–8in from a random pocket, cue 6–24in behind on the line through. Straight aim scores. Smoke test: perfect aim 100% pocket, random aim 4.9% pocket. Geometry validated.<br/>"
    "• Ran eval on early checkpoint: 65.4% deterministic HR, 18° median aim error — confirmed stochastic training HR was underestimating skill."))
story.append(code(
    "Transformer final (2000 iters, ~45 min CPU):\n"
    "  Best avg HR 97.8%, Deterministic eval: 100.0% HR, 5.4 deg median error.\n"
    "  >>> Phase 1 MASTERED. <<<\n\n"
    "MLP ReLU128 (plot twist: it DID work):\n"
    "  Iter  500: 67.7% HR\n"
    "  Iter 1050: 98.7% HR PEAK (entropy -1.06)\n"
    "  Iter 1650: COLLAPSED to ~1% HR (entropy -1.58)\n"
    "Classic PPO overconfident collapse. MLP best checkpoint still valid."))
story.append(body(
    "Backed up phase1 checkpoint to <font face='Courier'>phase1_best_100pct.pt</font>. Launched Phase 2 with warm start."))

# ─── Turn 6 ────
story.append(role('User'))
story.append(body("Back, it appears things are going well."))

story.append(role('Assistant'))
story.append(body("Summary delivered: Phase 1 mastered (100% det HR), MLP puzzle solved (Tanh+64 broken; ReLU+128 reaches 98% then collapses), Phase 2 warm-starting at 42.8% pocket at iter 30."))

# ─── Turn 7 ────
story.append(role('User'))
story.append(body("is the transformer the same size we will use when running on the H100?"))

story.append(role('Assistant'))
story.append(body(
    "Yes — 438K params (embed_dim=96, heads=6, layers=4). Same architecture in curriculum, deploy_gpu, and the class default. Only difference is act_dim=2 for Phase 1 vs act_dim=5 for the full-action GPU run. "
    "Noted that H100 can comfortably run 5–10M for Phase 4+ where multi-ball sequence planning needs richer representations."))

# ─── Turn 8 ────
story.append(role('User'))
story.append(body("yes, agreed. Prove out the curriculum with a smaller size transform on my machine and then switch over to the H100 with a larger size to capture sequence planning."))

story.append(role('Assistant'))
story.append(body(
    "Saved <b>project_scaling_plan.md</b> memory: 438K CPU for phases 1–3; upsize to 5–10M H100 at Phase 3→4 transition. Added MEMORY.md index entry."))

# ─── Turn 9 ────
story.append(role('User'))
story.append(body("how's it going?"))

story.append(role('Assistant'))
story.append(body("Phase 2 v1 collapsed similarly to the MLP: peaked at 76.7% avg pocket at iter ~470, then entropy plunged to -7.4 and skill degraded to 26%. Killed it, backed up peak checkpoint to <font face='Courier'>phase2_best_76pct.pt</font>."))
story.append(body(
    "Root-cause: no log_std floor. Added collapse prevention in train_phase2.py:<br/>"
    "• <font face='Courier'>log_std_min = -1.5</font> (std ≥ 0.22)<br/>"
    "• <font face='Courier'>entropy_coef</font>: 0.001 → 0.01<br/>"
    "• <font face='Courier'>net.log_std.clamp_(min=log_std_min)</font> after each opt.step()<br/>"
    "Relaunched Phase 2 v2 with fresh warm-start from Phase 1."))

# ─── Turn 10 ────
story.append(role('User'))
story.append(body(
    "look at Andre Karpathy's autoresearch strategy, in particular program.md, at /home/r-m-glover/claude_projects/autoresearch. "
    "Would this approach help us, or do you think that I need to participate in the decisions. I suppose we could try running it in parallel. what do you think?"))

story.append(role('Assistant'))
story.append(body(
    "Read both README.md and program.md. Karpathy's design is optimized for a specific shape of research problem that differs from ours:"))
story.append(body(
    "<b>Where autoresearch fits:</b> Single flat metric (val_bpb), 5-min experiments → ~100 trials overnight, local search in hyperparameter space, single-file edits."))
story.append(body(
    "<b>Where it doesn't fit us yet:</b><br/>"
    "1. Most of today's wins were <i>bug-finding</i>, not tuning. Env drift bug, Tanh+64 unlearnable, PPO collapse — diagnosing these needed diagnostic scripts and reading intermediate trajectories, not change-file/measure/revert loops.<br/>"
    "2. Our experiments are 30–45 min, not 5 — ~10 trials overnight, slow feedback.<br/>"
    "3. Our metric is phase-dependent, and an autonomous loop could accidentally clobber Phase 1's warm-start checkpoint while tuning Phase 2.<br/>"
    "4. PPO collapse wouldn't be caught by final-score-only metric; we had to watch intermediate peaks."))
story.append(body(
    "<b>Where it WOULD pay off in parallel:</b> as a narrow subroutine once the env/architecture is validated. Example: once Phase 2 env is fixed, sweep log_std_min / entropy_coef / LR / clip_eps. That's autoresearch's sweet spot."))
story.append(body("Recommendation: human-in-loop for phase-level decisions; spawn narrow autoresearch-style sub-agents for hyperparameter sweeps within a phase."))

# ─── Turn 11 ────
story.append(role('User'))
story.append(body("sounds good."))

story.append(role('Assistant'))
story.append(body("Phase 2 v2 status check: log_std floor holding at -0.162, no collapse. Pocket rate climbing steadily: 66% → 72% over iters 310–350. Matching v1's peak without instability."))

# ─── Turn 12 ────
story.append(role('User'))
story.append(body("sounds good."))

story.append(role('Assistant'))
story.append(body("Standing by. Will report when Phase 2 v2 finishes."))

# ─── Turn 13 ────
story.append(role('User'))
story.append(body("any updates"))

story.append(role('Assistant'))
story.append(body("Plateaued at 73.3% avg pocket / 96.9% HR since iter 1000. Entropy pinned at floor (-0.162). The floor prevented collapse but caps peak accuracy — std=0.22 always adds noise per shot. Deterministic eval would show true skill higher. ~310 iters remaining."))

# ─── Turn 14 ────
story.append(role('User'))
story.append(body("in our case, how is back prop working, what is the function used."))

story.append(role('Assistant'))
story.append(body("The loss backprop'd in our PPO setup:"))
story.append(code("loss = pg_loss + value_coef * v_loss - entropy_coef * entropy.mean()"))
story.append(body("Three differentiable terms:"))
story.append(body("<b>1. pg_loss — PPO-clipped policy gradient</b> (main signal):"))
story.append(code(
    "ratio  = exp(new_log_prob - old_log_prob)\n"
    "surr1  = ratio * advantage\n"
    "surr2  = clamp(ratio, 1-0.2, 1+0.2) * advantage\n"
    "pg_loss = -min(surr1, surr2).mean()"))
story.append(body(
    "For Gaussian policy, log_prob(a) = -½((a-μ)/σ)² - log(σ) - ½log(2π), summed across action dims. "
    "∂log_prob/∂μ flows back through the actor head → transformer → obs. "
    "Positive advantage pulls μ toward the sampled action; negative pushes away. "
    "The clip stops over-large updates when ratio drifts too far from 1."))
story.append(body("<b>2. v_loss — value function MSE:</b> (values - returns)² trains the critic head."))
story.append(body("<b>3. entropy bonus</b> (negated so maximizing entropy reduces loss): entropy = ½log(2πe·σ²) per dim. Only depends on log_std. Pushes σ up."))
story.append(body(
    "<b>Backward path:</b> loss.backward() → actor/critic heads + log_std → shared trunk (game encoder + transformer + entity encoders). "
    "Everything trainable except the fixed pocket feature buffer. clip_grad_norm_(0.5) caps global L2 norm. Adam (lr=3e-4, eps=1e-5). "
    "The env itself is NOT differentiable — the C physics sim is a black box; all learning happens through advantages weighting log-probs."))

# ─── Phase 2 completion notification ────
story.append(role('System'))
story.append(body("Background task <i>Phase 2 v2 with log_std floor and entropy bump</i> completed (exit 0)."))

story.append(role('Assistant'))
story.append(body("Phase 2 v2 final: best avg pocket rate <b>77.6%</b> (vs v1's 76.7%, no collapse). Entropy held at floor throughout. Wrote eval_phase2.py."))

# ─── Turn 15 ────
story.append(role('User'))
story.append(body("what is definition and value of the term advantage"))

story.append(role('Assistant'))
story.append(body("<b>Advantage</b> is the core signal in policy gradient methods:"))
story.append(code("A(s, a) = Q(s, a) - V(s)"))
story.append(body("\"How much better was taking action a in state s than the average action the policy would have taken there.\" Positive = \"do more of this\"; negative = \"do less\"."))
story.append(body("In our code (RolloutBuffer.compute_returns) we use GAE with γ=0.99, λ=0.95:"))
story.append(code(
    "delta_t     = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)\n"
    "A_t         = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}\n"
    "return_t    = A_t + V(s_t)    # critic trained against this"))
story.append(body(
    "<b>λ</b> trades bias vs. variance: 0 = pure 1-step TD (biased by V), 1 = Monte Carlo (unbiased, high variance). 0.95 is standard. "
    "Since Phase 1/2 is bandit (done=True every step), GAE degenerates to A_t = r_t - V(s_t)."))
story.append(body("<b>Typical Phase 2 values:</b>"))
story.append(code(
    "Pocket:     r=10, V≈1.5  ->  A ≈ +8.5   (huge \"do more\" signal)\n"
    "Hit only:   r≈1-2.5, V≈1.5  ->  A ≈ -0.5 to +1 (mild)\n"
    "Miss:       r≈[-0.5,0], V≈1.5  ->  A ≈ -2 (mild negative)"))
story.append(body(
    "Then <b>normalized per batch:</b> A ← (A - mean) / (std + 1e-8). "
    "So advantages fed to pg_loss are unitless z-scores around 0, which is why PPO is robust to reward scaling."))

# ─── Final turn ────
story.append(role('User'))
story.append(body("create a pdf called pool.pdf and add this recent conversation to it."))

story.append(role('Assistant'))
story.append(body("Generated this PDF via reportlab (pandoc/latex weren't installed). Saved to pool.pdf in the project root."))

# Build
doc = SimpleDocTemplate(
    '/home/r-m-glover/claude_projects/pool_player/pool.pdf',
    pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    title='Pool Player RL — Session Transcript 2026-04-20',
)
doc.build(story)
print('pool.pdf written to', doc.filename)
