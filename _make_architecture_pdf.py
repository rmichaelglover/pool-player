"""Generate pool_architecture.pdf summarizing the current Phase 7/8 pool AI architecture."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Preformatted, PageBreak, Table, TableStyle,
                                 ListFlowable, ListItem)
from reportlab.lib.enums import TA_LEFT

styles = getSampleStyleSheet()
H1 = ParagraphStyle(name='H1', parent=styles['Heading1'], fontSize=18,
                     spaceBefore=10, spaceAfter=10, textColor=HexColor('#1e3a8a'))
H2 = ParagraphStyle(name='H2', parent=styles['Heading2'], fontSize=14,
                     spaceBefore=14, spaceAfter=6, textColor=HexColor('#1e40af'))
H3 = ParagraphStyle(name='H3', parent=styles['Heading3'], fontSize=11,
                     spaceBefore=8, spaceAfter=4, textColor=HexColor('#1e40af'),
                     fontName='Helvetica-Bold')
BODY = ParagraphStyle(name='Body', parent=styles['BodyText'], fontSize=10,
                       leading=14, spaceAfter=6, alignment=TA_LEFT)
CODE = ParagraphStyle(name='Code', parent=styles['Code'],
                       fontName='Courier', fontSize=8.5, leading=10.5,
                       leftIndent=16, rightIndent=16,
                       textColor=HexColor('#111827'),
                       backColor=HexColor('#f3f4f6'),
                       borderPadding=6, borderColor=HexColor('#e5e7eb'),
                       borderWidth=0.5, spaceBefore=4, spaceAfter=8)
CAPTION = ParagraphStyle(name='Caption', parent=styles['BodyText'], fontSize=9,
                          textColor=HexColor('#6b7280'), alignment=TA_LEFT,
                          spaceAfter=10, fontName='Helvetica-Oblique')


def p(text, style=BODY):
    return Paragraph(text, style)


def tbl(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#dbeafe')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1e3a8a')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.3, HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


# ── Document ──────────────────────────────────────────────────────────────

doc = SimpleDocTemplate('pool_architecture.pdf', pagesize=letter,
                         leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                         topMargin=0.7 * inch, bottomMargin=0.7 * inch)
s = []

s.append(p('Pool AI — Current Architecture (Phase 7 / 8)', H1))
s.append(p('Token-based transformer actor-critic for 14.1 Continuous pool. '
           '566,918 parameters. Trained on CPU.', CAPTION))

# ── Overview ──
s.append(p('Design philosophy', H2))
s.append(p(
    'Pool shot-making has three layers: <b>geometric</b> (which balls can '
    'be pocketed into which pockets; line-of-sight checks; ghost-ball aim), '
    '<b>strategic</b> (which of the makeable shots to take, for position on '
    'the next shot), and <b>mechanical</b> (force and spin for cue ball '
    'control). We handle each layer with the right tool:'
))
s.append(ListFlowable([
    ListItem(p('<b>Geometric layer</b>: deterministic algorithm enumerates '
               'legal (ball, pocket) pairs. No learning needed.', BODY)),
    ListItem(p('<b>Strategic + mechanical layers</b>: learned by the '
               'network — picking among legal shots and choosing force/spin.', BODY)),
], bulletType='bullet'))
s.append(p(
    'This is the AlphaZero pattern adapted to pool: the solved parts are '
    'computed exactly; the parts that need judgment are learned.'
))

# ── Decision pipeline ──
s.append(p('Decision pipeline (per shot)', H2))
s.append(p('<b>1. Geometric enumeration</b> (Python, no learning). For the '
           'current cue and ball positions, enumerate all (ball, pocket) '
           'pairs where:'))
s.append(ListFlowable([
    ListItem(p('The <b>ghost-ball position</b> (where the cue ball\'s center '
               'must be at contact to send the object ball toward the pocket) '
               'is on the table — not behind a cushion.', BODY)),
    ListItem(p('The <b>cue → ghost</b> path is clear of other balls '
               '(line-of-sight with 2R + 0.3" clearance).', BODY)),
    ListItem(p('The <b>ball → pocket</b> path is clear of other balls.', BODY)),
    ListItem(p('The <b>cut angle</b> is ≤ 75° (very steep cuts discarded).', BODY)),
    ListItem(p('The cue is behind the ball relative to the pocket direction.', BODY)),
], bulletType='bullet'))
s.append(p('Output: a list of <i>LegalShot</i> records, each with ball_id, '
           'pocket_idx, ghost position, aim angle, cut angle, and distances.'))

s.append(p('<b>2. Network inference</b> on token batch:'))
s.append(p('The board state + legal shots become a token sequence fed to '
           'a transformer. Each token type has features:'))
s.append(tbl([
    ['Token type', 'Count', 'Features'],
    ['Ball', 'up to 16', '(x, y, is_cue)'],
    ['Pocket', '6', '(x, y, is_corner)'],
    ['Legal shot', 'up to 60', '(ghost_x, ghost_y, target_ball_x, target_ball_y, target_pocket_x, target_pocket_y, cut_angle_deg/90, cue→ghost_dist/TL, ball→pocket_dist/TL)'],
], col_widths=[0.9 * inch, 0.7 * inch, 4.6 * inch]))

s.append(p('<b>3. Action sampling</b>:'))
s.append(ListFlowable([
    ListItem(p('<b>Shot index</b>: categorical over legal shots '
               '(softmax of per-shot score logits).', BODY)),
    ListItem(p('<b>Force</b>: Gaussian sample around the chosen shot\'s force_mean; '
               'sigmoid to [50, 250] in/s.', BODY)),
    ListItem(p('<b>Spin</b>: Gaussian sample around chosen shot\'s spin_mean; '
               'tanh to [-2, +2] (spin factor relative to natural roll).', BODY)),
    ListItem(p('<b>Aim</b>: derived geometrically from the chosen shot\'s ghost — '
               'no learning needed.', BODY)),
], bulletType='bullet'))

s.append(PageBreak())

# ── Network ──
s.append(p('Network: PoolGameNet', H2))
s.append(p('Total parameters: <b>566,918</b>. Implementation in '
           '<code>rl/pool_game_net.py</code>.'))

s.append(p('Encoders', H3))
s.append(Preformatted(
    'ball_encoder:   Linear(3 → 128) → LayerNorm → GELU\n'
    'pocket_encoder: Linear(3 → 128) → LayerNorm → GELU\n'
    'shot_encoder:   Linear(9 → 128) → LayerNorm → GELU\n'
    'type_embed:     Embedding(3 types → 128), added to each token',
    CODE))

s.append(p('Transformer', H3))
s.append(Preformatted(
    'TransformerEncoder(\n'
    '    layers        = 4,\n'
    '    embed_dim     = 128,\n'
    '    num_heads     = 8,\n'
    '    ff_dim        = 256,    # 2 × embed_dim\n'
    '    activation    = GELU,\n'
    '    norm_first    = True,   # pre-LN for stable training\n'
    ')\n'
    '# Padding mask hides invalid ball/shot slots from attention.',
    CODE))

s.append(p('Output heads (per legal shot)', H3))
s.append(Preformatted(
    'shot_head: Linear(128 → 128) → GELU → Linear(128 → 3)\n'
    '           # outputs (score_logit, force_mean, spin_mean) per shot\n'
    '           # softmax over all valid shots → which shot to take\n'
    'log_std:   nn.Parameter of shape (2,) for (force_std, spin_std)\n'
    '           # clamped at log_std_min = -2.5',
    CODE))

s.append(p('Value head (state value)', H3))
s.append(Preformatted(
    'value_head: Linear(128 → 128) → GELU → Linear(128 → 1)\n'
    '            # input is mean-pool over ball + pocket tokens\n'
    '            # shot tokens excluded from the pooling',
    CODE))

# ── Training ──
s.append(p('Training', H2))
s.append(p(
    'Standard PPO with joint categorical-continuous action space:'
))
s.append(Preformatted(
    'log_prob = log_softmax(scores)[shot_idx]\n'
    '         + Normal(force_mean[shot_idx], force_std).log_prob(force_raw)\n'
    '         + Normal(spin_mean[shot_idx],  spin_std).log_prob(spin_raw)\n\n'
    'entropy  = Categorical(scores).entropy()\n'
    '         + Normal(0, force_std).entropy()\n'
    '         + Normal(0, spin_std).entropy()\n\n'
    'loss = pg_loss + 0.5 · v_loss − 0.01 · entropy',
    CODE))

s.append(p('Default hyperparameters', H3))
s.append(tbl([
    ['Hyperparameter', 'Phase 7 value', 'Phase 8 value'],
    ['Optimizer', 'Adam', 'Adam'],
    ['Learning rate', '1e-4', '3e-5 (gentler, avoid catastrophic forgetting)'],
    ['Env parallelism', '16', '16'],
    ['Steps per update', '32', '32'],
    ['PPO epochs / clip', '4 / 0.2', '4 / 0.2'],
    ['Value coef', '0.5', '0.5'],
    ['Entropy coef', '0.01', '0.01'],
    ['log_std min', '-2.5', '-2.5'],
    ['Iters', '500–1500', '500'],
], col_widths=[1.8 * inch, 1.4 * inch, 2.8 * inch]))

s.append(p('Environments', H3))
s.append(ListFlowable([
    ListItem(p('<b>Phase 7 (Phase7Env)</b>: full 15-ball rack; opening break '
               'auto-executed in reset(); agent takes over from shot 2. '
               'Includes rerack mechanic.', BODY)),
    ListItem(p('<b>Phase 8 (Phase8Env)</b>: break-ball drill. Rack is 14 '
               'balls (apex empty) + break ball placed in classic 14.1 '
               'positions + cue in kitchen. Agent\'s first shot is the break. '
               'If break succeeds, continues as full run-out (self-balancing '
               'training distribution).', BODY)),
], bulletType='bullet'))

s.append(p('Reward structure', H3))
s.append(ListFlowable([
    ListItem(p('<b>+10</b> per object ball pocketed on a shot (counts '
               'incidentals when the called shot succeeds).', BODY)),
    ListItem(p('<b>-10</b> scratch penalty (cue pocketed).', BODY)),
    ListItem(p('Episode ends on miss (called ball not in called pocket), '
               'scratch, or max_shots.', BODY)),
    ListItem(p('The opening break is exempt from strict call-shot — any '
               'pocketed ball counts, matching real 14.1 break rules.', BODY)),
], bulletType='bullet'))

s.append(PageBreak())

# ── Legal shot enumeration (detail) ──
s.append(p('Legal-shot enumeration (critical detail)', H2))
s.append(p(
    'This is the algorithmic heart of the system. Done right, it ensures '
    'the network only ever picks from physically realizable shots.'
))

s.append(p('Ghost ball geometry', H3))
s.append(Preformatted(
    'ghost = ball − 2R · (pocket − ball) / |pocket − ball|\n'
    '# 2R-offset from ball, away from the pocket direction.\n'
    '# This is where the cue ball center must be at the moment of contact\n'
    '# to send the object ball straight toward the pocket center.',
    CODE))

s.append(p('Line-of-sight check', H3))
s.append(Preformatted(
    'def segment_blocked(p1, p2, obstacles, exclude_ids, clearance=2R+0.3):\n'
    '    # Parameterize segment from p1 to p2.\n'
    '    for each obstacle b not in exclude_ids:\n'
    '        t = projection of (b − p1) onto segment direction\n'
    '        if t < 0:   continue          # obstacle is behind p1\n'
    '        if t > seg_len:\n'
    '            if |b − p2| < clearance: return BLOCKED   # near endpoint\n'
    '        else:\n'
    '            if perp(b to segment) < clearance: return BLOCKED\n'
    '    return CLEAR',
    CODE))
s.append(p('<b>Subtle detail</b>: obstacles behind the cue (t &lt; 0) are '
           'NOT considered blockers. A ball touching the cue ball on the '
           'back side does not prevent the cue from moving forward. This '
           'was a bug in an earlier version — fixed.'))

s.append(p('Cut angle constraint', H3))
s.append(p('The cut angle is the angle between the cue-approach direction '
           '(toward ghost) and the intended ball-exit direction (toward '
           'pocket). Shots with cut &gt; 75° are discarded — they require '
           'too-thin hits to be reliable.'))

# ── Action space ──
s.append(p('Action space details', H2))
s.append(tbl([
    ['Component', 'Type', 'Raw range', 'After activation'],
    ['shot_idx', 'categorical', '{0..N_legal−1}', 'chosen shot from legal set'],
    ['force_raw', 'continuous', '(−∞, ∞)', 'sigmoid → [50, 250] in/s'],
    ['spin_raw', 'continuous', '(−∞, ∞)', 'tanh → [−2, +2] (relative to natural roll)'],
    ['aim_angle', '—', 'derived', 'atan2 toward ghost of chosen shot'],
], col_widths=[1.0 * inch, 1.0 * inch, 1.2 * inch, 2.8 * inch]))
s.append(p('<b>Spin interpretation</b>: '
           '<code>spin_factor</code> = ω_initial / (v_initial / R). '
           '0 = pure linear (develops natural roll via sliding friction); '
           '+1 = natural roll from the start; +2 = max follow; −1 = full '
           'backspin; −2 = max draw.'))

# ── History / what we learned ──
s.append(p('What led here (brief)', H2))
s.append(ListFlowable([
    ListItem(p('<b>Phases 1–6</b>: flat-observation transformer '
               '(<i>PoolAttentionNet</i>, 438K params). Learned aim via '
               'continuous aim output. Plateaued because implicit ball '
               'selection via aim angle + sparse reward couldn\'t teach '
               'strategic shot choice from scratch.', BODY)),
    ListItem(p('<b>Key insight</b>: "which ball to shoot" is a solved '
               'geometric problem; only "which of the makeable shots, with '
               'what force and spin" needs learning. This is the AlphaZero '
               'insight applied to pool.', BODY)),
    ListItem(p('<b>Phase 7</b>: rebuilt with token-based network and '
               'legal-shot enumeration as a pipeline stage. Network picks '
               'among enumerated shots rather than generating aim '
               'directly. Immediate jump in play quality.', BODY)),
    ListItem(p('<b>Phase 8</b>: break-ball drill env. Self-balancing curriculum: '
               'if break succeeds, episode continues as a run-out, so good '
               'breaks are rewarded by cascading return, bad breaks end '
               'immediately. Warm-started from Phase 7 and fine-tuned at '
               'low lr to preserve general-play skills.', BODY)),
], bulletType='bullet'))

s.append(p('Open areas', H2))
s.append(ListFlowable([
    ListItem(p('<b>Search at inference</b> (AlphaZero-style MCTS over shots '
               '× force/spin samples). Currently inference uses deterministic '
               'argmax + mean output. Adding tree search should boost play.', BODY)),
    ListItem(p('<b>Scaling network capacity</b>: 566K may not be the '
               'bottleneck yet — we\'ll know by whether plateau metrics '
               'improve with 1–2M parameters. No move until warranted.', BODY)),
    ListItem(p('<b>Specialized drills</b>: Phase 8 is a break-ball drill; '
               'similar drills for "save the break ball" endgame, '
               'cluster-management, and safety play would further sharpen '
               'specific strategic concepts.', BODY)),
], bulletType='bullet'))

# Build
doc.build(s)
print('Wrote pool_architecture.pdf')
