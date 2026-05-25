"""Generate the Pool AI demo writeup PDF for sharing with a software-nerd
pool buddy. Self-contained; uses fpdf2."""
from fpdf import FPDF
from datetime import date


URL = "https://all-newsletter-gentleman-leg.trycloudflare.com"
DEMO_PORT = 8001
GENERATED_DATE = "2026-05-23"


class WriteupPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120)
        self.cell(0, 8, "Pool AI - 14.1 Continuous demo writeup",
                  border=0, align="L")
        self.ln(10)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, f"page {self.page_no()}", align="C")
        self.set_text_color(0)

    def h1(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 18)
        self.cell(0, 10, text, ln=True)
        self.ln(1)

    def h2(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 7, text, ln=True)
        self.ln(1)

    def h3(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 6, text, ln=True)
        self.ln(0.5)

    def para(self, text):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(0, 5.5, text)
        self.ln(1.5)

    def bullets(self, items):
        self.set_font("Helvetica", "", 10.5)
        for item in items:
            self.set_x(self.l_margin + 4)
            self.cell(4, 5.5, "-")
            self.multi_cell(0, 5.5, item)
        self.ln(1)

    def codebox(self, lines):
        self.set_font("Courier", "", 9.5)
        self.set_fill_color(245, 245, 240)
        for ln in lines:
            self.set_x(self.l_margin)
            self.cell(0, 5, ln, ln=True, fill=True)
        self.set_font("Helvetica", "", 10.5)
        self.ln(1.5)

    def kvtable(self, rows, label_w=55):
        # Simple definition-list style: label on its own line bold, value
        # below it indented.  Avoids cell-then-multi_cell horizontal layout
        # issues in fpdf2.
        for k, v in rows:
            self.set_font("Helvetica", "B", 10.5)
            self.cell(0, 5.5, k, ln=True)
            self.set_font("Helvetica", "", 10.5)
            self.set_x(self.l_margin + 4)
            self.multi_cell(0, 5.5, v)
            self.ln(0.5)
        self.ln(0.5)


pdf = WriteupPDF(format="letter")
pdf.set_margins(left=18, top=18, right=18)
pdf.set_auto_page_break(True, margin=18)
pdf.add_page()


# --- Title ---
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 12, "Pool AI - 14.1 Continuous", ln=True)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(80)
pdf.cell(0, 6,
         f"A reinforcement-learning pool player.  Demo writeup, {GENERATED_DATE}.",
         ln=True)
pdf.set_text_color(0)
pdf.ln(4)


# --- Demo link box ---
pdf.set_fill_color(235, 240, 250)
pdf.set_draw_color(160, 180, 220)
y0 = pdf.get_y()
pdf.rect(pdf.l_margin, y0, 210 - 2 * pdf.l_margin, 22, "DF")
pdf.set_xy(pdf.l_margin + 3, y0 + 2)
pdf.set_font("Helvetica", "B", 11)
pdf.cell(0, 6, "Play the demo", ln=True)
pdf.set_x(pdf.l_margin + 3)
pdf.set_font("Helvetica", "", 10.5)
pdf.cell(0, 5, "Just open this URL in any browser (phone, tablet, laptop):", ln=True)
pdf.set_x(pdf.l_margin + 3)
pdf.set_font("Courier", "B", 10.5)
pdf.set_text_color(40, 60, 140)
pdf.cell(0, 6, URL, ln=True, link=URL)
pdf.set_text_color(0)
pdf.set_y(y0 + 22 + 4)


# --- 1. What it is ---
pdf.h1("1. What it is")
pdf.para(
    "A pool AI that plays Straight Pool (14.1 Continuous), trained with "
    "reinforcement learning rather than written as hand-tuned heuristics.  "
    "It pockets balls, manages clusters, plans end-of-rack sequences (key ball "
    "-> break ball -> scatter the rack), and runs multiple racks in succession.  "
    "First-try run on the live demo as of writing: 86 balls."
)
pdf.para(
    "The aim was not just to be accurate at individual shots, but to learn "
    "the *game* - decide between equally pocketable shots based on what comes "
    "next, preserve break balls, set up clean continuations after reracking.  "
    "Several recognizable 14.1 strategic patterns emerged without being "
    "explicitly programmed."
)


# --- 2. Architecture ---
pdf.h1("2. Architecture (simplified)")
pdf.para(
    "Three pieces.  Each shot decision flows top-to-bottom:"
)

pdf.h3("(a) Shot enumerator - pure geometry, not learned")
pdf.para(
    "Given current ball positions and the cue ball, generates the list of legal "
    "direct shots (ball -> pocket pairs) via line-of-sight and ghost-ball "
    "computation.  Blocking checks, cushion-corridor clearance, max cut angle "
    "(80 degrees).  No network involved here - shot legality is a solved "
    "geometric problem and forcing the network to learn it wastes capacity."
)

pdf.h3("(b) Transformer network - the policy and value head")
pdf.bullets([
    "567K parameters, embed=128, heads=8, layers=4.",
    "Inputs are tokens: one per ball on the table, one per pocket, one per legal-shot candidate.",
    "Outputs: (1) policy logits over the legal shot candidates, (2) per-shot force and spin means, (3) a scalar value estimate of the current state.",
    "All-CPU.  Runs in single-digit milliseconds per forward pass.",
])

pdf.h3("(c) Depth-1 search - inference-time lookahead")
pdf.para(
    "At each decision, the search takes the top-K shots by policy probability, "
    "simulates each one forward in the physics engine, and re-ranks by "
    "Q = immediate_reward + gamma * V(next_state).  K=4 in production.  "
    "This is what unlocks the strategic depth - the bare policy plays okay, but "
    "search is what surfaces 'pocket this ball AND clear a blocker AND leave "
    "the cue here' as preferable to 'pocket the easiest shot.'  About 5-40 ms "
    "per decision."
)

pdf.h3("Data flow")
pdf.codebox([
    "  state (balls + cue)",
    "         |",
    "         v",
    "  shot_enumerator  ->  list of legal shots",
    "         |",
    "         v",
    "  transformer  ->  policy probs, force/spin per shot, V(state)",
    "         |",
    "         v",
    "  search: for top-K shots, run physics rollout,",
    "          score by  imm_reward + gamma * V(next)",
    "         |",
    "         v",
    "  best-Q (shot, force, spin)",
    "         |",
    "         v",
    "  physics engine executes the shot",
])


# --- 3. Training ---
pdf.h1("3. How it was trained")
pdf.para(
    "Training was a long sequence of incremental versions, each addressing a "
    "specific weakness observed in the previous one.  Two phases of training "
    "method, both warm-starting from the previous best checkpoint:"
)
pdf.bullets([
    "Phases 1-3 (early): Standard PPO on simpler sub-tasks - aim, then pocket straight-in, then cut angles.  Built up the basic shot mechanics.",
    "Phase 7 (current): Token-based transformer trained with PPO, then AlphaZero-style distillation - run search during training, use the search-chosen action as the supervised target for the policy.  This closed most of the gap between 'bare policy' and 'policy + search' play.",
])

pdf.h3("Reward shaping (additive)")
pdf.bullets([
    "+10 per ball pocketed (pocket_reward).",
    "+5 (eor_bonus) for preserving the better break-ball candidate at n=2 balls remaining; -5 for pocketing it instead.  Half-strength at n=3.",
    "+10 (post-rerack scatter bonus) if the break shot scatters 3+ balls.",
    "+1.5 (next_shape_bonus_max) scaled by the ease of the easiest legal next shot - 'leave good shape.'",
    "-(up to 6) shape_bonus_max penalty proportional to current shot's difficulty (cut angle + total distance).  Discourages 78-degree-cut heroics when easier shots exist.",
    "+1 per ball pocketed that was within 3 inches of a cushion (rail_shot_bonus) - counters learned aversion to rail shots.",
    "Small penalties for cue-ball travel distance, multi-ball ricochets (encourages clean isolation).",
])

pdf.h3("Curriculum drills (env mix)")
pdf.para(
    "About 40 percent of training episodes are short curriculum drills "
    "instead of full racks:"
)
pdf.bullets([
    "Rail-frozen drill: single object ball within 1.5 inches of a rail - trains rail-shot mechanics.",
    "3-ball end-of-rack drill: key1 + key2 + break-ball - trains the EOR sequencing pattern.",
    "Key-and-break drill: classic curriculum scenario for the rack-opening break.",
    "Rail-ball break drill (newest, in v23): 14 reracked balls + 1 break ball near a rail + cue ready for the break - targets a specific failure mode where the network uses weak force when the break ball is on a rail.",
])


# --- 4. Emergent strategies ---
pdf.h1("4. Strategies that emerged without being told")
pdf.para(
    "These behaviors arose from the combination of long-horizon rewards (run "
    "length, EOR, post-break scatter) and search at inference, not from "
    "specific reward shaping for the behavior:"
)
pdf.bullets([
    "Side-pocket break setup.  Network leaves the break ball mid-table when possible, then pockets it into a side pocket while sending the cue ball into the rack apex with calibrated force.  Classic Mosconi-era pattern.",
    "Blocker removal.  Picks lower-probability shots that simultaneously pocket a ball AND clear a blocker from a pocket throat, when the long-horizon Q says it's worth it.",
    "Cluster nudging.  When two balls are touching, sometimes chooses a lower-probability shot that brushes them apart while pocketing a different ball - 'deal with the cluster now before it bites later.'",
    "Multi-rack continuation.  Reliably executes key-ball -> break-ball -> rack-scatter -> continue on the new rack, often across 5+ reracks.",
])
pdf.para(
    "Caveat: most of these require search ON to reliably surface.  The value "
    "head has learned the depth; the bare policy doesn't always expose it.  "
    "Distillation has narrowed but not closed this gap."
)


# --- 5. How to play ---
pdf.h1("5. How to play the demo")
pdf.para(
    "Open the URL on the cover page.  No login, no install.  Layout works "
    "best on a laptop or tablet in landscape; phone portrait is cramped."
)

pdf.h3("Controls")
pdf.kvtable([
    ("Start / Pause / Reset",
     "Start a game, pause animation, or reset to a fresh rack."),
    ("Auto vs Manual",
     "Auto: AI plays through the whole game on its own.  Manual: AI shows "
     "you the planned shot, you click Shoot to execute (also lets you "
     "inspect the per-shot probability annotations)."),
    ("Auto delay slider",
     "Sets the pause between shots in Auto mode.  0 = as fast as the "
     "animation runs.  Useful for watching long runs without 5 seconds per shot."),
    ("Trajectory line",
     "Shows the planned cue ball path before the animation plays.  In Manual "
     "mode you also see thin dashed lines for each candidate shot with "
     "(probability | cut angle) annotations."),
])

pdf.h3("What's interesting to watch for")
pdf.bullets([
    "End of rack: when 2-3 balls remain, watch the choice of which ball to pocket.  The model picks the one that's NOT the preferred break ball (better line to the rack apex, in the sweet-spot distance range).",
    "Side pockets near end of rack: the side-pocket break pattern.",
    "After the break: how well the cue ball lands on the new rack of 14 - whether it has a follow-up shot ready.",
    "Cluster handling: if two balls are touching, watch how the network deals with them.",
    "Rail shots: the model is now decent on these but the 'easy ones' are still occasionally misplayed.  Hard rail shots almost always pocket cleanly.",
])


# --- 6. Known weaknesses ---
pdf.h1("6. Known weaknesses")
pdf.bullets([
    "Easy rail shots occasionally missed.  Hard ones are reliable.  Diagnosis: deterministic physics gives loose force/spin tolerances on easy shots, so the model learns a vague default that fails on certain rail geometries.  An attempted fix via training noise (v21) traded EOR depth for rail calibration; further work in flight.",
    "Weak force on rail-ball break shots.  When the post-rerack break ball is positioned near a rail, the model uses ~80 force instead of the ~240 needed to scatter the rack.  Its learned rail-shot prior ('low force = safe pocket') overrides the scatter bonus.  Targeted curriculum drill added in v23 (in training as of writing).",
    "Search-dependent strategic depth.  Bare-policy play is noticeably weaker than policy+search play.  More distillation would close this further.",
    "78-degree-cut shots still get non-trivial probability in some positions, though they're now rarely *chosen* thanks to the search probability threshold.",
])


# --- 7. Tech stack ---
pdf.h1("7. Tech stack")
pdf.bullets([
    "Physics: custom C simulator (pool_sim.c, compiled to libpool_sim.so), called from Python via ctypes.  Calibrated friction/restitution, continuous spin factor.",
    "ML: PyTorch, CPU-only.  Transformer policy/value net is small (567K params) and was trained over ~3-4 weeks of incremental experiments.",
    "Demo: a Python HTTP server (~700 LOC) serves a single static HTML page that renders the table in a canvas and posts shot decisions to the server.",
    "Tunnel for remote access: cloudflared quick tunnel - one command, no account.",
])


# --- 8. URL again ---
pdf.h1("8. The link")
pdf.set_font("Helvetica", "", 10.5)
pdf.cell(0, 6, "Play it:", ln=True)
pdf.set_font("Courier", "B", 11)
pdf.set_text_color(40, 60, 140)
pdf.cell(0, 7, URL, ln=True, link=URL)
pdf.set_text_color(0)
pdf.set_font("Helvetica", "", 10.5)
pdf.ln(1)
pdf.para(
    "The tunnel needs the host machine to keep cloudflared running.  If the "
    "link stops working it's because that process was stopped - the AI itself "
    "is fine, it's just unreachable from the public internet.  Direct LAN "
    "access on the same WiFi as the host is via http://192.168.1.90:8001/."
)

out = "/home/r-m-glover/claude_projects/pool_player/Pool_AI_writeup.pdf"
pdf.output(out)
print(f"wrote {out}")
