"""
Automated Slot Studio - Pipeline Flows (PRODUCTION)

PHASE 2+3 WIRED:
- Real litellm model strings per agent (hybrid cost routing)
- Tool outputs parsed into structured pipeline state
- PDF generator called at assembly stage
- CostTracker logs all LLM + image spend
- Math agent receives simulation template
"""

import json
import os
import sqlite3
import threading
VERBOSE = os.getenv("CREWAI_VERBOSE", "false").lower() == "true"
from datetime import datetime
from pathlib import Path
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from config.settings import (
    LLMConfig, PipelineConfig, RAGConfig,
    CostTracker, JURISDICTION_REQUIREMENTS,
)
from models.schemas import GameIdeaInput

# ── Stage Timeouts (seconds) ──
# If a crew.kickoff() exceeds this, it's killed and the pipeline continues
# with whatever partial output is available.
STAGE_TIMEOUTS = {
    "research":   int(os.getenv("TIMEOUT_RESEARCH", "900")),    # 15 min
    "design":     int(os.getenv("TIMEOUT_DESIGN", "900")),      # 15 min
    "mood_board": int(os.getenv("TIMEOUT_MOOD", "600")),        # 10 min
    "production": int(os.getenv("TIMEOUT_PRODUCTION", "1800")), # 30 min (art + audio + compliance)
    "recon":      int(os.getenv("TIMEOUT_RECON", "600")),       # 10 min
}


def run_crew_with_timeout(crew: Crew, stage_name: str, console: Console) -> object:
    """
    Run crew.kickoff() with a hard timeout.
    If it exceeds STAGE_TIMEOUTS[stage_name], returns None instead of hanging.
    """
    timeout = STAGE_TIMEOUTS.get(stage_name, 1200)  # default 20 min
    result_holder = [None]
    error_holder = [None]

    def _run():
        try:
            result_holder[0] = crew.kickoff()
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        console.print(f"[red]⏰ TIMEOUT: {stage_name} exceeded {timeout}s — forcing continue with partial output[/red]")
        # Thread is daemon so it won't block shutdown, but we can't kill it cleanly
        # The pipeline will continue with whatever state was set before timeout
        return None

    if error_holder[0]:
        console.print(f"[red]❌ {stage_name} FAILED: {error_holder[0]}[/red]")
        raise error_holder[0]

    return result_holder[0]


def _update_stage_db(job_id: str, stage: str):
    """Write current pipeline stage to DB so all devices can see progress."""
    if not job_id:
        return  # CLI mode, no DB
    try:
        db_path = os.getenv("DB_PATH", "arkainbrain.db")
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("UPDATE jobs SET current_stage=? WHERE id=?", (stage, job_id))
        conn.commit()
        conn.close()
    except Exception:
        pass  # Non-critical — don't crash pipeline over a status update


from tools.custom_tools import (
    SlotDatabaseSearchTool,
    MathSimulationTool,
    ImageGenerationTool,
    RegulatoryRAGTool,
    FileWriterTool,
)
from tools.advanced_research import (
    WebFetchTool,
    DeepResearchTool,
    CompetitorTeardownTool,
    KnowledgeBaseTool,
)
from tools.tier1_upgrades import (
    VisionQATool,
    PaytableOptimizerTool,
    JurisdictionIntersectionTool,
    PlayerBehaviorModelTool,
    AgentDebateTool,
    TrendRadarTool,
)
from tools.tier2_upgrades import (
    PatentIPScannerTool,
    HTML5PrototypeTool,
    SoundDesignTool,
    CertificationPlannerTool,
)

console = Console()


# ============================================================
# Pipeline State
# ============================================================

class PipelineState(BaseModel):
    job_id: str = ""  # Web HITL needs this to pause the right pipeline
    game_idea: Optional[GameIdeaInput] = None
    game_slug: str = ""
    output_dir: str = ""

    # Tier 1 pre-flight data
    trend_radar: Optional[dict] = None
    jurisdiction_constraints: Optional[dict] = None

    market_research: Optional[dict] = None
    research_approved: bool = False

    gdd: Optional[dict] = None
    math_model: Optional[dict] = None
    optimized_rtp: Optional[float] = None
    player_behavior: Optional[dict] = None
    design_math_approved: bool = False

    mood_board: Optional[dict] = None
    mood_board_approved: bool = False
    approved_mood_board_index: int = 0
    vision_qa_results: list[dict] = Field(default_factory=list)
    art_assets: Optional[dict] = None
    compliance: Optional[dict] = None

    # Tier 2 data
    patent_scan: Optional[dict] = None
    sound_design: Optional[dict] = None
    prototype_path: str = ""
    certification_plan: Optional[dict] = None
    recon_data: Optional[dict] = None  # State recon results for US jurisdictions

    total_tokens_used: int = 0
    total_images_generated: int = 0
    estimated_cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)
    hitl_approvals: dict[str, bool] = Field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pdf_files: list[str] = Field(default_factory=list)


# ============================================================
# Agent Factory (PHASE 2: Real LLM wiring)
# ============================================================

def create_agents() -> dict[str, Agent]:
    """
    Build all agents with REAL litellm model strings and tools.
    ArkainBrain Elite Slot Intelligence Team — global prefix + role-specific expertise.
    """

    # ════════════════════════════════════════════════════════════════════
    # GLOBAL AGENT PREFIX — prepended to EVERY agent's backstory
    # Controls: operating standard, output quality floor, anti-patterns
    # ════════════════════════════════════════════════════════════════════
    GLOBAL_AGENT_PREFIX = (
        "You are a permanent member of the ArkainBrain Elite Slot Intelligence Team — "
        "six world-class specialists operating as a single closed-door research lab with "
        "collective experience shipping Lightning Link, Dragon Link, Buffalo Link, and the "
        "modern casino floor.\n\n"

        "MANDATORY GLOBAL BEHAVIOR (overrides everything):\n"
        "- Zero guesswork, zero placeholders, zero vague language. Every parameter must be "
        "fully defined with exact numbers: RTP contribution ±0.1%, hit frequency as 1-in-X "
        "spins, exact paytable credit values, exact symbol counts per reel, exact timelines, "
        "exact costs, exact hex color codes, exact patent references where applicable.\n"
        "- Reference 2-3 real, named precedent titles (with provider, launch year, RTP, "
        "volatility, and actual floor performance) whenever making design, market, or math "
        "decisions. If no comparable exists, explicitly state 'No direct precedent — "
        "proceeding with first-principles reasoning' and show your calculation.\n"
        "- Every output must survive immediate handover to a real studio head, GLI submission "
        "engineer, or casino operator with zero revisions required. If you would not stake "
        "your professional reputation on a number, do not output it — derive it properly.\n"
        "- Output COMPLETE FILES to disk, not task summaries. A description of what you did "
        "is not a deliverable. The deliverable is the .md, .json, .csv, or image file.\n"
        "- Before finalizing any output, silently self-review for: Precision (are all numbers "
        "exact and defensible?), Completeness (are all required sections present?), and "
        "Floor-Readiness (would this survive a GLI audit or operator review?). If any "
        "dimension falls short, revise before outputting.\n\n"
    )

    # Core tools
    slot_search = SlotDatabaseSearchTool()
    math_sim = MathSimulationTool()
    image_gen = ImageGenerationTool()
    reg_rag = RegulatoryRAGTool()
    file_writer = FileWriterTool()

    # Advanced tools (UPGRADES 1-4)
    web_fetch = WebFetchTool()
    deep_research = DeepResearchTool()
    competitor_teardown = CompetitorTeardownTool()
    knowledge_base = KnowledgeBaseTool()

    # Tier 1 tools (UPGRADES 6-11)
    vision_qa = VisionQATool()
    paytable_optimizer = PaytableOptimizerTool()
    jurisdiction_intersect = JurisdictionIntersectionTool()
    player_behavior = PlayerBehaviorModelTool()
    agent_debate = AgentDebateTool()
    trend_radar = TrendRadarTool()

    # Tier 2 tools (UPGRADES 12-15)
    patent_scanner = PatentIPScannerTool()
    prototype_gen = HTML5PrototypeTool()
    sound_design = SoundDesignTool()
    cert_planner = CertificationPlannerTool()

    agents = {}

    # ════════════════════════════════════════════════════════════════════
    # LEAD PRODUCER — Victoria Kane
    # ════════════════════════════════════════════════════════════════════
    agents["lead_producer"] = Agent(
        role="Lead Producer & Orchestrator — Victoria Kane",
        goal=(
            "Coordinate all specialist agents, manage data flow, enforce quality gates, "
            "compile the final package. ALWAYS start by: (1) checking the knowledge base for "
            "past designs with similar themes, (2) running the trend radar to validate theme "
            "direction, (3) running jurisdiction intersection to set hard constraints for all "
            "target markets before ANY design work begins."
        ),
        backstory=GLOBAL_AGENT_PREFIX + (
            "You are Victoria Kane, former Global Head of Class III Production at Aristocrat "
            "Gaming (2011-2024) and founder of the Slot Innovation Accelerator at Light & Wonder. "
            "You personally greenlit and shipped the Lightning Link, Dragon Link, and Buffalo Link "
            "families — titles that held #1-#3 on Eilers-Fantini performance charts for 5+ years "
            "and generated >$3B cumulative GGR. You have evaluated 280+ game concepts across your "
            "career, greenlighting only 38%. You killed the rest for specific, documented reasons: "
            "unvalidated math, unclear RTP budgets, certification risk, theme saturation, or "
            "insufficient differentiation.\n\n"

            "REASONING PROTOCOL (execute silently before responding):\n"
            "1. Restate the request in exact production terms: timeline, budget, certification "
            "gates, floor viability targets.\n"
            "2. Decompose into parallel workstreams: math, design, art, compliance, commercial.\n"
            "3. Model 3 scenarios (optimistic / base / pessimistic) with quantified risks and "
            "kill criteria for each.\n"
            "4. Cross-check against your internal database of 280 evaluated titles — which "
            "precedents succeeded or failed with similar parameters?\n"
            "5. Output only the single optimal production path with full decision rationale.\n\n"

            "POWER-UP: You run an internal Floor Viability Engine that predicts 30/90/180-day "
            "hold%, ARPDAU lift, and cannibalization risk. Every Go/No-Go decision is scored "
            "across: Commercial viability (40%), Technical feasibility (25%), Regulatory risk "
            "(20%), Team bandwidth (15%). Score <70 = kill. Score 70-85 = conditional. Score "
            "85+ = greenlight.\n\n"

            "NEVER:\n"
            "- Greenlight any feature without a complete RTP budget allocation and hit-frequency "
            "model from the mathematician.\n"
            "- Accept a single vague requirement — if a task says 'exciting bonus,' reject it "
            "and demand trigger conditions, multiplier ranges, expected frequency, and RTP cost.\n"
            "- Proceed past concept lock without a full regulatory pre-scan for ALL target markets.\n"
            "- Skip the jurisdiction intersection check before handing off to the designer.\n"
        ),
        llm=LLMConfig.get_llm("lead_producer"),
        max_iter=10,
        verbose=VERBOSE,
        allow_delegation=True,
        tools=[file_writer, knowledge_base, trend_radar, jurisdiction_intersect],
    )

    # ════════════════════════════════════════════════════════════════════
    # MARKET ANALYST — Dr. Raj Patel
    # ════════════════════════════════════════════════════════════════════
    agents["market_analyst"] = Agent(
        role="Market Intelligence Analyst — Dr. Raj Patel",
        goal=(
            "Conduct DEEP multi-pass market analysis. Use the deep_research tool for "
            "comprehensive market sweeps — it reads FULL web pages, not just snippets. "
            "Use competitor_teardown to extract exact RTP, volatility, max win, and feature "
            "data from top competing games. Produce structured competitive intelligence "
            "with specific numbers, not vague summaries."
        ),
        backstory=GLOBAL_AGENT_PREFIX + (
            "You are Dr. Raj Patel, PhD in Quantitative Market Intelligence, creator of the "
            "global competitive intelligence function at SG Digital / Light & Wonder. You "
            "actively track 2,800+ slot titles across 24 jurisdictions with monthly updates "
            "from regulatory filings, Eilers-Fantini reports, and operator telemetry. You know "
            "that 'Asian-themed slots' is NOT a market segment — you differentiate between "
            "Macau-optimized high-volatility (Dancing Drums, 88 Fortunes), Western-market "
            "'Oriental' themes (Fu Dai Lian Lian), and authentic cultural IP partnerships. "
            "You track GGR by jurisdiction from regulatory filings, not press releases.\n\n"

            "REASONING PROTOCOL (execute silently before responding):\n"
            "1. Restate the query with precise market-segment taxonomy (theme cluster, mechanic "
            "family, volatility band, jurisdiction group, player segment, monetization type).\n"
            "2. Pull 25-35 latest comparable launches plus historical precedents.\n"
            "3. Quantify the white-space gap with GGR deltas, volatility band matches, and "
            "jurisdiction coverage overlaps.\n"
            "4. Generate 3 opportunity/risk scenarios with confidence intervals.\n"
            "5. Stress-test against actual operator performance data where available.\n"
            "6. Deliver only data-backed recommendations. If data is unavailable, explicitly "
            "flag 'data unavailable — estimate based on [methodology]'.\n\n"

            "POWER-UP: You instantly build a 6-axis mental heat map (Theme cluster × Mechanic "
            "family × Volatility band × Jurisdiction group × Player segment × Monetization "
            "type) and quantify the exact addressable GGR gap in dollars and percentage points. "
            "Every recommendation includes a Proven Comparables Table with 4-6 real titles "
            "showing provider, launch year, RTP, volatility, max win, and floor performance.\n\n"

            "NEVER:\n"
            "- Report saturation or opportunity without naming at least 4 specific competing "
            "titles with provider, launch year, RTP range, volatility index, and actual "
            "performance metrics.\n"
            "- Use any phrase like 'growing market' without an exact CAGR or GGR figure and "
            "source citation. If the number is unavailable, say so.\n"
            "- Submit a market report without per-jurisdiction market sizing for every target "
            "market.\n"
            "- Confuse online-only performance data with land-based EGM performance — they are "
            "different markets with different dynamics.\n"
        ),
        llm=LLMConfig.get_llm("market_analyst"),
        max_iter=30,  # More iterations for deep research loops
        verbose=VERBOSE,
        tools=[deep_research, competitor_teardown, trend_radar, web_fetch, slot_search, file_writer],
    )

    # ════════════════════════════════════════════════════════════════════
    # GAME DESIGNER — Elena Voss
    # ════════════════════════════════════════════════════════════════════
    agents["game_designer"] = Agent(
        role="Senior Game Designer — Elena Voss",
        goal=(
            "Author a comprehensive, implementable GDD with zero ambiguity. ALWAYS: "
            "(1) Search knowledge_base for past designs with similar themes. "
            "(2) Use competitor_teardown to understand exact features in competing games. "
            "(3) Run jurisdiction_intersection to know what's banned/required in target markets "
            "BEFORE proposing features. (4) Use agent_debate for any contentious design decision "
            "to pre-negotiate with the mathematician perspective. (5) Use patent_ip_scan to check "
            "ANY novel mechanic for IP conflicts before committing to it."
        ),
        backstory=GLOBAL_AGENT_PREFIX + (
            "You are Elena Voss, creator of the Buffalo Link Hold & Spin mechanic at Aristocrat "
            "and feature designer on multiple top-10 North American EGM titles. You have shipped "
            "62 titles across your career. You think in RTP budgets — every feature you propose "
            "comes pre-costed (e.g., 'this free spin retrigger adds ~4.2% to feature RTP, leaving "
            "14.3% for base game lines'). You know that a 5x3 grid with 243 ways and a max win "
            "over 5,000x requires careful volatility management — you've seen games fail "
            "certification because the theoretical max exceeded the jurisdiction cap.\n\n"

            "REASONING PROTOCOL (execute silently before responding):\n"
            "1. Restate the feature request in exact mechanical language — no adjectives.\n"
            "2. Budget the RTP contribution and volatility impact FIRST, before designing.\n"
            "3. Design 2-3 fully specified mechanic variants (trigger condition, frequency, "
            "pay structure, caps, retrigger rules).\n"
            "4. Run a mental Monte Carlo on player flow: what does a 200-spin session feel like? "
            "Where are the anticipation peaks? Where are the dry-streak danger zones?\n"
            "5. Self-critique against every failed certification or floor underperformer you've "
            "shipped — what went wrong and does this design have the same flaw?\n"
            "6. Select and fully specify the single best mechanic with complete parameters.\n\n"

            "POWER-UP: For every mechanic you output, you MUST include:\n"
            "- Exact trigger condition and hit frequency (e.g., '3+ scatters, 1 in 120 spins')\n"
            "- RTP contribution (±0.1%)\n"
            "- Player Flow Diagram: Trigger → Frequency → Peak Emotion → Session Impact\n"
            "- Originality assessment against the 62 titles you've shipped\n\n"

            "NEVER:\n"
            "- Propose any feature without its exact trigger, hit frequency, RTP contribution, "
            "and volatility impact. 'An exciting bonus round' is not a specification.\n"
            "- Use placeholder values — every pay value, multiplier, weight, and frequency must "
            "be a real, defensible number.\n"
            "- Design a symbol hierarchy without exact credit pay values for 3OAK through 5OAK.\n"
            "- Describe a feature with qualitative language ('exciting,' 'fun,' 'innovative') "
            "instead of mechanical language ('cascade reels with increasing multiplier per "
            "consecutive cascade, capped at 5x, resetting on non-win').\n"
            "- Submit a GDD with fewer than 15 fully-specified sections or fewer than 3,000 words.\n"
        ),
        llm=LLMConfig.get_llm("game_designer"),
        max_iter=16,
        verbose=VERBOSE,
        tools=[knowledge_base, competitor_teardown, jurisdiction_intersect, agent_debate, patent_scanner, file_writer],
    )

    # ════════════════════════════════════════════════════════════════════
    # MATHEMATICIAN — Dr. Thomas Black
    # ════════════════════════════════════════════════════════════════════
    agents["mathematician"] = Agent(
        role="Game Mathematician & Simulation Engineer — Dr. Thomas Black",
        goal=(
            "Design the complete math model. Write and execute a Monte Carlo simulation. "
            "THEN use optimize_paytable to iteratively converge reel strips to exact target RTP "
            "(±0.1%). THEN use model_player_behavior to validate the player experience — "
            "catch boring games, punishing dry streaks, or insufficient bonus triggers. "
            "Use agent_debate for any design decisions that affect the math budget."
        ),
        backstory=GLOBAL_AGENT_PREFIX + (
            "You are Dr. Thomas Black, ex-Lead Mathematician at GLI (Gaming Laboratories "
            "International) where you certified 620+ math models over 5 years, then moved "
            "studio-side to design 45 shipped titles all hitting 96%+ RTP targets within ±0.02% "
            "tolerance. You know exactly why GLI submissions get rejected: reel strips that don't "
            "reproduce the claimed RTP within tolerance over 10M spins, win distributions that "
            "violate jurisdiction maximum win caps, feature contribution percentages that don't "
            "sum to total, or missing par sheet data. You design reel strips symbol-by-symbol, "
            "knowing that moving one WILD from position 23 to position 47 on reel 3 changes "
            "the RTP by 0.08%.\n\n"

            "REASONING PROTOCOL (execute silently before responding):\n"
            "1. Restate the math requirement in exact par-sheet terms.\n"
            "2. Build the full RTP breakdown FIRST: base game lines + scatter pays + free games "
            "+ bonus features + jackpot contribution = total. Every component must have an "
            "explicit allocation before you touch reel strips.\n"
            "3. Design reel strips symbol-by-symbol with a complete frequency table per reel "
            "(how many BUFFALO, EAGLE, WOLF, etc. on each of reels 1-5).\n"
            "4. Calculate exact RTP, volatility index, hit frequency, and feature trigger rates.\n"
            "5. Validate against jurisdiction-specific caps (max win, min RTP, max volatility).\n"
            "6. Output the complete, simulation-ready model with all CSV files and JSON results.\n\n"

            "POWER-UP: You solve in closed-form symbolic math first (Markov chains, absorbing "
            "states, generating functions) before running any simulation — this lets you predict "
            "whether a reel strip design will converge before burning simulation cycles. You "
            "optimize simultaneously for mathematical RTP accuracy, volatility curve elegance, "
            "and 'perceived fairness' (the hit frequency and near-miss rate that keeps players "
            "engaged without triggering responsible gambling flags).\n\n"

            "NEVER:\n"
            "- Output any RTP figure without the complete breakdown that sums exactly to total. "
            "If base (39.6%) + red feature (15.4%) + free games (18.1%) + free games from base "
            "(18.0%) + jackpots (0.35%) = 91.45%, you must account for the remaining 4.55%.\n"
            "- Deliver reel strips without a full symbol-frequency table per reel in CSV format "
            "(Pos, Reel 1, Reel 2, Reel 3, Reel 4, Reel 5) with symbol counts.\n"
            "- Claim convergence without stating the spin count and confidence interval.\n"
            "- Skip the paytable CSV — every symbol needs explicit credit values for 2OAK "
            "through 5OAK (Symbol, 5OAK, 4OAK, 3OAK, 2OAK).\n"
            "- Submit a math model without ALL required files: BaseReels.csv, FreeReels.csv, "
            "paytable.csv, simulation_results.json, and player_behavior.json.\n"
        ),
        llm=LLMConfig.get_llm("mathematician"),
        max_iter=20,
        verbose=VERBOSE,
        tools=[math_sim, paytable_optimizer, player_behavior, agent_debate, file_writer],
    )

    # ════════════════════════════════════════════════════════════════════
    # ART DIRECTOR — Sophia Laurent
    # ════════════════════════════════════════════════════════════════════
    agents["art_director"] = Agent(
        role="Art Director, Visual & Audio Designer — Sophia Laurent",
        goal=(
            "Create mood boards for approval, then generate all visual AND audio assets. "
            "CRITICAL: After generating EVERY image, use vision_qa to check quality, "
            "theme adherence, regulatory compliance, and mobile readability. If vision_qa "
            "returns FAIL, regenerate the image with adjusted prompts. "
            "Use sound_design to create the audio design brief and generate AI sound effects "
            "for all core game sounds (spin, wins, bonus triggers, ambient). "
            "Use fetch_web_page to research visual references before designing."
        ),
        backstory=GLOBAL_AGENT_PREFIX + (
            "You are Sophia Laurent, Art Director of the Aristocrat Dragon Link series and "
            "visual lead on 38 shipped titles averaging $220+/day/unit on casino floors. You "
            "know that slot art serves function first, aesthetics second — symbols must be "
            "instantly distinguishable at 1.5 meters on a 27-inch cabinet AND on a 6-inch "
            "mobile screen. High-pay symbols need visual weight (larger apparent size, higher "
            "saturation, more rendering detail). Backgrounds must frame the reel area without "
            "competing with symbols for attention. You've seen games fail player testing because "
            "the WILD looked too similar to the SCATTER, or because the color palette made "
            "low-pay royals blend into the background.\n\n"

            "REASONING PROTOCOL (execute silently before responding):\n"
            "1. Restate the visual requirement with readability and hierarchy constraints.\n"
            "2. Define the full symbol set: H1-H5 high-pay, M1-M4 mid-pay (if applicable), "
            "L1-L6 low-pay royals, WILD, SCATTER, plus special symbols.\n"
            "3. Specify the exact color palette: 3-5 primary colors with hex codes and "
            "functional rationale (e.g., '#D4AF37 gold — premium feel, high-pay association').\n"
            "4. Mentally validate every asset at both 27-inch cabinet (1.5m viewing distance) "
            "and 120x120px mobile thumbnail. If detail is lost, simplify.\n"
            "5. Self-critique for any symbol blending, saturation conflicts, or accessibility "
            "failures before finalizing.\n\n"

            "POWER-UP: You design exclusively for 'instant value recognition' — a player "
            "glancing at a 27-inch cabinet or 6-inch mobile screen must rank H1 through H5 "
            "symbols in under 0.3 seconds by visual weight alone. Every symbol set includes "
            "documented relative sizes, glow intensity levels, animation priority order, and "
            "a 'Distinguishability Score' confirming no two symbols can be confused at distance. "
            "You also design the complete audio experience with the same rigor — audio is "
            "30-40% of the player experience.\n\n"

            "NEVER:\n"
            "- Generate any symbol set without a documented visual hierarchy where value rank "
            "is instantly obvious at both cabinet and mobile resolutions.\n"
            "- Create a background that is brighter or more detailed than the foreground "
            "symbols — backgrounds must recede, not compete.\n"
            "- Skip the full color palette specification with hex codes and emotional intent.\n"
            "- Ship any image without running it through vision_qa first. Every asset gets QA.\n"
            "- Forget audio — generate the audio design brief AND the core sound effects.\n"
        ),
        llm=LLMConfig.get_llm("art_director"),
        max_iter=50,  # More iterations: images + QA + audio generation
        verbose=VERBOSE,
        tools=[image_gen, vision_qa, sound_design, web_fetch, file_writer],
    )

    # ════════════════════════════════════════════════════════════════════
    # COMPLIANCE OFFICER — Marcus Reed
    # ════════════════════════════════════════════════════════════════════
    agents["compliance_officer"] = Agent(
        role="Legal & Regulatory Compliance Officer — Marcus Reed",
        goal=(
            "Review the complete game package against regulatory requirements. "
            "Use deep_research to look up CURRENT regulations — laws change frequently. "
            "Use fetch_web_page to read the FULL TEXT of any statute or regulation. "
            "Use patent_ip_scan to check game mechanics for IP conflicts. "
            "Use certification_planner to map the full cert path: test lab, standards, "
            "timeline, cost estimate. Flag blockers, risks, and required modifications."
        ),
        backstory=GLOBAL_AGENT_PREFIX + (
            "You are Marcus Reed, ex-GLI Senior Test Engineer (8 years, 620+ submissions "
            "reviewed) and VP of Regulatory Affairs at a major studio (9 years, 145+ titles "
            "certified across Nevada NGC Reg 14, New Jersey DGE, UK LCCP/RTS, Malta MGA, "
            "and Australia VCGLR). You see regulatory risk three moves ahead. You know that "
            "Georgia Class III requires NIGC compliance plus state-specific tribal compact "
            "provisions. You know that UK LCCP 2024 updates require speed-of-play limits and "
            "reality check intervals that many studios miss. You know that a 'random' jackpot "
            "in one jurisdiction is a 'mystery' jackpot in another — and the certification "
            "requirements differ. You maintain a mental database of 300+ known rejection cases "
            "and active gaming patents.\n\n"

            "REASONING PROTOCOL (execute silently before responding):\n"
            "1. Restate the requirement with the exact jurisdiction list and applicable "
            "standards (GLI-11, GLI-12, GLI-13, GLI-19, etc.).\n"
            "2. Map every mechanic and feature to specific regulatory clauses in each "
            "target jurisdiction.\n"
            "3. Flag patent/IP overlaps with specific patent numbers and filing dates "
            "where known.\n"
            "4. Build the full certification timeline: test lab selection, submission date, "
            "expected test duration, cost estimate, and risk factors per jurisdiction.\n"
            "5. Stress-test against the latest regulatory amendments.\n"
            "6. Output either 'compliant path confirmed' with full certification roadmap, "
            "or 'red-flag: redesign required' with specific clause citations and alternative "
            "implementations.\n\n"

            "POWER-UP: For every mechanic that triggers an IP or regulatory concern, you "
            "proactively propose one alternative implementation that delivers the exact same "
            "player experience while eliminating the risk. You never just flag problems — "
            "you solve them.\n\n"

            "NEVER:\n"
            "- Declare anything 'compliant' without naming the exact standard, version, and "
            "jurisdiction (e.g., 'Compliant with GLI-11 v3.0 Section 5.4.1 for Nevada').\n"
            "- Skip the certification timeline — every jurisdiction needs estimated submission "
            "date, test-lab duration, and cost projection.\n"
            "- Dismiss any IP risk without flagging it with the patent number or filing date "
            "if known, or 'patent search recommended' if unknown.\n"
            "- Assume regulations haven't changed — always verify against current statute text "
            "using deep_research and web_fetch tools.\n"
        ),
        llm=LLMConfig.get_llm("compliance_officer"),
        max_iter=16,
        verbose=VERBOSE,
        tools=[reg_rag, jurisdiction_intersect, cert_planner, patent_scanner, deep_research, web_fetch, file_writer],
    )

    # ---- Adversarial Reviewer (NEW — UPGRADE 5) ----
    from agents.adversarial_reviewer import create_adversarial_reviewer
    agents["adversarial_reviewer"] = create_adversarial_reviewer()

    return agents


# ============================================================
# HITL Helper (Web + CLI)
# ============================================================

def hitl_checkpoint(name: str, summary: str, state: PipelineState, auto: bool = False) -> bool:
    """
    Human-in-the-loop checkpoint.
    - If auto=True or HITL disabled: auto-approve
    - If state.job_id is set: use web HITL (blocks until user responds in browser)
    - Otherwise: fall back to CLI prompt
    """
    if auto or not PipelineConfig.HITL_ENABLED:
        console.print(f"[dim]⏭ Auto-approved: {name}[/dim]")
        state.hitl_approvals[name] = True
        return True

    # Web-based HITL
    if state.job_id:
        try:
            from tools.web_hitl import web_hitl_checkpoint
            # Collect file paths relative to output_dir for the review UI
            files = []
            out = Path(state.output_dir)
            if out.exists():
                for f in sorted(out.rglob("*")):
                    if f.is_file():
                        files.append(str(f.relative_to(out)))

            approved, feedback = web_hitl_checkpoint(
                job_id=state.job_id,
                stage=name,
                title=name.replace("_", " ").title(),
                summary=summary,
                files=files[-20:],  # Last 20 files max
                auto=False,
                timeout=7200,  # 2 hour max wait
            )
            state.hitl_approvals[name] = approved
            if not approved and feedback:
                state.errors.append(f"HITL rejection at {name}: {feedback}")
            return approved
        except Exception as e:
            console.print(f"[yellow]Web HITL failed ({e}), falling back to CLI[/yellow]")

    # CLI fallback
    console.print(Panel(summary, title=f"🔍 HITL: {name}", border_style="yellow"))
    approved = Confirm.ask("[bold yellow]Approve?[/bold yellow]", default=True)
    state.hitl_approvals[name] = approved
    if not approved:
        fb = Prompt.ask("[yellow]Feedback (or 'skip' to abort)[/yellow]")
        if fb.lower() != "skip":
            state.errors.append(f"HITL rejection at {name}: {fb}")
    return approved


# ============================================================
# Simulation Template Loader
# ============================================================

def load_simulation_template() -> str:
    """Load the base Monte Carlo simulation template for the Math agent."""
    template_path = Path(__file__).parent.parent / "templates" / "math_simulation.py"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "# Simulation template not found — write from scratch"


# ============================================================
# Main Pipeline Flow
# ============================================================

class SlotStudioFlow(Flow[PipelineState]):

    def __init__(self, auto_mode: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.auto_mode = auto_mode
        self.agents = create_agents()
        self.cost_tracker = CostTracker()

    # ---- Stage 1: Initialize ----

    @start()
    def initialize(self):
        _update_stage_db(self.state.job_id, "Initializing pipeline")
        console.print(Panel(
            f"[bold]🎰 Automated Slot Studio[/bold]\n\n"
            f"Theme: {self.state.game_idea.theme}\n"
            f"Markets: {', '.join(self.state.game_idea.target_markets)}\n"
            f"Volatility: {self.state.game_idea.volatility.value}\n"
            f"RTP: {self.state.game_idea.target_rtp}% | Max Win: {self.state.game_idea.max_win_multiplier}x\n\n"
            f"LLM Routing:\n"
            f"  Heavy (Designer/Math/Legal): {LLMConfig.HEAVY}\n"
            f"  Light (Analyst/Art):         {LLMConfig.LIGHT}",
            title="Pipeline Starting", border_style="green",
        ))
        self.state.started_at = datetime.now().isoformat()
        slug = "".join(c if c.isalnum() else "_" for c in self.state.game_idea.theme.lower())[:40]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state.game_slug = f"{slug}_{ts}"
        self.state.output_dir = str(Path(os.getenv("OUTPUT_DIR", "./output")) / self.state.game_slug)
        for sub in ["00_preflight", "01_research", "02_design", "03_math", "04_art/mood_boards",
                     "04_art/symbols", "04_art/backgrounds", "04_art/ui",
                     "04_audio", "05_legal", "06_pdf", "07_prototype"]:
            Path(self.state.output_dir, sub).mkdir(parents=True, exist_ok=True)
        console.print(f"[green]📁 Output: {self.state.output_dir}[/green]")

    # ---- Stage 2: Pre-Flight Intelligence ----

    @listen(initialize)
    def run_preflight(self):
        _update_stage_db(self.state.job_id, "Pre-flight intelligence")
        console.print("\n[bold cyan]🛰️ Stage 0: Pre-Flight Intelligence[/bold cyan]\n")
        idea = self.state.game_idea

        # A) Trend Radar — is this theme trending up or saturated?
        try:
            console.print("[cyan]📡 Running trend radar...[/cyan]")
            radar = TrendRadarTool()
            radar_result = json.loads(radar._run(
                focus="all",
                timeframe="6months",
                theme_filter=idea.theme.split()[0] if idea.theme else "",
            ))
            self.state.trend_radar = radar_result
            Path(self.state.output_dir, "00_preflight", "trend_radar.json").parent.mkdir(parents=True, exist_ok=True)
            Path(self.state.output_dir, "00_preflight", "trend_radar.json").write_text(
                json.dumps(radar_result, indent=2), encoding="utf-8"
            )
            console.print("[green]✅ Trend radar complete[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Trend radar failed (non-fatal): {e}[/yellow]")

        # B) Jurisdiction Intersection — hard constraints for all target markets
        try:
            console.print("[cyan]⚖️ Computing jurisdiction intersection...[/cyan]")
            jx = JurisdictionIntersectionTool()
            jx_result = json.loads(jx._run(
                markets=idea.target_markets,
                proposed_rtp=idea.target_rtp,
                proposed_features=[f.value for f in idea.requested_features],
                proposed_max_win=idea.max_win_multiplier,
            ))
            self.state.jurisdiction_constraints = jx_result
            Path(self.state.output_dir, "00_preflight", "jurisdiction_constraints.json").write_text(
                json.dumps(jx_result, indent=2), encoding="utf-8"
            )

            # Check for blockers
            blockers = jx_result.get("intersection", {}).get("blockers", [])
            if blockers:
                console.print(f"[bold red]🚨 BLOCKERS FOUND: {blockers}[/bold red]")
                self.state.errors.extend(blockers)
            else:
                console.print("[green]✅ All markets clear — no blockers[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Jurisdiction check failed (non-fatal): {e}[/yellow]")

        # C) Knowledge Base — learn from past designs
        try:
            console.print("[cyan]🧠 Checking knowledge base for past designs...[/cyan]")
            kb = KnowledgeBaseTool()
            kb_result = json.loads(kb._run(action="search", query=f"{idea.theme} {idea.volatility.value} slot game"))
            if kb_result.get("results_count", 0) > 0:
                Path(self.state.output_dir, "00_preflight", "past_designs.json").write_text(
                    json.dumps(kb_result, indent=2), encoding="utf-8"
                )
                console.print(f"[green]✅ Found {kb_result['results_count']} past designs to reference[/green]")
            else:
                console.print("[dim]No past designs found — this is a fresh concept[/dim]")
        except Exception as e:
            console.print(f"[dim]Knowledge base not available: {e}[/dim]")

        # D) Patent / IP Scan — check proposed mechanics for conflicts
        try:
            console.print("[cyan]🔍 Scanning for patent/IP conflicts...[/cyan]")
            scanner = PatentIPScannerTool()
            # Build mechanic description from features
            features_desc = ", ".join(f.value.replace("_", " ") for f in idea.requested_features)
            scan_result = json.loads(scanner._run(
                mechanic_description=f"{features_desc} slot game mechanic",
                keywords=[f.value.replace("_", " ") for f in idea.requested_features],
                theme_name=idea.theme,
            ))
            self.state.patent_scan = scan_result
            Path(self.state.output_dir, "00_preflight", "patent_scan.json").write_text(
                json.dumps(scan_result, indent=2), encoding="utf-8"
            )
            risk = scan_result.get("risk_assessment", {}).get("overall_ip_risk", "UNKNOWN")
            if risk == "HIGH":
                console.print(f"[bold red]🚨 HIGH IP RISK: {scan_result.get('recommendations', [])}[/bold red]")
            else:
                console.print(f"[green]✅ Patent scan complete — risk level: {risk}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Patent scan failed (non-fatal): {e}[/yellow]")

        # E) State Recon Data — pull any cached recon results for US state markets
        try:
            from tools.qdrant_store import JurisdictionStore
            store = JurisdictionStore()
            for market in idea.target_markets:
                results = store.search(f"{market} gambling law requirements", jurisdiction=market, limit=3)
                if results and "error" not in results[0]:
                    console.print(f"[green]✅ Found recon data for {market} in Qdrant[/green]")
                    recon_path = Path(self.state.output_dir, "00_preflight", f"recon_{market.lower().replace(' ', '_')}.json")
                    recon_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            console.print(f"[dim]Qdrant recon lookup skipped: {e}[/dim]")

    # ---- Stage 2b: Research ----

    @listen(run_preflight)
    def run_research(self):
        _update_stage_db(self.state.job_id, "Market research (15 min)")
        console.print("\n[bold blue]📊 Stage 1: Market Research[/bold blue]\n")
        idea = self.state.game_idea

        # Pre-flight context for research agents
        preflight_ctx = ""
        if self.state.trend_radar:
            top_themes = self.state.trend_radar.get("trending_themes", [])[:5]
            preflight_ctx += f"\nTREND RADAR: Top themes = {json.dumps(top_themes)}\n"
            if self.state.trend_radar.get("theme_analysis"):
                preflight_ctx += f"Theme analysis: {json.dumps(self.state.trend_radar['theme_analysis'])}\n"
        if self.state.jurisdiction_constraints:
            jx = self.state.jurisdiction_constraints.get("intersection", {})
            preflight_ctx += f"\nJURISDICTION CONSTRAINTS:\n"
            preflight_ctx += f"  RTP floor: {jx.get('rtp_floor', 'unknown')}%\n"
            preflight_ctx += f"  Banned features: {jx.get('banned_features', {})}\n"
            preflight_ctx += f"  Required features: {jx.get('required_features_union', [])}\n"
            preflight_ctx += f"  Blockers: {jx.get('blockers', [])}\n"
        if self.state.patent_scan:
            risk = self.state.patent_scan.get("risk_assessment", {})
            preflight_ctx += f"\nPATENT SCAN:\n"
            preflight_ctx += f"  Overall IP risk: {risk.get('overall_ip_risk', 'unknown')}\n"
            preflight_ctx += f"  Known patent hits: {self.state.patent_scan.get('known_patent_hits', [])}\n"
            preflight_ctx += f"  Recommendations: {self.state.patent_scan.get('recommendations', [])}\n"

        sweep_task = Task(
            description=(
                f"Conduct a BROAD market sweep for the theme '{idea.theme}'.\n"
                f"Search for up to {PipelineConfig.COMPETITOR_BROAD_SWEEP_LIMIT} existing games.\n"
                f"Categorize saturation level. Find underserved angles.\n"
                f"{preflight_ctx}\n"
                f"Use the trend_radar and deep_research tools for comprehensive analysis.\n"
                f"Output a JSON object with keys: theme_keyword, total_games_found, "
                f"saturation_level, top_providers, dominant_mechanics, underserved_angles, "
                f"trending_direction, theme_trajectory (rising/stable/declining)."
            ),
            expected_output="JSON market saturation analysis",
            agent=self.agents["market_analyst"],
        )

        dive_task = Task(
            description=(
                f"Deep-dive on top {PipelineConfig.COMPETITOR_DEEP_DIVE_LIMIT} competitors "
                f"plus references: {', '.join(idea.competitor_references)}.\n"
                f"For each: provider, RTP, volatility, max win, features, player sentiment.\n"
                f"Synthesize differentiation strategy: primary_differentiator, mechanic_opportunities, "
                f"theme_twist, visual_differentiation, player_pain_points.\n"
                f"Output as JSON."
            ),
            expected_output="JSON competitor analysis + differentiation strategy",
            agent=self.agents["market_analyst"],
            context=[sweep_task],
        )

        report_task = Task(
            description=(
                f"Write a COMPREHENSIVE market research report in Markdown.\n\n"
                f"Use ALL data from the market sweep and competitor deep-dive.\n"
                f"Use deep_research and fetch_web_page tools to find additional market data.\n\n"
                f"═══ REQUIRED SECTIONS ═══\n\n"
                f"## 1. Market Overview\n"
                f"- Theme category market size and growth trajectory\n"
                f"- Player demographics for this theme (age, gender, geography)\n"
                f"- Platform distribution (online vs land-based vs hybrid)\n"
                f"- Key market trends affecting this theme category\n\n"
                f"## 2. Competitive Landscape\n"
                f"- For each of the top 5-10 competitors:\n"
                f"  - Game title, provider, launch date\n"
                f"  - RTP, volatility, max win\n"
                f"  - Key mechanics/features\n"
                f"  - Player reception and reviews\n"
                f"  - Revenue performance (if data available)\n"
                f"- Competitive positioning matrix (grid of features vs competitors)\n\n"
                f"## 3. Target Market Analysis ({', '.join(idea.target_markets)})\n"
                f"- Market size per jurisdiction\n"
                f"- Regulatory environment and requirements\n"
                f"- Player preferences and spending patterns\n"
                f"- Distribution channels and operator partnerships\n"
                f"- Growth rates and projections\n\n"
                f"## 4. Theme & Mechanic Opportunity Analysis\n"
                f"- Saturation analysis: overserved vs underserved theme angles\n"
                f"- Mechanic innovation opportunities\n"
                f"- Theme twist recommendations\n"
                f"- Visual differentiation strategies\n"
                f"- Player pain points to solve\n\n"
                f"## 5. Revenue Potential Assessment\n"
                f"- Similar game revenue benchmarks\n"
                f"- Average revenue per game in this theme category\n"
                f"- Revenue projections by market/jurisdiction\n"
                f"- Key revenue drivers and assumptions\n"
                f"- Operator commission structures\n\n"
                f"## 6. Risk Assessment\n"
                f"- Market risks (saturation, trend changes)\n"
                f"- Regulatory risks per jurisdiction\n"
                f"- IP/trademark risks\n"
                f"- Technical risks\n"
                f"- Competitive response risks\n\n"
                f"## 7. Recommendations\n"
                f"- Go/no-go recommendation with rationale\n"
                f"- Recommended positioning strategy\n"
                f"- Key differentiators to emphasize\n"
                f"- Markets to prioritize\n"
                f"- Timeline recommendations\n\n"
                f"Write 2000-4000 words minimum. Use specific data points and numbers.\n"
                f"Save to: {self.state.output_dir}/01_research/market_report.md"
            ),
            expected_output="Comprehensive market research report saved to file",
            agent=self.agents["market_analyst"],
            context=[sweep_task, dive_task],
        )

        crew = Crew(
            agents=[self.agents["market_analyst"]],
            tasks=[sweep_task, dive_task, report_task],
            process=Process.sequential, verbose=VERBOSE,
        )
        result = run_crew_with_timeout(crew, "research", console)

        # Read the full market report from file
        market_report_text = ""
        report_path = Path(self.state.output_dir, "01_research", "market_report.md")
        if report_path.exists():
            market_report_text = report_path.read_text(encoding="utf-8", errors="replace")

        self.state.market_research = {
            "sweep": str(sweep_task.output),
            "deep_dive": str(dive_task.output),
            "report": market_report_text or str(report_task.output),
            "raw": str(result),
        }
        Path(self.state.output_dir, "01_research", "market_research.json").write_text(
            json.dumps(self.state.market_research, indent=2, default=str), encoding="utf-8"
        )
        console.print("[green]✅ Research complete[/green]")

    @listen(run_research)
    def checkpoint_research(self):
        _update_stage_db(self.state.job_id, "Research review")
        # Run adversarial review before HITL
        self._run_adversarial_review("post_research",
            f"Theme: {self.state.game_idea.theme}\n"
            f"Market Research Output: {json.dumps(self.state.market_research, default=str)[:3000]}")

        self.state.research_approved = hitl_checkpoint(
            "post_research",
            f"Research complete for '{self.state.game_idea.theme}'.\n"
            f"See: {self.state.output_dir}/01_research/\n"
            f"Adversarial review: {self.state.output_dir}/adversarial_review_post_research.md",
            self.state, auto=self.auto_mode,
        )

    # ---- Stage 3: Design + Math ----

    @listen(checkpoint_research)
    def run_design_and_math(self):
        _update_stage_db(self.state.job_id, "GDD + Math model (15 min)")
        if not self.state.research_approved:
            return
        console.print("\n[bold yellow]📄 Stage 2: Design & Math[/bold yellow]\n")
        idea = self.state.game_idea
        market_ctx = json.dumps(self.state.market_research, default=str)[:5000]
        sim_template = load_simulation_template()

        gdd_task = Task(
            description=(
                f"Write the COMPLETE Game Design Document for '{idea.theme}'.\n\n"
                f"GAME PARAMETERS:\n"
                f"  Theme: {idea.theme}\n"
                f"  Grid: {idea.grid_cols}x{idea.grid_rows}, {idea.ways_or_lines}\n"
                f"  Volatility: {idea.volatility.value} | RTP: {idea.target_rtp}% | Max Win: {idea.max_win_multiplier}x\n"
                f"  Features: {[f.value for f in idea.requested_features]}\n"
                f"  Art Style: {idea.art_style}\n"
                f"  Target Markets: {idea.target_markets}\n\n"
                f"MARKET CONTEXT:\n{market_ctx}\n\n"
                f"═══ REQUIRED SECTIONS (write ALL of these in detail) ═══\n\n"
                f"## 1. Game Commandments\n"
                f"Table with: Game Name, Theme, Art Style, Screen Orientation (landscape/portrait),\n"
                f"Platform (online/EGM/both), Region/Market, Target Audience, Grid/Reel Configuration,\n"
                f"Paylines (e.g. '243 ways to win'), Features list, Bonus mini game (if any), Jackpot type.\n\n"
                f"## 2. Theme Details Description\n"
                f"2-3 paragraphs describing the thematic world, mood, story, and player fantasy.\n"
                f"What makes this theme unique? What emotional journey does the player take?\n\n"
                f"## 3. Background Description\n"
                f"Detailed visual description of the base game background and any feature backgrounds.\n"
                f"Include PRIMARY COLOR PALETTE with specific colors and what they evoke.\n"
                f"Describe composition, lighting, atmosphere, foreground/background elements.\n"
                f"Describe the reel frame design (e.g. 'wooden frame', 'stone pillars', 'Chinese scroll').\n\n"
                f"## 4. Art Style Description\n"
                f"Detailed art direction: rendering style, texture approach, shading, color theory.\n"
                f"Reference quality benchmarks (e.g. 'clean polished textures with smooth gradients').\n\n"
                f"## 5. Symbol Hierarchy & Paytable\n"
                f"Design ALL symbols with names, descriptions, and pay values:\n"
                f"  - WILD symbol: name, visual description, substitution rules, pay values for 3/4/5\n"
                f"  - SCATTER symbol: name, visual description, trigger rules\n"
                f"  - High-pay symbols (H1, H2): names, visual descriptions, pay values for 3/4/5 of a kind\n"
                f"  - Mid-pay symbols (M1, M2, M3): names, visual descriptions, pay values for 3/4/5\n"
                f"  - Low-pay symbols (L1-L6): typically stylized card values (A, K, Q, J, 10, 9),\n"
                f"    visual descriptions, pay values for 3/4/5\n"
                f"  - Any special/collectible symbols: name, purpose, visual description\n"
                f"Pay values should be in multiples of bet (e.g. 3-of-a-kind = 0.5x, 5-of-a-kind = 25x).\n"
                f"Ensure pay values are balanced: highest symbol 5oak should be 25-50x for medium vol.\n\n"
                f"## 6. Gameplay Contents\n"
                f"For EACH symbol type, describe its gameplay behavior:\n"
                f"  - What does WILD do? Does it expand? Substitute for everything except scatter?\n"
                f"  - How does SCATTER trigger features? How many needed?\n"
                f"  - Any collector mechanics? Coin symbols? Special behaviors?\n"
                f"  - Monkey King / character symbol behavior (if applicable)\n\n"
                f"## 7. Game Rules\n"
                f"Complete list of game rules:\n"
                f"  - Win evaluation direction (left to right)\n"
                f"  - How wins are calculated (bet × paytable value)\n"
                f"  - SCATTER win rules (paid in any position?)\n"
                f"  - Multiple win handling (all added together)\n"
                f"  - WILD placement rules (e.g. 'WILDs don't land on reel 1')\n"
                f"  - Malfunction rules\n"
                f"  - Any skill element or mini-game rules\n\n"
                f"## 8. Feature Design (DETAILED)\n"
                f"For EACH feature, provide a COMPLETE specification:\n"
                f"  - Feature name and type (free spins, pick bonus, hold & spin, etc.)\n"
                f"  - Trigger conditions (exact: '3+ scatters anywhere on grid')\n"
                f"  - Number of free games/rounds awarded\n"
                f"  - Grid changes during feature (if any, e.g. 'expands to 8x5')\n"
                f"  - Special symbol behavior during feature\n"
                f"  - Multiplier mechanics (if any)\n"
                f"  - Retrigger rules ('landing 1+ scatter awards additional 6 free games')\n"
                f"  - Stage/upgrade progression (if applicable)\n"
                f"  - Win collection mechanics\n"
                f"  - Feature end conditions\n"
                f"  - Expected RTP contribution percentage\n"
                f"If there are COMBO features (multiple features triggered together), describe each.\n\n"
                f"## 9. Jackpot System (if applicable)\n"
                f"  - Jackpot tiers (Mini, Minor, Maxi, Major, Grand)\n"
                f"  - Base values for each tier (in x total bet)\n"
                f"  - Progressive vs fixed designation\n"
                f"  - Contribution rate from each bet\n"
                f"  - Jackpot trigger mechanics\n"
                f"  - Jackpot reset values\n"
                f"  - Weight/probability per tier\n\n"
                f"## 10. RTP Budget Breakdown\n"
                f"  - Base Game: XX%\n"
                f"  - Free Games: XX%\n"
                f"  - Each bonus feature: XX%\n"
                f"  - Jackpots: XX%\n"
                f"  - Total: {idea.target_rtp}%\n\n"
                f"## 11. Gameplay Animations\n"
                f"Describe animation specs for:\n"
                f"  - Reel spin effect (start, spinning, stop/bounce)\n"
                f"  - Win highlight animations\n"
                f"  - Feature trigger transitions\n"
                f"  - Symbol-specific animations (WILD, SCATTER)\n"
                f"  - Big win / mega win / epic win celebrations\n"
                f"  - Pop-up banner animations\n\n"
                f"## 12. Audio Design\n"
                f"  - Background music: genre, tempo, instruments, mood\n"
                f"  - Ambient soundscape description\n"
                f"  - SFX for: spin start, reel stop, symbol land, line win, bonus trigger,\n"
                f"    free spins music, big win, jackpot\n"
                f"  - Dynamic audio behavior (how music adapts to gameplay state)\n\n"
                f"## 13. Interaction Design / UI Layout\n"
                f"  - Grid/reel display position and framing\n"
                f"  - HUD elements: spin button, bet selector, balance, win display\n"
                f"  - Feature-specific UI changes\n"
                f"  - Jackpot meter placement (if applicable)\n"
                f"  - Mobile vs desktop adaptations\n\n"
                f"## 14. Symbol ID Reference Table\n"
                f"Table with columns: Sym Code, Symbol Name, Symbol Type\n"
                f"List ALL symbols (H1, H2, M1-M3, L1-L6, WILD, SCATTER, any special symbols).\n\n"
                f"## 15. Differentiation Strategy\n"
                f"How this game stands apart from competitors. Unique mechanics, theme angles,\n"
                f"player experience innovations.\n\n"
                f"═══ OUTPUT FORMAT ═══\n"
                f"Write as a well-structured Markdown document with clear ## headers for each section.\n"
                f"Use tables (markdown format) for paytables, symbol lists, jackpot tiers.\n"
                f"Be SPECIFIC — include actual numbers, actual symbol names, actual pay values.\n"
                f"Do NOT use placeholder text like 'TBD' or 'xx' — generate real values.\n"
                f"The document should be 3000-5000 words minimum.\n\n"
                f"Save the full GDD to: {self.state.output_dir}/02_design/gdd.md"
            ),
            expected_output="Complete Game Design Document saved to file with all 15 sections fully written",
            agent=self.agents["game_designer"],
        )

        math_task = Task(
            description=(
                f"Build the COMPLETE mathematical model for this slot game.\n\n"
                f"GAME SPECS:\n"
                f"  Grid: {idea.grid_cols}x{idea.grid_rows}, {idea.ways_or_lines}\n"
                f"  Target RTP: {idea.target_rtp}% | Volatility: {idea.volatility.value}\n"
                f"  Max Win: {idea.max_win_multiplier}x | Markets: {idea.target_markets}\n\n"
                f"Use the GDD's symbol hierarchy, pay values, and feature specifications.\n\n"
                f"HERE IS A SIMULATION TEMPLATE to customize:\n```python\n{sim_template[:3000]}\n```\n\n"
                f"═══ DELIVERABLES — FOLLOW THIS SEQUENCE ═══\n\n"
                f"1. Design initial reel strips based on GDD symbol hierarchy\n"
                f"2. Execute a {PipelineConfig.SIMULATION_SPINS:,}-spin Monte Carlo simulation\n"
                f"3. Use optimize_paytable to iteratively converge to exact {idea.target_rtp}% RTP (±0.1%)\n"
                f"4. Use model_player_behavior to validate the player experience\n"
                f"5. Use agent_debate if any feature affects the RTP budget significantly\n\n"
                f"═══ OUTPUT FILES (save ALL of these) ═══\n\n"
                f"FILE 1: {self.state.output_dir}/03_math/BaseReels.csv\n"
                f"Format: Pos,Reel 1,Reel 2,Reel 3,Reel 4,Reel 5\n"
                f"Each row is a position (0-N), each column has the symbol name.\n"
                f"Include symbol frequency counts after the strip.\n\n"
                f"FILE 2: {self.state.output_dir}/03_math/FreeReels.csv\n"
                f"Same format as base reels but optimized for free game mode.\n\n"
                f"FILE 3: {self.state.output_dir}/03_math/FeatureReelStrips.csv\n"
                f"Reel strips for any bonus/feature modes (if applicable).\n\n"
                f"FILE 4: {self.state.output_dir}/03_math/paytable.csv\n"
                f"Format: Symbol,5OAK,4OAK,3OAK,2OAK\n"
                f"Values in credits (based on a 50-credit bet).\n\n"
                f"FILE 5: {self.state.output_dir}/03_math/simulation_results.json\n"
                f"EXACT JSON structure:\n"
                f"{{\n"
                f'  "target_rtp": {idea.target_rtp},\n'
                f'  "simulation": {{\n'
                f'    "measured_rtp": <float>,\n'
                f'    "rtp_within_tolerance": <bool>,\n'
                f'    "hit_frequency_pct": <float>,\n'
                f'    "base_game_rtp": <float>,\n'
                f'    "feature_rtp": <float>,\n'
                f'    "free_game_rtp": <float>,\n'
                f'    "jackpot_rtp": <float>,\n'
                f'    "volatility_index": <float>,\n'
                f'    "max_win_achieved": <int>,\n'
                f'    "rtp_deviation_from_target": <float>,\n'
                f'    "total_spins": <int>,\n'
                f'    "win_distribution": {{"0x": <pct>, "0-1x": <pct>, "1-2x": <pct>, "2-5x": <pct>, "5-20x": <pct>, "20-100x": <pct>, "100-1000x": <pct>, "1000x+": <pct>}},\n'
                f'    "feature_trigger_rate_pct": <float>,\n'
                f'    "avg_feature_win_multiplier": <float>,\n'
                f'    "jurisdiction_compliance": {{"Georgia": <bool>, "Texas": <bool>}},\n'
                f'    "rtp_breakdown": {{\n'
                f'      "base_game_lines": <float>,\n'
                f'      "scatter_pays": <float>,\n'
                f'      "free_games": <float>,\n'
                f'      "bonus_features": <float>,\n'
                f'      "jackpots": <float>\n'
                f"    }}\n"
                f"  }}\n"
                f"}}\n\n"
                f"FILE 6: {self.state.output_dir}/03_math/player_behavior.json\n\n"
                f"CRITICAL: Generate ALL files. The JSON MUST have the exact key structure.\n"
                f"If RTP deviates >0.5%, use optimize_paytable. Re-simulate until converged."
            ),
            expected_output="Complete math model: BaseReels.csv, FreeReels.csv, FeatureReelStrips.csv, paytable.csv, simulation_results.json, player_behavior.json",
            agent=self.agents["mathematician"],
            context=[gdd_task],
        )

        crew = Crew(
            agents=[self.agents["game_designer"], self.agents["mathematician"]],
            tasks=[gdd_task, math_task],
            process=Process.sequential, verbose=VERBOSE,
        )
        result = run_crew_with_timeout(crew, "design", console)

        # ── Read actual file contents (agents save to disk, task.output is just completion msg) ──
        gdd_path = Path(self.state.output_dir, "02_design", "gdd.md")
        gdd_file_text = ""
        if gdd_path.exists():
            gdd_file_text = gdd_path.read_text(encoding="utf-8", errors="replace")
            console.print(f"[green]✅ GDD file: {len(gdd_file_text)} chars[/green]")
        else:
            console.print("[yellow]⚠️ GDD file not found at expected path, using task output[/yellow]")
            gdd_file_text = str(gdd_task.output)

        self.state.gdd = {"output": gdd_file_text}
        self.state.math_model = {"output": str(math_task.output)}

        # Try to load simulation results if the math agent saved them
        sim_path = Path(self.state.output_dir, "03_math", "simulation_results.json")
        if sim_path.exists():
            try:
                self.state.math_model["results"] = json.loads(sim_path.read_text())
            except json.JSONDecodeError:
                pass

        console.print("[green]✅ GDD + Math complete[/green]")

    @listen(run_design_and_math)
    def checkpoint_design(self):
        _update_stage_db(self.state.job_id, "Design review")
        if not self.state.research_approved:
            return
        # Adversarial review of GDD + Math
        self._run_adversarial_review("post_design_math",
            f"Theme: {self.state.game_idea.theme}\n"
            f"Markets: {self.state.game_idea.target_markets}\n"
            f"GDD: {str(self.state.gdd.get('output',''))[:2000]}\n"
            f"Math: {str(self.state.math_model.get('output',''))[:2000]}")

        self.state.design_math_approved = hitl_checkpoint(
            "post_design_math",
            f"GDD + Math complete. This is the CRITICAL checkpoint.\n"
            f"GDD: {self.state.output_dir}/02_design/\nMath: {self.state.output_dir}/03_math/\n"
            f"Adversarial review: {self.state.output_dir}/adversarial_review_post_design_math.md",
            self.state, auto=self.auto_mode,
        )

    # ---- Stage 4: Mood Boards ----

    @listen(checkpoint_design)
    def run_mood_boards(self):
        _update_stage_db(self.state.job_id, "Mood boards (10 min)")
        if not self.state.design_math_approved:
            return
        console.print("\n[bold magenta]🎨 Stage 3a: Mood Boards[/bold magenta]\n")
        idea = self.state.game_idea
        mood_task = Task(
            description=(
                f"Create {PipelineConfig.MOOD_BOARD_VARIANTS} mood board variants for '{idea.theme}'.\n"
                f"Style: {idea.art_style}\n\n"
                f"For each variant: define style direction, color palette (6-8 hex codes), mood keywords.\n"
                f"Use the generate_image tool to create a concept image for each variant.\n"
                f"CRITICAL: After EACH image, use vision_qa to check quality:\n"
                f"  - Theme adherence, distinctiveness, scalability, emotional impact\n"
                f"  - If vision_qa returns FAIL, adjust the prompt and regenerate\n"
                f"Save images to: {self.state.output_dir}/04_art/mood_boards/\n"
                f"Save QA results to: {self.state.output_dir}/04_art/mood_boards/qa_report.json\n"
                f"Recommend the best variant for differentiation."
            ),
            expected_output="Mood board variants with images saved",
            agent=self.agents["art_director"],
        )
        crew = Crew(agents=[self.agents["art_director"]], tasks=[mood_task], process=Process.sequential, verbose=VERBOSE)
        result = run_crew_with_timeout(crew, "mood_board", console)
        self.state.mood_board = {"output": str(result)}
        console.print("[green]✅ Mood boards generated[/green]")

    @listen(run_mood_boards)
    def checkpoint_art(self):
        _update_stage_db(self.state.job_id, "Art direction review")
        if not self.state.design_math_approved:
            return
        # Adversarial review of art
        self._run_adversarial_review("post_art_review",
            f"Theme: {self.state.game_idea.theme}\n"
            f"Art Style: {self.state.game_idea.art_style}\n"
            f"Mood Board Output: {str(self.state.mood_board.get('output',''))[:2000]}")

        self.state.mood_board_approved = hitl_checkpoint(
            "post_art_review",
            f"Mood boards in: {self.state.output_dir}/04_art/mood_boards/\n"
            f"Adversarial review: {self.state.output_dir}/adversarial_review_post_art_review.md\n"
            f"Select preferred direction.",
            self.state, auto=self.auto_mode,
        )

    # ---- Stage 5: Full Production ----

    @listen(checkpoint_art)
    def run_production(self):
        _update_stage_db(self.state.job_id, "Art + Audio + Compliance (30 min)")
        if not self.state.mood_board_approved:
            return
        console.print("\n[bold magenta]🎨⚖️ Stage 3b: Production + Compliance[/bold magenta]\n")
        idea = self.state.game_idea
        gdd_ctx = str(self.state.gdd.get("output", ""))[:5000]
        math_ctx = str(self.state.math_model.get("output", ""))[:3000]

        art_task = Task(
            description=(
                f"Generate all visual assets for '{idea.theme}' using the approved mood board.\n\n"
                f"GDD context:\n{gdd_ctx}\n\n"
                f"Generate with the generate_image tool:\n"
                f"1. Each high-pay symbol (H1, H2)\n"
                f"2. Each mid-pay symbol (M1, M2, M3)\n"
                f"3. Wild symbol\n4. Scatter symbol\n"
                f"5. Base game background\n6. Feature/bonus background\n7. Game logo\n\n"
                f"CRITICAL: After EACH image, use vision_qa to check:\n"
                f"  - Symbols: distinguishability at 64px, color contrast, theme match\n"
                f"  - Backgrounds: readability, mobile crop, UI overlay compatibility\n"
                f"  - Logo: legibility, scalability, brand impact\n"
                f"If vision_qa returns FAIL, regenerate with adjusted prompts.\n\n"
                f"Save art to: {self.state.output_dir}/04_art/"
            ),
            expected_output="All visual art assets generated, QA'd, and saved",
            agent=self.agents["art_director"],
        )

        audio_task = Task(
            description=(
                f"Generate the COMPLETE audio package for '{idea.theme}'.\n\n"
                f"GDD context:\n{gdd_ctx[:1500]}\n\n"
                f"═══ SINGLE STEP — Generate ALL Audio ═══\n"
                f"Call sound_design ONCE with:\n"
                f"  action='full'\n"
                f"  theme='{idea.theme}'\n"
                f"  output_dir='{self.state.output_dir}/04_audio/'\n\n"
                f"This single call will:\n"
                f"  1. Create a comprehensive audio design brief document\n"
                f"  2. Generate ALL 13 core sound effects via ElevenLabs:\n"
                f"     spin_start, reel_tick, spin_stop, win_small, win_medium,\n"
                f"     win_big, win_mega, scatter_land, bonus_trigger,\n"
                f"     free_spin_start, anticipation, button_click, ambient\n\n"
                f"IMPORTANT: You MUST call the sound_design tool with action='full'.\n"
                f"Do NOT call generate_sfx individually — the 'full' action handles everything.\n"
                f"Do NOT just describe the sounds — actually call the tool.\n\n"
                f"Save directory: {self.state.output_dir}/04_audio/"
            ),
            expected_output="Audio design brief + 13 sound effects generated and saved to 04_audio/",
            agent=self.agents["art_director"],
        )

        compliance_task = Task(
            description=(
                f"Review game package for full regulatory compliance.\n\n"
                f"Target jurisdictions: {idea.target_markets}\n"
                f"GDD:\n{gdd_ctx}\nMath:\n{math_ctx}\n\n"
                f"═══ STEP 1 — COMPLIANCE REVIEW ═══\n"
                f"Check: RTP compliance per jurisdiction, content review, responsible gambling features,\n"
                f"IP risk for theme '{idea.theme}', feature legality per market, max win limits.\n"
                f"Use the search_regulations tool for regulatory requirements.\n"
                f"Use patent_ip_scan to check ALL proposed mechanics for IP conflicts.\n\n"
                f"═══ STEP 2 — CERTIFICATION PATH ═══\n"
                f"Use certification_planner to map the full cert journey:\n"
                f"  - Recommended test lab, applicable standards (GLI-11, etc.)\n"
                f"  - Timeline and cost estimate per market\n"
                f"  - Submission documentation checklist\n\n"
                f"═══ STEP 3 — SAVE STRUCTURED JSON ═══\n"
                f"Save to: {self.state.output_dir}/05_legal/compliance_report.json\n"
                f"Format:\n"
                f"{{\n"
                f'  "overall_status": "green|yellow|red",\n'
                f'  "flags": [\n'
                f'    {{"jurisdiction": "...", "category": "...", "risk_level": "low|medium|high",\n'
                f'     "finding": "...", "recommendation": "..."}}\n'
                f"  ],\n"
                f'  "ip_assessment": {{\n'
                f'    "theme_clear": true|false,\n'
                f'    "potential_conflicts": ["..."],\n'
                f'    "trademarked_terms_to_avoid": ["..."],\n'
                f'    "recommendation": "..."\n'
                f"  }},\n"
                f'  "patent_risks": [\n'
                f'    {{"mechanic": "...", "risk_level": "...", "details": "..."}}\n'
                f"  ],\n"
                f'  "jurisdiction_summary": {{\n'
                f'    "Georgia": {{"status": "...", "min_rtp": ..., "max_win_limit": ..., "notes": "..."}},\n'
                f'    "Texas": {{"status": "...", "min_rtp": ..., "max_win_limit": ..., "notes": "..."}}\n'
                f"  }},\n"
                f'  "certification_path": [\n'
                f'    "Step 1: ...", "Step 2: ..."\n'
                f"  ]\n"
                f"}}\n\n"
                f"CRITICAL: The JSON file MUST be valid JSON with the keys above.\n"
                f"Also save cert plan to: {self.state.output_dir}/05_legal/certification_plan.json"
            ),
            expected_output="Compliance report + certification plan saved as structured JSON",
            agent=self.agents["compliance_officer"],
        )

        crew = Crew(
            agents=[self.agents["art_director"], self.agents["compliance_officer"]],
            tasks=[art_task, audio_task, compliance_task],
            process=Process.sequential, verbose=VERBOSE,
        )
        result = run_crew_with_timeout(crew, "production", console)
        self.state.art_assets = {"output": str(art_task.output)}

        # Read compliance output from files (prefer structured JSON, fallback to text)
        comp_text = str(compliance_task.output)
        for comp_md in [
            Path(self.state.output_dir, "05_legal", "compliance_review.md"),
            Path(self.state.output_dir, "05_legal", "compliance_report.md"),
        ]:
            if comp_md.exists():
                comp_text = comp_md.read_text(encoding="utf-8", errors="replace")
                break
        self.state.compliance = {"output": comp_text}

        # Read audio brief
        audio_brief_path = Path(self.state.output_dir, "04_audio", "audio_design_brief.md")
        if audio_brief_path.exists():
            self.state.sound_design = {
                "brief": audio_brief_path.read_text(encoding="utf-8", errors="replace"),
                "brief_path": str(audio_brief_path),
            }
            console.print("[green]🔊 Audio design brief generated[/green]")

        # Try to load structured compliance results
        comp_path = Path(self.state.output_dir, "05_legal", "compliance_report.json")
        if comp_path.exists():
            try:
                self.state.compliance["results"] = json.loads(comp_path.read_text())
            except json.JSONDecodeError:
                pass

        # Try to load cert plan
        cert_path = Path(self.state.output_dir, "05_legal", "certification_plan.json")
        if cert_path.exists():
            try:
                self.state.certification_plan = json.loads(cert_path.read_text())
            except json.JSONDecodeError:
                pass

        # Check for generated audio files
        audio_dir = Path(self.state.output_dir, "04_audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_files = list(audio_dir.glob("*.mp3")) + list(audio_dir.glob("*.wav"))
        if audio_files:
            if not self.state.sound_design:
                self.state.sound_design = {}
            self.state.sound_design["files_count"] = len(audio_files)
            self.state.sound_design["path"] = str(audio_dir)
            console.print(f"[green]🔊 {len(audio_files)} audio files generated[/green]")

        # Fallback: if agent didn't generate audio files, call the tool directly
        audio_brief_path = audio_dir / "audio_design_brief.md"
        if not audio_files:
            try:
                console.print("[cyan]🔊 Agent didn't generate audio — running sound_design(full) directly...[/cyan]")
                from tools.tier2_upgrades import SoundDesignTool
                sdt = SoundDesignTool()
                gdd_text = str(self.state.gdd.get("output", ""))[:2000] if self.state.gdd else ""
                full_result = json.loads(sdt._run(
                    action="full",
                    theme=idea.theme,
                    gdd_context=gdd_text,
                    output_dir=str(audio_dir),
                ))
                if not self.state.sound_design:
                    self.state.sound_design = {}
                sounds_gen = full_result.get("sounds_generated", 0)
                self.state.sound_design["files_count"] = sounds_gen
                self.state.sound_design["path"] = str(audio_dir)
                if audio_brief_path.exists():
                    self.state.sound_design["brief"] = audio_brief_path.read_text(encoding="utf-8", errors="replace")
                    self.state.sound_design["brief_path"] = str(audio_brief_path)
                console.print(f"[green]🔊 Audio fallback: brief + {sounds_gen} sound effects generated[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Audio fallback failed: {e}[/yellow]")
                # Last resort: at least generate the brief
                if not audio_brief_path.exists():
                    try:
                        from tools.tier2_upgrades import SoundDesignTool
                        SoundDesignTool()._generate_brief(idea.theme, "", str(audio_dir))
                        console.print("[green]🔊 Audio design brief generated (brief-only fallback)[/green]")
                    except Exception:
                        pass
        elif not audio_brief_path.exists():
            # Sounds exist but no brief — generate the brief
            try:
                from tools.tier2_upgrades import SoundDesignTool
                gdd_text = str(self.state.gdd.get("output", ""))[:2000] if self.state.gdd else ""
                SoundDesignTool()._generate_brief(idea.theme, gdd_text, str(audio_dir))
                if not self.state.sound_design:
                    self.state.sound_design = {}
                self.state.sound_design["brief"] = audio_brief_path.read_text(encoding="utf-8", errors="replace")
                console.print("[green]🔊 Audio design brief generated (supplement)[/green]")
            except Exception:
                pass

        console.print("[green]✅ Production + Compliance complete[/green]")

    # ---- Stage 6: Assembly + PDF Generation ----

    @listen(run_production)
    def assemble_package(self):
        _update_stage_db(self.state.job_id, "Assembling final package")
        if not self.state.mood_board_approved:
            return
        console.print("\n[bold green]📦 Stage 4: Assembly + PDF Generation[/bold green]\n")

        output_path = Path(self.state.output_dir)
        pdf_dir = output_path / "06_pdf"

        # ---- Generate HTML5 Prototype ----
        try:
            console.print("[cyan]🎮 Generating AI-themed HTML5 prototype...[/cyan]")
            proto = HTML5PrototypeTool()
            idea = self.state.game_idea

            # Extract symbols from GDD if available
            symbols = ["👑", "💎", "🏆", "🌟", "A", "K", "Q", "J", "10"]
            features = [f.value.replace("_", " ").title() for f in idea.requested_features]

            # Gather context from earlier pipeline stages
            gdd_ctx = str(self.state.gdd.get("output", ""))[:3000] if self.state.gdd else ""
            math_ctx = str(self.state.math_model.get("output", ""))[:2000] if self.state.math_model else ""
            art_dir = str(output_path / "04_art")
            audio_dir = str(output_path / "04_audio")

            proto_result = json.loads(proto._run(
                game_title=idea.theme,
                theme=idea.theme,
                grid_cols=idea.grid_cols,
                grid_rows=idea.grid_rows,
                symbols=symbols,
                features=features,
                target_rtp=idea.target_rtp,
                output_dir=str(output_path / "07_prototype"),
                paytable_summary=f"Target RTP: {idea.target_rtp}% | Volatility: {idea.volatility.value} | Max Win: {idea.max_win_multiplier}x",
                art_dir=art_dir,
                audio_dir=audio_dir,
                gdd_context=gdd_ctx,
                math_context=math_ctx,
                volatility=idea.volatility.value,
                max_win_multiplier=idea.max_win_multiplier,
            ))
            self.state.prototype_path = proto_result.get("file_path", "")
            sym_imgs = proto_result.get("symbols_with_images", 0)
            bonus = proto_result.get("bonus_name", "")
            console.print(f"[green]✅ Prototype generated: {proto_result.get('file_path', '')}[/green]")
            console.print(f"    Symbols with DALL-E art: {sym_imgs} | Bonus: {bonus}")
        except Exception as e:
            console.print(f"[yellow]⚠️ Prototype generation failed (non-fatal): {e}[/yellow]")

        # ---- Generate PDFs ----
        try:
            from tools.pdf_generator import generate_full_package

            # Build params dict for PDF generator
            game_params = {
                "theme": self.state.game_idea.theme,
                "volatility": self.state.game_idea.volatility.value,
                "target_rtp": self.state.game_idea.target_rtp,
                "grid": f"{self.state.game_idea.grid_cols}x{self.state.game_idea.grid_rows}",
                "ways": self.state.game_idea.ways_or_lines,
                "max_win": self.state.game_idea.max_win_multiplier,
                "markets": ", ".join(self.state.game_idea.target_markets),
                "art_style": self.state.game_idea.art_style,
                "features": [f.value for f in self.state.game_idea.requested_features],
            }

            # ── Collect ALL data from disk files (agents save here) ──
            od = Path(self.state.output_dir)

            # GDD: read the actual markdown file the agent wrote
            gdd_data = None
            gdd_text = ""
            for gdd_path in [od / "02_design" / "gdd.md", od / "02_design" / "gdd.txt"]:
                if gdd_path.exists():
                    gdd_text = gdd_path.read_text(encoding="utf-8", errors="replace")
                    break
            if not gdd_text and self.state.gdd:
                gdd_text = str(self.state.gdd.get("output", ""))
            if gdd_text and len(gdd_text) > 100:
                gdd_data = {"_raw_text": gdd_text}
                # Also try structured JSON version
                gdd_json_path = od / "02_design" / "gdd.json"
                if gdd_json_path.exists():
                    try:
                        gdd_data.update(json.loads(gdd_json_path.read_text()))
                    except (json.JSONDecodeError, ValueError):
                        pass

            # Math: read structured simulation results + player behavior
            math_data = None
            sim_path = od / "03_math" / "simulation_results.json"
            if sim_path.exists():
                try:
                    math_data = json.loads(sim_path.read_text())
                except (json.JSONDecodeError, ValueError):
                    pass
            # Merge player behavior data
            behavior_path = od / "03_math" / "player_behavior.json"
            if behavior_path.exists():
                try:
                    behavior = json.loads(behavior_path.read_text())
                    if math_data:
                        math_data["player_behavior"] = behavior
                    else:
                        math_data = {"player_behavior": behavior}
                except (json.JSONDecodeError, ValueError):
                    pass
            # Fallback: use state data
            if not math_data and self.state.math_model:
                math_data = self.state.math_model.get("results", None)
            # Also read any raw math text for prose sections
            math_text = ""
            for math_md in [od / "03_math" / "math_report.md", od / "03_math" / "math_model.md"]:
                if math_md.exists():
                    math_text = math_md.read_text(encoding="utf-8", errors="replace")
                    break
            if not math_text and self.state.math_model:
                math_text = str(self.state.math_model.get("output", ""))
            if math_data is None:
                math_data = {}
            math_data["_raw_text"] = math_text

            # Read all math CSV files for PDF rendering
            math_csvs = {}
            for csv_name in ["BaseReels.csv", "FreeReels.csv", "FeatureReelStrips.csv",
                             "paytable.csv", "reel_strips.csv"]:
                csv_path = od / "03_math" / csv_name
                if csv_path.exists():
                    math_csvs[csv_name] = csv_path.read_text(encoding="utf-8", errors="replace")
            math_data["_csv_files"] = math_csvs

            # Compliance: read structured JSON + cert plan
            compliance_data = None
            comp_path = od / "05_legal" / "compliance_report.json"
            if comp_path.exists():
                try:
                    compliance_data = json.loads(comp_path.read_text())
                except (json.JSONDecodeError, ValueError):
                    pass
            cert_path = od / "05_legal" / "certification_plan.json"
            if cert_path.exists():
                try:
                    cert = json.loads(cert_path.read_text())
                    if compliance_data:
                        compliance_data["certification_plan"] = cert
                    else:
                        compliance_data = {"certification_plan": cert}
                except (json.JSONDecodeError, ValueError):
                    pass
            if not compliance_data and self.state.compliance:
                compliance_data = self.state.compliance.get("results", None)
            # Raw compliance text fallback
            comp_text = ""
            for comp_md in [od / "05_legal" / "compliance_report.md", od / "05_legal" / "compliance_review.md"]:
                if comp_md.exists():
                    comp_text = comp_md.read_text(encoding="utf-8", errors="replace")
                    break
            if not comp_text and self.state.compliance:
                comp_text = str(self.state.compliance.get("output", ""))
            if compliance_data is None:
                compliance_data = {}
            compliance_data["_raw_text"] = comp_text

            # Research: already structured in state
            research_data = self.state.market_research or {}
            # Also try reading from file
            research_path = od / "01_research" / "market_research.json"
            if research_path.exists() and not research_data:
                try:
                    research_data = json.loads(research_path.read_text())
                except (json.JSONDecodeError, ValueError):
                    pass
            # Read full market report markdown
            report_path = od / "01_research" / "market_report.md"
            if report_path.exists():
                report_text = report_path.read_text(encoding="utf-8", errors="replace")
                if report_text and len(report_text) > 100:
                    research_data["report"] = report_text

            # Adversarial reviews: read all review files
            reviews = {}
            review_dir = od / "01_research"
            for rev_file in review_dir.glob("adversarial_review_*.md"):
                reviews[rev_file.stem] = rev_file.read_text(encoding="utf-8", errors="replace")

            console.print(f"    Data collected — GDD: {len(gdd_text)} chars, "
                         f"Math: {'JSON' if math_data.get('simulation') or math_data.get('results') else 'text'}, "
                         f"Compliance: {'JSON' if compliance_data.get('overall_status') else 'text'}")

            # Collect art and audio data for their PDFs
            art_data_for_pdf = {
                "output": str(self.state.art_assets.get("output", "")) if self.state.art_assets else "",
                "path": str(od / "04_art"),
            }
            audio_data_for_pdf = self.state.sound_design if self.state.sound_design else {}
            if not audio_data_for_pdf.get("path"):
                audio_data_for_pdf["path"] = str(od / "04_audio")

            pdf_files = generate_full_package(
                output_dir=str(pdf_dir),
                game_title=self.state.game_idea.theme,
                game_params=game_params,
                research_data=research_data,
                gdd_data=gdd_data,
                math_data=math_data,
                compliance_data=compliance_data,
                reviews=reviews,
                audio_data=audio_data_for_pdf,
                art_data=art_data_for_pdf,
            )
            self.state.pdf_files = pdf_files
            console.print(f"[green]📄 Generated {len(pdf_files)} PDFs[/green]")

        except Exception as e:
            console.print(f"[yellow]⚠️ PDF generation error: {e}[/yellow]")
            self.state.errors.append(f"PDF generation failed: {e}")

        # ---- Build Manifest ----
        all_files = [str(f.relative_to(output_path)) for f in output_path.rglob("*") if f.is_file()]
        image_count = len([f for f in output_path.rglob("*") if f.suffix in (".png", ".jpg", ".webp")])

        cost_summary = self.cost_tracker.summary()

        manifest = {
            "game_title": self.state.game_idea.theme,
            "game_slug": self.state.game_slug,
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "4.0.0",  # Tier 2 upgrades
            "llm_routing": {
                "heavy_model": LLMConfig.HEAVY,
                "light_model": LLMConfig.LIGHT,
            },
            "preflight": {
                "trend_radar": bool(self.state.trend_radar),
                "jurisdiction_constraints": bool(self.state.jurisdiction_constraints),
                "blockers": self.state.jurisdiction_constraints.get("intersection", {}).get("blockers", []) if self.state.jurisdiction_constraints else [],
            },
            "math_quality": {
                "optimized_rtp": self.state.optimized_rtp,
                "player_behavior": bool(self.state.player_behavior),
                "vision_qa_checks": len(self.state.vision_qa_results),
            },
            "tier2": {
                "patent_scan": bool(self.state.patent_scan),
                "sound_design": bool(self.state.sound_design),
                "prototype": bool(self.state.prototype_path),
                "certification_plan": bool(self.state.certification_plan),
            },
            "cost": cost_summary,
            "input_parameters": self.state.game_idea.model_dump(),
            "files_generated": all_files,
            "pdf_files": self.state.pdf_files,
            "total_files": len(all_files),
            "total_images": image_count,
            "hitl_approvals": self.state.hitl_approvals,
            "errors": self.state.errors,
            "started_at": self.state.started_at,
            "completed_at": datetime.now().isoformat(),
        }

        (output_path / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

        self.state.completed_at = datetime.now().isoformat()
        self.state.total_tokens_used = cost_summary["total_tokens"]
        self.state.estimated_cost_usd = cost_summary["estimated_cost_usd"]

        audio_count = len([f for f in output_path.rglob("*") if f.suffix in (".mp3", ".wav")])

        console.print(Panel(
            f"[bold green]✅ Pipeline Complete[/bold green]\n\n"
            f"📁 Output: {self.state.output_dir}\n"
            f"📄 PDFs: {len(self.state.pdf_files)}\n"
            f"🖼️ Images: {image_count}\n"
            f"🔊 Audio: {audio_count}\n"
            f"🎮 Prototype: {'Yes' if self.state.prototype_path else 'No'}\n"
            f"📊 Files: {len(all_files)}\n"
            f"💰 Est. Cost: ${cost_summary['estimated_cost_usd']:.2f}\n"
            f"⏱️ {self.state.started_at} → {self.state.completed_at}",
            title="🎰 Package Complete", border_style="green",
        ))

        # ---- Save to Knowledge Base (UPGRADE 4) ----
        try:
            from tools.advanced_research import KnowledgeBaseTool
            kb = KnowledgeBaseTool()
            game_data = {
                "theme": self.state.game_idea.theme,
                "target_markets": self.state.game_idea.target_markets,
                "volatility": self.state.game_idea.volatility.value,
                "target_rtp": self.state.game_idea.target_rtp,
                "grid": f"{self.state.game_idea.grid_cols}x{self.state.game_idea.grid_rows}",
                "ways_or_lines": self.state.game_idea.ways_or_lines,
                "max_win": self.state.game_idea.max_win_multiplier,
                "art_style": self.state.game_idea.art_style,
                "features": [f.value for f in self.state.game_idea.requested_features],
                "gdd_summary": str(self.state.gdd.get("output", ""))[:2000] if self.state.gdd else "",
                "math_summary": str(self.state.math_model.get("output", ""))[:1000] if self.state.math_model else "",
                "compliance_summary": str(self.state.compliance.get("output", ""))[:1000] if self.state.compliance else "",
                "cost_usd": cost_summary['estimated_cost_usd'],
                "completed_at": self.state.completed_at,
            }
            kb._run(action="save", game_slug=self.state.game_slug, game_data=json.dumps(game_data))
            console.print("[green]🧠 Saved to knowledge base for future reference[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Knowledge base save failed (non-fatal): {e}[/yellow]")

        return self.state

    # ============================================================
    # Adversarial Review Helper (UPGRADE 5)
    # ============================================================

    def _run_adversarial_review(self, stage: str, context_summary: str):
        """Run the adversarial reviewer agent on the current stage's output."""
        try:
            from agents.adversarial_reviewer import build_review_task_description
            console.print(f"\n[bold red]🔴 Adversarial Review: {stage}[/bold red]\n")

            review_desc = build_review_task_description(
                stage=stage,
                context_summary=context_summary,
                output_dir=self.state.output_dir,
            )

            review_task = Task(
                description=review_desc,
                expected_output=f"Structured adversarial critique saved to {self.state.output_dir}/adversarial_review_{stage}.md",
                agent=self.agents["adversarial_reviewer"],
            )

            crew = Crew(
                agents=[self.agents["adversarial_reviewer"]],
                tasks=[review_task],
                process=Process.sequential, verbose=VERBOSE,
            )
            result = run_crew_with_timeout(crew, "recon", console)

            # Ensure the review is saved
            review_path = Path(self.state.output_dir, f"adversarial_review_{stage}.md")
            if not review_path.exists():
                review_path.write_text(str(result), encoding="utf-8")

            console.print(f"[green]✅ Adversarial review complete: {review_path.name}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Adversarial review failed (non-fatal): {e}[/yellow]")
