"""ConsensusX: Research Demonstration & Evaluation Dashboard (Streamlit).

Interactive prototype for final-year AIML demonstration:
1. Live Claim Verification & Multi-Agent Interactive Audit.
2. Retrieved Evidence Inspection (FAISS Vector Store + Provenance).
3. Multi-Agent Stance, Rebuttal, and Sycophancy Inspection.
4. Evidence-Based Judge, Epistemic Safety Overrides & Adaptive Stopping Audit.
5. Comparative Benchmark Analytics (Configs A, B, C, D) from Recorded Experiments.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import settings
from src.data.schema import FEVERLabel, FEVERRecord
from src.llm.factory import get_llm_client
from src.pipeline.baseline import BaselinePipeline
from src.pipeline.consensus_x import ConsensusXPipeline
from src.pipeline.debate import DebatePipeline
from src.pipeline.voting import MajorityVotingPipeline
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.retriever import FAISSRetriever
from src.retrieval.vector_store import FAISSVectorStore

# ------------------------------------------------------------------------------
# Page Configuration & Theming
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ConsensusX | Multi-Agent Fact Verification",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for research styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .subtitle { font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem; }
    .verdict-banner {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.4rem;
        text-align: center;
        margin: 1rem 0;
    }
    .verdict-supports { background-color: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
    .verdict-refutes { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    .verdict-nei { background-color: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val { font-size: 1.6rem; font-weight: 700; color: #0F172A; }
    .metric-lbl { font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; }
    .agent-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .adversary-tag {
        background-color: #EF4444;
        color: white;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Cached Resources (Vector Store & Retriever)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_vector_store_retriever(index_dir: str = "data/processed/faiss_index"):
    """Loads FAISS vector store with mock embedding model for instant zero-cost retrieval."""
    idx_path = Path(index_dir)
    if not idx_path.exists():
        return None, "Vector index directory not found. Please run index builder first."
    try:
        emb_model = get_embedding_model(provider="mock")
        vector_store = FAISSVectorStore.load(index_dir, embedding_model=emb_model)
        retriever = FAISSRetriever(vector_store=vector_store, default_top_k=5)
        return retriever, None
    except Exception as e:
        return None, str(e)


# ------------------------------------------------------------------------------
# Sidebar Controls & Global State
# ------------------------------------------------------------------------------
st.sidebar.markdown("### ⚖️ **ConsensusX Navigation**")
app_mode = st.sidebar.radio(
    "Select Interface Mode:",
    [
        "🔬 Live Claim Verification",
        "📊 Recorded Benchmark Analytics",
        "📖 Architecture & Research Notes",
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ **Execution Settings**")

provider_choice = st.sidebar.selectbox(
    "LLM Provider:",
    ["mock", "gemini", "xai"],
    index=0,
    help="Select 'mock' for instant, zero-cost pipeline demonstration. Select 'gemini' or 'xai' for live API inference."
)

# API Credential Status
has_gemini = bool(os.getenv("GEMINI_API_KEY"))
has_xai = bool(os.getenv("XAI_API_KEY"))

if provider_choice == "gemini":
    if has_gemini:
        st.sidebar.success("🟢 Gemini API Key Active")
    else:
        st.sidebar.error("🔴 Gemini API Key Missing in .env")
elif provider_choice == "xai":
    if has_xai:
        st.sidebar.success("🟢 xAI API Key Active")
    else:
        st.sidebar.error("🔴 xAI API Key Missing in .env")
else:
    st.sidebar.info("💡 Mock Mode (Offline Simulated Reasoning)")

model_name: Optional[str] = None
if provider_choice == "gemini":
    model_name = st.sidebar.selectbox("Gemini Model:", ["gemini-2.0-flash", "gemini-2.5-flash"], index=0)
elif provider_choice == "xai":
    model_name = st.sidebar.selectbox("xAI Model:", ["grok-2", "grok-beta"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎭 **Adversary Injection (Phase 7)**")
inject_adv = st.sidebar.checkbox("Inject Adversary Agent", value=False, help="Compromise one agent with deceptive arguments or fabricated citations.")
adv_persona = "persuasive_contrarian"
adv_strength = "assertive"
if inject_adv:
    adv_persona = st.sidebar.selectbox(
        "Adversary Persona:",
        ["persuasive_contrarian", "citation_distorter"],
        index=0,
        help="'persuasive_contrarian' inverts verdicts with high rhetorical confidence. 'citation_distorter' invents fabricated passage IDs."
    )
    adv_strength = st.sidebar.selectbox(
        "Attack Strength:",
        ["assertive", "subtle"],
        index=0,
        help="'assertive' sets 98% confidence. 'subtle' sets 76% confidence to sow doubt."
    )


# ==============================================================================
# MODE 1: LIVE CLAIM VERIFICATION & MULTI-AGENT AUDIT
# ==============================================================================
if app_mode == "🔬 Live Claim Verification":
    st.markdown('<div class="main-title">ConsensusX: Live Fact Verification & Multi-Agent Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Evaluate factual claims across Single-Agent, Majority Voting, Structured Debate, and ConsensusX with live epistemic safety overrides.</div>', unsafe_allow_html=True)

    # Preloaded FEVER test claims
    EXAMPLE_CLAIMS = {
        "Custom Claim Input": "",
        "SUPPORTS Example: Civilization IV": "Civilization IV was released for Microsoft Windows and received critical acclaim.",
        "REFUTES Example: Firefox Hardware": "Firefox is a piece of hardware manufactured for personal computers.",
        "NOT ENOUGH INFO Example: Film Viewership": "The independent documentary was watched by exactly twenty million people on its opening night.",
        "Adversarial Challenge: Shakespeare": "William Shakespeare wrote Hamlet and was born in Stratford-upon-Avon.",
    }

    selected_example = st.selectbox("Load Example Benchmark Claim:", list(EXAMPLE_CLAIMS.keys()), index=1)
    default_text = EXAMPLE_CLAIMS[selected_example]

    claim_input = st.text_area(
        "Factual Claim to Verify:",
        value=default_text if default_text else "Firefox is a piece of hardware manufactured for personal computers.",
        height=85
    )

    col_cfg, col_run = st.columns([3, 1])
    with col_cfg:
        config_choice = st.radio(
            "Select Architecture Configuration:",
            [
                "Config 1: Single-Agent Baseline",
                "Config 2: Majority Voting (N=3)",
                "Config 3: Structured Debate (N=3, R=2)",
                "Config 4: ConsensusX (Debate + RAG + Judge + Safety Override)",
            ],
            index=3,
            horizontal=True
        )
    with col_run:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Verify Claim", type="primary", use_container_width=True)

    if run_btn and claim_input.strip():
        # Setup simulated or real record
        dummy_record = FEVERRecord(
            id=10101,
            claim=claim_input.strip(),
            label=FEVERLabel.NOT_ENOUGH_INFO,  # Unbiased label for live inference
            verifiable="VERIFIABLE",
            evidence=[]
        )

        retriever, ret_err = load_vector_store_retriever()
        if ret_err:
            st.warning(f"⚠️ Vector Store Notice: {ret_err}. Proceeding with ungrounded context.")

        llm_client = get_llm_client(provider=provider_choice, model_name=model_name, temperature=0.0)

        with st.spinner(f"Executing {config_choice} using provider '{provider_choice}'..."):
            start_wall = time.perf_counter()

            # Execute Selected Pipeline
            if config_choice == "Config 1: Single-Agent Baseline":
                pipeline = BaselinePipeline(llm_client=llm_client, retriever=retriever)
                res = pipeline.evaluate_records([dummy_record])
                claim_result = res["results"][0]
                verdict = claim_result["predicted_label"]
                conf = claim_result["confidence"]
                evidence_passages = retriever.retrieve(claim_input) if retriever else []

            elif config_choice == "Config 2: Majority Voting (N=3)":
                pipeline = MajorityVotingPipeline(
                    llm_client=llm_client,
                    num_agents=3,
                    retriever=retriever,
                    inject_adversary=inject_adv,
                    adversary_persona=adv_persona,
                    adversary_strength=adv_strength,
                    adversary_idx=1
                )
                res = pipeline.evaluate_records([dummy_record])
                claim_result = res["results"][0]
                verdict = claim_result["predicted_label"]
                conf = claim_result["confidence"]
                evidence_passages = retriever.retrieve(claim_input) if retriever else []

            elif config_choice == "Config 3: Structured Debate (N=3, R=2)":
                pipeline = DebatePipeline(
                    llm_client=llm_client,
                    num_agents=3,
                    max_rounds=2,
                    retriever=retriever,
                    inject_adversary=inject_adv,
                    adversary_persona=adv_persona,
                    adversary_strength=adv_strength,
                    adversary_idx=1
                )
                debate_out = pipeline.run_debate(dummy_record)
                claim_result = {
                    "predicted_label": debate_out.predicted_label.value,
                    "confidence": debate_out.confidence,
                    "transcript": debate_out.transcript,
                    "is_tie": debate_out.metadata.get("is_tie", False),
                    "consensus_reached": debate_out.consensus_reached,
                }
                verdict = debate_out.predicted_label.value
                conf = debate_out.confidence
                evidence_passages = retriever.retrieve(claim_input) if retriever else []

            else:  # Config 4: ConsensusX
                pipeline = ConsensusXPipeline(
                    llm_client=llm_client,
                    retriever=retriever,
                    num_agents=3,
                    max_rounds=2,
                    min_rounds=1,
                    inject_adversary=inject_adv,
                    adversary_persona=adv_persona,
                    adversary_strength=adv_strength,
                    adversary_idx=1
                )
                cx_out = pipeline.run_consensus_x(dummy_record)
                claim_result = {
                    "predicted_label": cx_out.predicted_label.value,
                    "confidence": cx_out.confidence,
                    "judge_output": cx_out.judge_output,
                    "transcript": cx_out.transcript,
                    "stopping_reason": cx_out.stopping_reason,
                    "rounds_used": cx_out.rounds_used,
                    "agent_consensus_reached": cx_out.agent_consensus_reached,
                    "agent_consensus_label": cx_out.agent_consensus_label.value if cx_out.agent_consensus_label else None,
                }
                verdict = cx_out.predicted_label.value
                conf = cx_out.confidence
                evidence_passages = retriever.retrieve(claim_input) if retriever else []

            elapsed = time.perf_counter() - start_wall

        # Display Final Verdict Banner
        if verdict == "SUPPORTS":
            banner_class = "verdict-supports"
            banner_icon = "✅"
            display_verdict = "SUPPORTED"
        elif verdict == "REFUTES":
            banner_class = "verdict-refutes"
            banner_icon = "❌"
            display_verdict = "REFUTED"
        else:
            banner_class = "verdict-nei"
            banner_icon = "⚠️"
            display_verdict = "NOT ENOUGH EVIDENCE (ABSTAINED)"

        st.markdown(
            f'<div class="verdict-banner {banner_class}">{banner_icon} FINAL SYSTEM VERDICT: {display_verdict} (Confidence: {conf*100:.1f}%)</div>',
            unsafe_allow_html=True
        )

        # High-level Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{display_verdict.split()[0]}</div><div class="metric-lbl">Decision</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{conf*100:.1f}%</div><div class="metric-lbl">Confidence Score</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{elapsed:.2f}s</div><div class="metric-lbl">Latency</div></div>', unsafe_allow_html=True)
        with m4:
            prov_mode = f"Mock Logic" if getattr(llm_client, "is_mock", True) else f"Live {provider_choice.upper()}"
            st.markdown(f'<div class="metric-card"><div class="metric-val">{prov_mode}</div><div class="metric-lbl">Execution Mode</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Tabs for deep research audit
        tab_evidence, tab_agents, tab_judge = st.tabs([
            "📚 Retrieved Evidence & Grounding",
            "🤖 Multi-Agent Turns & Debate Rounds",
            "⚖️ Evidence Judge & Epistemic Audit"
        ])

        # TAB 1: Evidence Inspection
        with tab_evidence:
            st.markdown("#### Retrieved Source Passages (FAISS Vector Search Top-K)")
            if evidence_passages:
                for p in evidence_passages:
                    with st.expander(f"📄 Passage ID: `{p.passage_id}` | Wikipedia Page: **{p.title}** (Score: {p.score:.3f}, Rank #{p.rank})", expanded=True):
                        st.write(p.text)
            else:
                st.info("No external passages retrieved. Running in ungrounded baseline mode.")

        # TAB 2: Multi-Agent Turns & Debate
        with tab_agents:
            st.markdown("#### Multi-Agent Deliberation & Peer Interactions")
            if "transcript" in claim_result and hasattr(claim_result["transcript"], "rounds"):
                transcript = claim_result["transcript"]
                for r_num, turns in transcript.rounds.items():
                    st.markdown(f"##### **Round {r_num}: {'Initial Independent Reasoning' if r_num == 1 else 'Cross-Examination & Rebuttals'}**")
                    cols = st.columns(len(turns))
                    for idx, (c_col, turn) in enumerate(zip(cols, turns)):
                        is_adv = (turn.metadata or {}).get("is_adversarial", False)
                        with c_col:
                            adv_badge = ' <span class="adversary-tag">ADVERSARY</span>' if is_adv else ""
                            st.markdown(f'<div class="agent-card"><b>{turn.agent_id}</b>{adv_badge}</div>', unsafe_allow_html=True)
                            st.write(f"**Verdict:** `{turn.predicted_label.value}`")
                            st.write(f"**Confidence:** {turn.confidence:.2f}")
                            if turn.rebuttal:
                                st.write(f"**Rebuttal:** {turn.rebuttal}")
                            st.write(f"**Evidence:** {turn.evidence or 'None'}")
                            st.write(f"**Inference:** {turn.inference or 'None'}")
                            st.write(f"**Citations:** `{turn.cited_evidence}`")
                    st.write("")

                if transcript.opinion_shifts:
                    st.warning(f"🔄 **Opinion Shifts Detected**: {len(transcript.opinion_shifts)} agent(s) changed stance during cross-examination.")
                    st.json(transcript.opinion_shifts)
                else:
                    st.info("ℹ️ Stance Stability: All agents maintained their initial stance through cross-examination.")

            elif "agent_votes" in claim_result:
                st.markdown("##### **Independent Agent Votes (Majority Voting)**")
                st.table([
                    {
                        "Agent ID": v["agent_id"],
                        "Verdict": v["label"],
                        "Confidence": f"{v['confidence']:.2f}",
                        "Reasoning": v["reasoning"][:120] + "...",
                        "Adversarial?": "YES" if v.get("is_adversarial") else "NO"
                    }
                    for v in claim_result["agent_votes"]
                ])
            else:
                st.write("**Single-Agent Reasoning:**")
                st.write(claim_result.get("reasoning", "No reasoning recorded."))

        # TAB 3: Evidence Judge & Adaptive Stopping Audit
        with tab_judge:
            st.markdown("#### Evidence-Based Judge Reasoning (Config 4)")
            if "judge_output" in claim_result:
                j_out = claim_result["judge_output"]
                st.write(f"**Raw Model Proposed Verdict:** `{j_out.raw_predicted_label.value if j_out.raw_predicted_label else 'N/A'}`")
                st.write(f"**Validated Final Verdict:** `{j_out.predicted_label.value}`")

                # Safety Override Notice
                if j_out.downgrade_applied:
                    st.error(f"🛡️ **Epistemic Safety Override Active**: Downgraded to `NOT ENOUGH INFO`. Reason: `{j_out.downgrade_reason}`")
                else:
                    st.success("✅ **Citation Grounding Valid**: Verdict is directly substantiated by retrieved source documents.")

                col_j1, col_j2 = st.columns(2)
                with col_j1:
                    st.write(f"**Valid Cited Passage IDs:** `{j_out.valid_citations}`")
                    st.write(f"**Unsupported / Fabricated Citations:** `{j_out.unsupported_citations}`")
                    st.write(f"**Citation Status:** `{j_out.citation_validation_status}`")
                with col_j2:
                    st.write(f"**Stopping Reason:** `{claim_result.get('stopping_reason')}`")
                    st.write(f"**Rounds Used:** `{claim_result.get('rounds_used')}` / 2")
                    st.write(f"**Committee Consensus:** `{claim_result.get('agent_consensus_label')}` (Reached: {claim_result.get('agent_consensus_reached')})")

                st.markdown("##### **Judge Rationale:**")
                st.info(j_out.rationale)
                if j_out.uncertainty:
                    st.write(f"**Epistemic Uncertainty:** {j_out.uncertainty}")
            else:
                st.info("Select **Config 4: ConsensusX** to audit the Evidence-Based Judge and citation validation overrides.")


# ==============================================================================
# MODE 2: RECORDED BENCHMARK ANALYTICS
# ==============================================================================
elif app_mode == "📊 Recorded Benchmark Analytics":
    st.markdown('<div class="main-title">ConsensusX: Experimental Robustness Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Inspect reproducibility metrics, False Consensus Rates (FCR), Adversarial Vulnerability Scores (AVS), and 4-category influence distributions across Configurations 1–4.</div>', unsafe_allow_html=True)

    # Available output files
    outputs_dir = REPO_ROOT / "outputs"
    json_files = list(outputs_dir.glob("*.json")) if outputs_dir.exists() else []

    if not json_files:
        st.warning("No recorded experiment JSON files found in `outputs/`. Please run an experiment script first.")
    else:
        file_map = {f.name: f for f in json_files}
        default_idx = 0
        preferred = ["adversarial_experiment_report.json", "benchmark_comparison_full300.json"]
        for p in preferred:
            if p in file_map:
                default_idx = list(file_map.keys()).index(p)
                break

        selected_file_name = st.selectbox("Select Experiment Report File:", list(file_map.keys()), index=default_idx)
        selected_file_path = file_map[selected_file_name]

        with open(selected_file_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)

        is_mock_report = (exp_data.get("provider") == "mock")
        if is_mock_report:
            st.info("💡 **Dataset Notice (Mock Mode)**: This experiment was generated using the deterministic `mock` provider to prove pipeline mechanics, cohort alignment, and metric algorithms without paid API calls.")
        else:
            st.success(f"🟢 **Live LLM Evaluation Report**: Provider: `{exp_data.get('provider')}` | Model: `{exp_data.get('model')}`")

        # Top summary card
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{exp_data.get("sample_size", len(exp_data.get("cohort_claim_ids", [])))}</div><div class="metric-lbl">Matched Cohort Claims</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{exp_data.get("provider", "unknown").upper()}</div><div class="metric-lbl">Model Provider</div></div>', unsafe_allow_html=True)
        with s3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{exp_data.get("adversary_persona", "contrarian")}</div><div class="metric-lbl">Adversary Persona</div></div>', unsafe_allow_html=True)
        with s4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{exp_data.get("attack_strength", "assertive")}</div><div class="metric-lbl">Attack Strength</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        benchmarks = exp_data.get("benchmarks") or exp_data.get("configurations")
        if benchmarks:
            st.markdown("### 1. Comparative Performance & Robustness Metrics")
            table_rows = []
            for cfg_name, d in benchmarks.items():
                c_acc = d.get("clean_accuracy", d.get("accuracy", 0.0)) * 100
                c_ci = d.get("clean_ci", (0.0, 0.0))
                c_ci_str = f"[{c_ci[0]*100:.0f}%-{c_ci[1]*100:.0f}%]" if c_ci != (0.0, 0.0) else "N/A"

                if "adv_accuracy" in d:
                    a_acc = d["adv_accuracy"] * 100
                    a_ci = d.get("adv_ci", (0.0, 0.0))
                    a_ci_str = f"[{a_ci[0]*100:.0f}%-{a_ci[1]*100:.0f}%]"
                    avs = f"{d.get('avs', 0.0)*100:+.1f}%"
                    a_fcr = f"{d.get('adv_fcr', 0.0)*100:.1f}%" if d.get("adv_fcr") is not None else "N/A"
                    afr = f"{d.get('flip_metrics', {}).get('adversarial_flip_rate', 0.0)*100:.1f}%"
                else:
                    a_acc = None
                    a_ci_str = "N/A"
                    avs = "N/A"
                    a_fcr = "N/A"
                    afr = "N/A"

                c_fcr = f"{d.get('clean_fcr', d.get('false_consensus_rate', 0.0))*100:.1f}%" if d.get('clean_fcr') is not None or d.get('false_consensus_rate') is not None else "N/A"

                table_rows.append({
                    "Configuration": cfg_name,
                    "Clean Accuracy": f"{c_acc:.1f}% {c_ci_str}",
                    "Adversarial Accuracy": f"{a_acc:.1f}% {a_ci_str}" if a_acc is not None else "N/A (Single)",
                    "AVS (Degradation)": avs,
                    "Clean FCR": c_fcr,
                    "Adversarial FCR": a_fcr,
                    "Flip Rate (AFR)": afr
                })

            st.table(table_rows)

            # Four-Category Influence Breakdown
            st.markdown("### 2. Four-Category Adversarial Influence Breakdown")
            cat_rows = []
            for cfg_name, d in benchmarks.items():
                if "flip_metrics" not in d:
                    continue
                fm = d["flip_metrics"]
                cats = fm.get("categories", {})
                denom = fm.get("clean_correct_count", 0)
                c1 = cats.get("category_1_corrupted", 0)
                c2 = cats.get("category_2_changed_remained_correct", 0)
                c3 = cats.get("category_3_no_effect_unchanged", 0)
                c4 = cats.get("category_4_baseline_already_incorrect", 0)
                tot = exp_data.get("sample_size", 30)

                cat_rows.append({
                    "Configuration": cfg_name,
                    "Cat 1: Corrupted (Clean OK -> Adv Bad)": f"{c1} / {denom} ({fm.get('adversarial_flip_rate', 0.0)*100:.1f}%)" if denom > 0 else f"{c1} (Denom=0)",
                    "Cat 2: Changed / Remained OK": f"{c2} ({c2/tot*100:.1f}%)",
                    "Cat 3: Unchanged / No Effect": f"{c3} ({c3/tot*100:.1f}%)",
                    "Cat 4: Baseline Already Incorrect": f"{c4} ({c4/tot*100:.1f}%)",
                    "Total Matched Claims": tot
                })

            if cat_rows:
                st.table(cat_rows)
            else:
                st.info("Influence breakdown categories available in adversarial experiment reports.")

            # Per-claim analysis browser
            st.markdown("### 3. Matched Cohort Claim Browser")
            cfg_options = [k for k in benchmarks.keys() if "flip_metrics" in benchmarks[k]]
            if cfg_options:
                target_cfg = st.selectbox("Inspect Claims for Configuration:", cfg_options, index=len(cfg_options)-1)
                claims_data = benchmarks[target_cfg]["flip_metrics"].get("per_claim_analysis", [])

                if claims_data:
                    search_id = st.text_input("Filter by Claim ID or text keyword:", "")
                    filtered = [
                        c for c in claims_data
                        if not search_id or str(c["id"]) in search_id or search_id.lower() in c.get("claim", "").lower()
                    ]
                    st.write(f"Displaying **{len(filtered)}** claims:")
                    st.dataframe(
                        [
                            {
                                "ID": c["id"],
                                "Gold": c["gold_label"],
                                "Clean Pred": c["clean_predicted_label"],
                                "Adv Pred": c["adversarial_predicted_label"],
                                "Changed?": "YES" if c["prediction_changed"] else "NO",
                                "Category": f"Cat {c['category']} ({c['category_name']})",
                                "Claim Text": c.get("claim", "")[:90] + "..."
                            }
                            for c in filtered
                        ],
                        use_container_width=True
                    )


# ==============================================================================
# MODE 3: ARCHITECTURE & RESEARCH NOTES
# ==============================================================================
else:
    st.markdown('<div class="main-title">ConsensusX: Research Architecture & Objectives</div>', unsafe_allow_html=True)
    st.markdown("""
### Final-Year AIML Demonstration Overview
**ConsensusX** addresses the critical failure mode of standard multi-agent debate: **Majority Sycophancy and False Consensus**.
In standard LLM committees, confident, contrarian agents can mislead honest agents into false consensus.

#### Four Comparable Configurations:
1. **Config 1: Single-Agent Baseline**: Standard zero-shot verification without peer interaction.
2. **Config 2: Majority Voting**: $N$ independent verifiers voting concurrently with confidence-weighted tie-breaking.
3. **Config 3: Structured Debate**: Multi-turn cross-examination where agents inspect peer arguments and rebut counter-claims.
4. **Config 4: ConsensusX**: Debate panel verified by an **Evidence-Based Judge** with FAISS retrieval grounding and **Adaptive Stopping**.

#### Epistemic Safety Override:
When an adversary injects fabricated citations (e.g. `[Counter_Archive_Wiki:1]`), the judge cross-references IDs against retrieved Wikipedia documents in FAISS.
Nonexistent passage IDs trigger an automatic safety downgrade to `NOT ENOUGH INFO`, neutralizing false consensus.
""")

st.sidebar.markdown("---")
st.sidebar.caption("ConsensusX Research Prototype v1.0.0 | Kavya Gada")

