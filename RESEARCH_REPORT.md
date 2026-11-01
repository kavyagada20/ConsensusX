# ConsensusX: Epistemic Evidence-Based Judging in Multi-Agent Fact Verification
## Master Research & Engineering Evaluation Report

**Candidate:** Final-Year AIML Capstone Project  
**Repository:** `ConsensusX`  
**Development Branch:** `consensusx-development`  
**Verification Date:** November 2026  
**Automated Test Suite Status:** **95 / 95 Passing** (`python -m pytest` in 6.94s)

---

## 1. Problem Statement & Motivation

Recent advancements in large language models (LLMs) have led to widespread adoption of multi-agent debate and majority voting ensembles for complex reasoning and factual verification. However, standard multi-agent deliberation architectures suffer from three critical epistemic failure modes:

1. **Sycophancy and Groupthink**: LLM agents often concede to persuasive peer arguments regardless of empirical truth, converging to shared errors.
2. **False Consensus**: When multiple agents reach unanimous agreement based on ungrounded pre-training priors or hallucinated facts, voting mechanisms blindly accept the majority verdict as ground truth.
3. **Adversarial Vulnerability**: In the presence of a contrarian or deceptive agent that asserts false claims or manufactures fake citations with high rhetorical confidence, standard voting and debate committees frequently flip from correct to incorrect verdicts.

### Research Objective
ConsensusX investigates whether introducing a **neutral, evidence-grounded epistemic judge** with **strict citation verification** and **adaptive stopping** can neutralize deceptive arguments and suppress false consensus without sacrificing decision accuracy.

---

## 2. System Architecture & Workflow

ConsensusX organizes deliberation into four sequential stages:

```
[Claim Input]
     │
     ▼
[Dense RAG Retriever] ──── Top-k Passages with Stable IDs (e.g. [Firefox:1])
     │
     ▼
[Stage 1: Isolated Round 1 Judgment]
     ├─ Agent 1 (Affirmative stance prompt)  ─┐
     ├─ Agent 2 (Skeptic stance prompt)     ─┼─ Stances committed BEFORE any argument sharing
     └─ Agent 3 / Adversary                 ─┘
     │
     ▼
[Stage 2: Multi-Round Structured Debate]
     ├─ Turn-taking cross-examination (Rounds 1..R)
     ├─ Peer argument inspection & counter-rebuttal
     └─ Convergence tracking (opinion shifts recorded)
     │
     ▼
[Stage 3: Evidence-Based Judge & Citation Audit]
     ├─ Cross-references cited IDs against retrieved corpus
     ├─ Flags nonexistent or malformed citation IDs
     ├─ Evaluates directional entailment (SUPPORTS / REFUTES)
     └─ Epistemic Safety Override: Downgrades ungrounded consensus to NOT ENOUGH INFO
```

---

## 3. Four Comparable Experimental Configurations

To evaluate the contribution of each architectural component, ConsensusX implements four configurations evaluated on the **exact same deterministic cohort** of FEVER claims:

| Config | Name | Topology | Aggregation / Decision Rule | Epistemic Grounding |
| :---: | :--- | :--- | :--- | :--- |
| **Config 1** | **Single-Agent Baseline** | 1 Agent | Direct zero-shot / unassisted inference | No debate; relies solely on internal parametric knowledge |
| **Config 2** | **Independent Majority Voting** | 3 Agents | Plurality vote with confidence-weighted tie-breaker | Agents vote independently without cross-examination |
| **Config 3** | **Structured Debate** | 3 Agents ($R \le 2$) | Unweighted majority of final-round stances | Agents debate and critique peer arguments; vulnerable to rhetorical manipulation |
| **Config 4** | **ConsensusX (Ours)** | 3 Agents + 1 Judge | Evidence-grounded adjudication + Adaptive stopping | **Mandatory citation verification; overrides consensus if evidence is lacking** |

---

## 4. Evaluation Metrics & Mathematical Formulations

### A. Core Classification Metrics
- **Accuracy**: Fraction of claims where predicted label equals gold label ($y_i = \hat{y}_i$).
- **Macro-F1**: Unweighted harmonic mean of Precision and Recall across classes:
  $$\text{Macro-F1} = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
  where $C = \{\text{SUPPORTS}, \text{REFUTES}, \text{NOT ENOUGH INFO}\}$.
- **Wilson Score 95% Confidence Interval**:
  $$w = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z = 1.95996)$$

### B. Epistemic & Deliberation Metrics
- **False Consensus Rate (FCR)**: Proportion of claims where agents formed unanimous or majority consensus on an incorrect verdict:
  $$\text{FCR} = \frac{\sum_{i=1}^N \mathbf{1}(\text{consensus}_i = \text{True} \land \hat{y}_i \neq y_i)}{\sum_{i=1}^N \mathbf{1}(\text{consensus}_i = \text{True})}$$
- **Citation Grounding Rate**: Percentage of decisions citing exclusively valid passage IDs present in the retrieved context:
  $$\text{Grounding Rate} = \frac{N_{\text{valid citations}}}{N_{\text{total decisions with citations}}}$$

### C. Adversarial Robustness Metrics
- **Adversarial Vulnerability Score (AVS)**:
  $$\text{AVS} = \text{Accuracy}_{\text{clean}} - \text{Accuracy}_{\text{adversarial}}$$
- **Adversarial Flip Rate (AFR)**: Frequency with which an injected adversary converts a correct baseline decision into an incorrect one:
  $$\text{AFR} = \frac{\text{Cat 1}}{\text{Cat 1} + \text{Cat 2} + \text{Cat 3}}$$
  where:
  - **Category 1 (Corrupted to Incorrect)**: Clean was correct, attack caused flip to incorrect (Numerator of AFR).
  - **Category 2 (Changed Remained Correct)**: Clean was correct, attack caused shift but verdict remained correct.
  - **Category 3 (No Effect / Unchanged)**: Clean was correct, attack failed to sway the outcome.
  - **Category 4 (Baseline Already Incorrect)**: Excluded from AFR denominator to prevent distortion.

---

## 5. Experimental Results & Verification Audit

### A. Evidence Retrieval Evaluation
- **Evaluation Dataset**: 200 verifiable claims (100 `SUPPORTS`, 100 `REFUTES`) from FEVER dev split (`dev_sample_300.jsonl`).
- **Ground Truth Relevance**: Defined by exact `(wikipedia_title, sentence_id)` coordinates from FEVER gold annotations.
- **Corpus**: 319 unique Wikipedia passages indexed into a FAISS vector store.
- **Embedding Methodology**: Evaluated using deterministic token-hashing & character n-gram projection (`HashEmbeddingModel`, 128-d).
- **Execution Mode**: **[MOCK / LOCAL DETERMINISTIC]**

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Recall@5** | **81.00%** | 162 out of 200 verifiable claims retrieved at least one gold sentence in top-5. |
| **MRR@5** | **0.6903** | Relevant evidence appears on average between rank 1 and rank 2. |
| **nDCG@5** | **0.7039** | Strong ranking quality placing gold passages near the top. |
| **Precision@5** | **0.2100** | On average $\sim 1.05$ of the top-5 retrieved sentences are gold annotations. |
| **Unretrieved Queries** | **38 / 200 (19%)** | Occurs on paraphrased claims where lexical n-gram overlap with Wikipedia text is low. |

> [!NOTE]
> For claims labeled `NOT ENOUGH INFO`, no gold evidence exists in the corpus by definition. The retriever returns 5 distractor passages. An uncalibrated system hallucinates support from these distractors, whereas ConsensusX correctly detects that they fail to entail the claim and abstains.

---

### B. Controlled Adversarial Benchmark Evaluation
- **Evaluation Cohort**: Deterministic 10-claim stratified sample (Seed = 42, `data/processed/dev_sample_300.jsonl`).
- **Adversary Profile**: `persuasive_contrarian` (assertive attack strength, target inversion).
- **Cohort Alignment Invariant**: 100% matched claim IDs across Configs 1, 2, 3, and 4 (asserted zero skipped or reordered claims).
- **Provider & Model**: `mock-model` (offline pipeline contract verification).
- **Execution Mode**: **[MOCK / CONTRACT VERIFICATION]**

| Configuration | Clean Accuracy [95% CI] | Adv Accuracy [95% CI] | AVS | Clean FCR | Adv FCR | AFR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1: Baseline** | 40.0% [17%–69%] | N/A | N/A | N/A | N/A | N/A |
| **Config 2: Majority Voting** | 40.0% [17%–69%] | 40.0% [17%–69%] | +0.0% | 60.0% | 60.0% | 0.0% |
| **Config 3: Debate** | 40.0% [17%–69%] | 40.0% [17%–69%] | +0.0% | 60.0% | 60.0% | 0.0% |
| **Config 4: ConsensusX** | 40.0% [17%–69%] | 40.0% [17%–69%] | +0.0% | 0.0% | 0.0% | 0.0% |

> [!IMPORTANT]
> **Scientific Integrity Disclosure on Mock Results**:  
> In mock mode, agents produce deterministic heuristic responses. The identical 40% accuracy and 0% flip rates demonstrate that the pipeline routing, tie-breaking logic, and cohort assertion contracts function without bugs. They do **not** serve as empirical proof of LLM cognitive behavior.

---

### C. Live Gemini 2.5 Flash Pilot Evaluation
- **Provider**: Google GenAI (`google-genai` SDK)
- **Model**: `gemini-2.5-flash` (Temperature = 0.0, Thinking Budget = 0)
- **API Status**: Active key on Google Cloud Free Tier
- **Execution Mode**: **[LIVE API / EMPIRICAL EXECUTION]**

#### 1. Baseline Pilot (Config 1, 3 Claims)
- **File**: `outputs/live_pilot_baseline_3.json`
- **Results**:
  - Claim 101932: *"Firefox is a piece of hardware."* $\to$ Gold: `REFUTES`, Pred: `REFUTES` (Correct, Latency: 3.1s)
  - Claim 78162: *"Andrew Kevin Walker is American and Irish."* $\to$ Gold: `NOT ENOUGH INFO`, Pred: `NOT ENOUGH INFO` (Correct, Latency: 2.8s)
  - Claim 59964: *"Civilization IV received high ratings."* $\to$ Gold: `SUPPORTS`, Pred: `NOT ENOUGH INFO` (Incorrect, unassisted agent abstained due to lack of retrieval evidence)
- **Accuracy**: **66.7%** (2 / 3) | **Macro-F1**: **0.5556**

#### 2. ConsensusX Full Pipeline Pilot (Config 4, 1 Claim with RAG + Debate + Judge)
- **File**: `outputs/live_pilot_consensusx_1.json`
- **Claim 101932**: *"Firefox is a piece of hardware."*
  - **Gold Label**: `REFUTES`
  - **ConsensusX Predicted Label**: `REFUTES` (**Correct**)
  - **Cited Evidence**: `['Firefox:1']` (**100% Valid, Authentic Wikipedia Passage**)
  - **Judge Rationale**: *"The claim states that 'Firefox is a piece of hardware.' However, the reference evidence explicitly states, 'Firefox is a software application, not a piece of hardware.' [Firefox:1]. This directly contradicts the claim, unequivocally establishing it as false. All debaters correctly identified this direct contradiction and cited the relevant passage."*
  - **Grounding Rate**: **100.0%** | **False Consensus Rate**: **0.0%**
  - **Latency**: **21.84s** | **Cost**: **$0.001255 USD** (7,171 prompt tokens, 1,347 candidate tokens)

---

## 6. Real-World Limitations & Failure Modes

1. **Free-Tier Rate Limits (HTTP 429)**:  
   Google Cloud Free Tier limits requests to 5 RPM. A single ConsensusX claim with 3 debaters across 2 debate rounds plus a judge makes 6–7 API requests, exceeding the 5 RPM ceiling. An exponential backoff retry loop was added to `src/llm/gemini.py` that sleeps for ~16 seconds upon encountering 429 errors.
2. **Thinking Budget Interplay**:  
   In `gemini-2.5-flash`, internal reasoning tokens are counted against `max_output_tokens`. When `max_output_tokens=1024` and thinking is unconstrained, thinking tokens consume almost the entire budget, truncating JSON outputs. Disabling thinking (`thinking_budget=0`) and setting `max_output_tokens=2048` guarantees crisp, complete structured JSON responses.
3. **Strict Abstention on Multi-Hop Knowledge**:  
   ConsensusX requires explicit textual entailment or contradiction. If a true claim requires general background knowledge not present in the retrieved passages, the judge strictly outputs `NOT ENOUGH INFO`. This design choice prioritizes precision and hallucination prevention over ungrounded recall.

---

## 7. Guide Review & Demonstration Plan

To demonstrate the project during review:

```powershell
# 1. Launch the Streamlit Research Dashboard
streamlit run app.py
```

### Demonstration Script (3-Step Walkthrough)
1. **Live Claim Verification (SUPPORTS)**:
   - Select Claim: *"Civilization IV received high ratings."*
   - Select **Config 4 (ConsensusX)** with Mock or Gemini provider.
   - Show how the retriever pulls Wikipedia passage `[Civilization_IV:2]` and the judge validates the entailment.
2. **Live Claim Verification (REFUTES with Adversary)**:
   - Select Claim: *"Firefox is a piece of hardware."*
   - Enable **Adversary Injection** (`citation_distorter`).
   - Open the **Multi-Agent Turns** tab: point out the adversary claiming Firefox is hardware citing `[Hardware_Guide:99]`.
   - Open the **Judge & Epistemic Audit** tab: show how the judge flags `[Hardware_Guide:99]` as **unsupported/nonexistent** and overrides the adversary's deception.
3. **Recorded Benchmark Analytics**:
   - Switch to **Recorded Benchmark Analytics** tab.
   - Select `adversarial_experiment_report.json` or `benchmark_comparison_full300.json`.
   - Walk through the 95% Wilson confidence interval charts, False Consensus Rates, and the 4-category influence breakdown table.

