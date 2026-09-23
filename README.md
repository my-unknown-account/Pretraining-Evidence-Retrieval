<div align="center">

# How Pretraining Evidence Relates to LLM Reliance on External Context

🧠 Code, data, prompts, analyses, and human-evaluation materials for studying when
retrieval fixes factual errors and when it corrupts facts a model already knew.

</div>

---

## 🔎 In One Sentence

Facts with weaker observable pretraining evidence are more sensitive to
retrieved context: they are easier to correct with true evidence, but also
easier to corrupt with plausible contradictory evidence.

## ⚖️ Why This Matters

Retrieval-augmented generation is usually evaluated as a way to improve factual
answers. This project treats retrieval as a two-sided intervention:

| Retrieval condition | Question | Desired behavior | Failure mode |
| --- | --- | --- | --- |
| 🧠 Closed book | What does the model answer from memory? | Correct factual recall | Wrong or unsupported answer |
| ✅ Correct context | Does true evidence repair an error? | Correction | The model ignores evidence |
| ⚠️ Contradictory context | Does false evidence corrupt knowledge? | Resistance | The model adopts the false object |

The core claim is not simply that retrieval helps or hurts. It is that the same
facts that are easiest to repair can also be the easiest to destabilize.

## 🧭 Research Questions

| ID | Focus | Script |
| --- | --- | --- |
| RQ1 | 🛠️ Corrective retrieval vs. pretraining evidence | `scripts/analysis/analyze_rq1.py` |
| RQ2 | ⚠️ Contradictory retrieval vs. pretraining evidence | `scripts/analysis/analyze_rq2.py` |
| RQ3 | 📊 Semantic vs. lexical evidence signals | `scripts/analysis/analyze_rq3.py` |
| RQ4 | 📈 OLMo-3-7B vs. OLMo-3-32B scale comparison | `scripts/analysis/analyze_rq4.py` |

## 📡 Evidence Signals

The project compares surface-level corpus evidence, semantic corpus evidence,
and behavior-derived confidence.

| Signal | Meaning |
| --- | --- |
| `Lex-SO` | 🔤 Count of passages where subject and object aliases co-occur |
| `Lex-SRO` | 🔤 Count of passages where subject, relation wording, and object aliases co-occur |
| `RAS` | 📌 Count of passages that semantically support the full subject-relation-object fact |
| `Token logprob` | 🧮 Closed-book token-probability confidence |
| `Verbalized confidence` | 🗣️ Closed-book self-reported confidence |
| `P(true)` | ✅ Closed-book truth-probability confidence |
| `Self-consistency` | 🔁 Agreement across repeated closed-book generations |

`RAS` is intentionally stricter than lexical matching: a passage contributes
only when it clearly states or implies the complete factual proposition.

## 🏁 Headline Results

| Finding | Result |
| --- | --- |
| ✅ Correct retrieval repairs many errors | RCR: AmberChat-7B `0.773`, OLMo-3-7B `0.889`, RedPajama-7B `0.844` |
| ⚠️ Contradictory retrieval often corrupts correct answers | Rcp: AmberChat-7B `0.524`, OLMo-3-7B `0.744`, RedPajama-7B `0.614` |
| 🛡️ Stronger RAS reduces corruption risk | Highest-RAS facts are less likely to adopt the planted false object |
| 📊 Evidence and confidence signals are modest predictors | RAS macro AUROC: correction `0.568`, override `0.582` |
| 📈 Larger OLMo is more resistant | OLMo-3-32B lowers Rcp from `0.744` to `0.530` on shared facts |

## 🧪 Experimental Scale

| Model | Facts | Closed-book QA records | RQ1 eligible wrong | RQ2 eligible correct |
| --- | ---: | ---: | ---: | ---: |
| AmberChat-7B | 12,739 | 12,846 | 11,077 | 1,662 |
| RedPajama-7B | 12,309 | 12,516 | 9,942 | 2,367 |
| OLMo-3-7B | 12,788 | 12,934 | 11,255 | 1,533 |
| OLMo-3-32B | 12,788 | 12,934 | 9,489 | 1,249 |

Each fact is evaluated across 10 stochastic generation runs per model and
condition, then collapsed to one majority-selected QA record before analysis.

## 🗂️ Repository Map

```text
Pretraining-Evidence-Retrieval/
|-- data/                         # Dataset, exposure features, contradictions, Wikidata cache
|-- prompts/                      # Prompt text used in the paper
|-- scripts/
|   |-- pretraining_evidence/     # Dataset rebuilding
|   |-- counterfactual_passages/  # Counterfactual passage generation
|   |-- inference/                # Closed-book, correct-context, contradictory-context runs
|   `-- analysis/                 # RQ1-RQ4 analysis scripts
`-- human_evaluation/             # Filled workbooks and agreement scripts
```

## ⚡ Quickstart

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the checks that work immediately with the included human annotations:

```bash
python human_evaluation/ras/compute_agreement.py
python human_evaluation/rq2/compute_agreement.py
```

Full inference requires Hugging Face model access, GPU memory appropriate for
the selected checkpoint, and an OpenAI API key if you regenerate
counterfactual passages.

## 🔁 Full Pipeline

Run commands from the repository root unless noted otherwise.

### 1. 🧱 Rebuild The Dataset

`data/dataset.json` is already included. Rebuild it only after editing
`data/exposure_data.json` or `data/contradictions.json`.

```bash
python scripts/pretraining_evidence/clean_dataset.py
```

### 2. 🧪 Regenerate Counterfactual Passages

`data/contradictions.json` is already included. Regeneration requires
`OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY="..."
cd scripts/counterfactual_passages
bash generate.sh
cd ../..
```

### 3. 🤖 Generate Model Answers

Run each model under all three inference conditions:

```bash
python scripts/inference/closed_book/prompting.py --model amber --run_id 1 --batch_size 64
python scripts/inference/correct_context/prompting.py --model amber --run_id 1 --batch_size 64
python scripts/inference/contradictory_context/prompting.py --model amber --run_id 1 --batch_size 64
```

Repeat for `amber`, `redpajama`, `olmo`, and `olmo32`, across `run_id` values
`1` through `10`. Each inference directory also includes a `run.sh` launcher
for the full 10-run batch.

### 4. 🧑‍⚖️ Judge Answers

```bash
python scripts/inference/closed_book/compute_accuracy.py --model amber --q_type simple
python scripts/inference/correct_context/compute_accuracy.py --model amber
python scripts/inference/contradictory_context/compute_accuracy.py --model amber
```

Repeat for each model.

### 5. 🧮 Select Majority QA Records

Collapse the 10 sampled generations into one QA file by majority `is_correct`
label per fact/model/condition:

```bash
python scripts/inference/select_majority_results.py
```

This writes `scripts/inference/results.json`.

### 6. 📊 Run Analyses

RQ3 depends on outputs from RQ1 and RQ2.

```bash
python scripts/analysis/analyze_rq1.py
python scripts/analysis/analyze_rq2.py
python scripts/analysis/analyze_rq3.py
python scripts/analysis/analyze_rq4.py
```

## 📦 Output Locations

| Output type | Default location |
| --- | --- |
| 🤖 Model generations and labels | `scripts/inference/*/results/` and `scripts/inference/results.json` |
| 📊 RQ analysis outputs | `scripts/analysis/outputs/` |
| 📝 New annotation workbooks | `human_evaluation/*/workbooks/` |
| ✅ Filled human-evaluation workbooks | `human_evaluation/*/results/` |

Large model-generation outputs are not bundled unless explicitly placed in the
corresponding `results/` directories.

## 🧑‍🔬 Human Evaluation

The repository includes filled workbooks for two validation steps:

| Evaluation | Purpose | Agreement summary |
| --- | --- | --- |
| 📌 RAS support labels | Validates semantic support labeling for `RAS` | Individual agreement: `81.33`-`85.33`; majority vs. gold: `89.00` |
| ⚠️ RQ2 corruption labels | Validates the three-way contradictory-context judge | Overall agreement: `94.56` |

Run:

```bash
python human_evaluation/ras/compute_agreement.py
python human_evaluation/rq2/compute_agreement.py
```

## 🧠 Interpretation

The evidence scores in this repository measure observable support in
model-associated corpora. They should not be read as causal proof that a
specific passage produced a specific model behavior. The analyses are best
understood as fact-level observational tests of when retrieval is likely to
repair, corrupt, or leave answers unchanged.
