# Handover: Running the ML Agent on a GPU Machine

## Quick start

```bash
# 1. Clone the repo
git clone <repo-url>
cd TiktokTechJam2026

# 2. Create a Python 3.12 venv
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install PyTorch with CUDA first
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Install everything else
pip install -r requirements.txt

# 5. Verify CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expect: 2.6.x+cu124 True
```

## Dataset setup

The dataset is NOT in the repo (too large). Copy the `rec_datasets/` folder from
the laptop or download it fresh:

```
rec_datasets/
  KuaiRand-1K/
    data/
      log_standard_4_08_to_4_21_1k.csv
      log_standard_4_22_to_5_08_1k.csv
      social_network_1k.csv
      user_features_1k.csv
      video_features_basic_1k.csv
      video_features_statistic_1k.csv
```

All 6 CSV files must be present. The agent will fail silently if any are missing.

## Environment variables

Create a `.env` file in the repo root:

```bash
# Required — OpenAI API key (the agent uses gpt-5.5 by default)
OPENAI_API_KEY=sk-...

# Model (default: gpt-4o). We've been using gpt-5.5:
AGENT_MODEL=gpt-5.5

# Pricing for gpt-5.5 (for accurate cost tracking):
AGENT_INPUT_COST_PER_M=2.50
AGENT_OUTPUT_COST_PER_M=10.00

# Device auto-detects CUDA. Force CPU if needed:
# HARNESS_DEVICE=cpu

# Convergence floor (default 30 — don't lower this):
# HARNESS_MIN_SCORED=30
```

## Running the agent

### Terminal 1 — the agent (overnight run)

```bash
python run_overnight.py --dataset 1k --run-name record-run
```

This will:
- Create `logs-1k/record-run-N/` (auto-increments the number)
- Run up to 50 experiments (organiser hard cap)
- Stop on convergence (3 experiments without 0.002 improvement after 30 scored)
- Stop after 6 hours wall clock
- Auto-restart on crash (up to 5 times, with backoff)
- Exit cleanly on normal finish

### Terminal 2 — live watcher (optional)

```bash
python harness/watch.py --dataset 1k
```

Shows each experiment result as it lands, with running best and delta from baseline.

### Single status check

```bash
python harness/watch.py --dataset 1k --run-dir logs-1k/record-run-3
```

## What the agent does

It's a GPT-5.5-powered experiment loop that:
1. Reads its full experiment history (ledger) each iteration
2. Proposes a hypothesis and writes a standalone Python solution
3. The harness runs the solution on 3 random seeds, scores with the official evaluator
4. Results are logged; the agent sees them next iteration and decides what to try next

The agent searches over ML techniques: loss functions (BPR, focal, listwise),
features (time, auxiliary signals), architectures (DeepFM, attention),
ensembles, and post-processing (z-score normalisation, stacking).

## Key numbers

| | KuaiRand-1K |
|---|---|
| FM baseline (valid primary) | 0.6451 |
| Target (baseline + 0.002) | 0.6471 |
| Oracle ceiling | 0.8484 |
| Best so far (run 2, iter 41) | 0.6617 |
| Run cost (50 experiments) | ~$5 |
| Run time (CPU) | ~6 hours |
| Run time (GPU) | should be faster |

## Anti-cheat guardrail

The harness now flags any score > 0.90 as label leakage (status = `cheating`).
Run 2 had the agent discover it could read `long_view` from the raw CSV and get
0.9974. That exploit is now blocked in both the prompt (forbidden rule) and the
code (automatic detection).

## Flags reference

```
python -m agent --help

--dataset {pure,1k,27k}   Which KuaiRand variant (default: pure)
--run-name PREFIX          Run folder prefix, auto-numbered (default: run)
--run-id NAME              Exact folder name, skip auto-numbering
--max-iter N               Max iterations (default: 100, but 50-experiment cap fires first)
--supervised               Pause after each iteration for human approval
```

## If something goes wrong

- **Agent crashes repeatedly**: Check `logs-1k/record-run-N/events.jsonl` for error details
- **All experiments fail**: Usually a missing package or wrong data path. Check stderr in the JSON records
- **Scores suspiciously high (>0.90)**: The guardrail should catch this. If not, check the solution code for label leakage
- **"No baseline measured"**: The `--dataset` flag doesn't match what's configured. Use `1k` for KuaiRand-1K
- **Convergence too early**: Set `HARNESS_MIN_SCORED=30` in `.env` (this is the default)
