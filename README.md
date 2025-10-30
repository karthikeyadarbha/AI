# prompt-tune-quickstart — Codespaces soft prompt tuning (t5-small, SST-2)

Objective
- Run a minimal, reproducible soft (continuous) prompt tuning experiment on SST-2 using t5-small and PEFT inside GitHub Codespaces.
- Produce: saved prompt checkpoint, accuracy and ECE, single paraphrase robustness check, and a 1‑page report.

Repository layout
- devcontainer.json          # Codespaces/devcontainer config
- Dockerfile                # Devcontainer base image
- requirements.txt          # Python deps
- train_prompt_tuning.py    # Training & eval script
- README.md                 # This file

Quick notes about GitHub Codespaces resources
- Codespaces may not provide GPU by default on free plans. If you have a Codespaces plan that includes GPU machines, you can modify the Dockerfile to use a CUDA base and enable GPU support. If you do NOT have GPU, the workflow still runs on CPU—use smaller subsets and fewer epochs.
- For a fast proof-of-concept, use the `--subset` option (e.g., 2000 training examples) and small prompt length (10–20). This fits comfortably on CPU with longer wall time.

How to run in Codespaces (step-by-step)
1) Push this repo to GitHub under your account (e.g., karthikeyadarbha/prompt-tune-quickstart).
2) Open the repo page on GitHub and click "Code" -> "Open with Codespaces" -> "Create new codespace".
   - Choose machine type if your account allows (bigger RAM/CPU or GPU if available).
3) Codespaces will build the devcontainer automatically (this installs Python deps).
4) In the Codespace terminal, run a quick smoke test:
   python train_prompt_tuning.py --model_name t5-small --num_virtual_tokens 10 --subset 500 --num_train_epochs 1
5) For full-ish run (recommended small run):
   python train_prompt_tuning.py --model_name t5-small --num_virtual_tokens 20 --subset 2000 --num_train_epochs 3 --per_device_train_batch_size 4
6) Outputs:
   - Checkpoints and prompt artifacts saved under ./outputs/<run_id>
   - Console shows validation/test accuracy, ECE, and robustness check results.

Experiment variants to run (suggested)
- Baseline discrete prompt: run with --mode discrete (the script includes a simple discrete baseline).
- Soft prompt tuning: --mode prompt
- Optional full fine-tune: --mode finetune (only if you have GPU and time)

Logging & reproducibility
- Script supports --seed and --num_seeds to run multiple seeds; it reports mean ± std.
- Save outputs directory and hyperparameters for reproducibility.

Low-resource tips
- Use --subset 500–2000 for fast runs on CPU.
- Use num_virtual_tokens 10–20.
- Lower batch size and increase gradient_accumulation_steps if memory is tight.

What to report (1 page)
- Environment: Codespace machine, CPU/GPU, container image.
- Hyperparameters: model, prompt length, lr, batch size, epochs, seeds.
- Table: method | prompt len | val acc | test acc | ECE | robustness drop
- Short analysis and time/resource log.

```