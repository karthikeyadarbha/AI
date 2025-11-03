#!/usr/bin/env python3
"""
Minimal PEFT soft prompt tuning example for SST-2 using t5-small.
Uses a SimpleSeq2SeqCollator that removes any stray 'label' key (scalar class)
and only provides 'labels' (padded tensor) to the model.

Supports:
 - discrete baseline (--mode discrete)
 - prompt tuning (--mode prompt)

Outputs saved in ./outputs/<run_id>

Example:
 python train_prompt_tuning.py --model_name t5-small --mode prompt --num_virtual_tokens 20 --subset 2000 --num_train_epochs 3
"""
from __future__ import annotations
import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import PromptTuningConfig, get_peft_model, TaskType

# ----------------------------
# Utilities
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="t5-small")
    p.add_argument("--mode", type=str, default="prompt", choices=["discrete", "prompt"])
    p.add_argument("--num_virtual_tokens", type=int, default=20)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_input_length", type=int, default=128)
    p.add_argument("--max_target_length", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--subset", type=int, default=0, help="limit train set; 0 means full")
    p.add_argument("--no_cuda", action="store_true", help="force CPU")
    p.add_argument("--debug", action="store_true", help="enable debug output")
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_batch(examples: Dict[str, List], tokenizer: AutoTokenizer, max_input_length: int, max_target_length: int):
    """
    Tokenize inputs and targets, return dict suitable for dataset.map (batched=True).
    Ensures labels are list-of-lists and pad token ids are replaced with -100.
    """
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = ["sst2 sentence: " + s for s in examples["sentence"]]
    targets = ["positive" if l == 1 else "negative" for l in examples["label"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    pad_id = tokenizer.pad_token_id
    normalized_labels = []
    for lab in labels["input_ids"]:
        # coerce numpy / torch / scalar -> list
        if isinstance(lab, np.ndarray):
            lab = lab.tolist()
        if isinstance(lab, torch.Tensor):
            lab = lab.tolist()
        if isinstance(lab, int):
            lab = [lab]
        if not isinstance(lab, (list, tuple)):
            try:
                lab = list(lab)
            except Exception:
                lab = [-100]
        # coerce elements to ints and replace pad with -100
        lab_ints = [(int(tok) if tok != pad_id else -100) for tok in lab]
        normalized_labels.append(lab_ints)

    model_inputs["labels"] = normalized_labels
    return model_inputs

def generate_dataset_splits(tokenizer: AutoTokenizer, args) -> tuple[Dict[str, Any], Dataset, Dataset, Dataset]:
    ds = load_dataset("glue", "sst2")
    if args.subset and args.subset > 0:
        ds["train"] = ds["train"].select(range(min(args.subset, len(ds["train"]))))
    tokenized_train = ds["train"].map(lambda ex: preprocess_batch(ex, tokenizer, args.max_input_length, args.max_target_length), batched=True)
    tokenized_val = ds["validation"].map(lambda ex: preprocess_batch(ex, tokenizer, args.max_input_length, args.max_target_length), batched=True)
    tokenized_test = ds["test"].map(lambda ex: preprocess_batch(ex, tokenizer, args.max_input_length, args.max_target_length), batched=True)

    # Final defensive per-row normalization: ensure labels are python lists and remove raw 'label' if present
    def _fix_row(row):
        lab = row.get("labels", None)
        if lab is None:
            # if labels missing but 'label' (scalar) exists, create labels from it
            if "label" in row:
                scalar = row["label"]
                try:
                    row["labels"] = [int(scalar)]
                except Exception:
                    row["labels"] = [-100]
                # optionally remove the scalar to avoid being forwarded to model
                row.pop("label", None)
                return row
            return row
        # elementwise coercion
        if isinstance(lab, (np.ndarray,)):
            lab = lab.tolist()
        if isinstance(lab, (int, np.integer)):
            lab = [int(lab)]
        if isinstance(lab, (list, tuple)):
            lab = [int(x) if not (x is None) else -100 for x in lab]
        else:
            try:
                lab_list = list(lab)
                lab = [int(x) for x in lab_list]
            except Exception:
                lab = [-100]
        # replace pad ids in-case any slipped through (pad_id should exist)
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            lab = [(tok if tok != pad_id else -100) for tok in lab]
        row["labels"] = lab
        # ensure raw scalar 'label' doesn't exist
        row.pop("label", None)
        return row

    tokenized_train = tokenized_train.map(_fix_row, batched=False)
    tokenized_val = tokenized_val.map(_fix_row, batched=False)
    tokenized_test = tokenized_test.map(_fix_row, batched=False)
    return ds, tokenized_train, tokenized_val, tokenized_test

# ----------------------------
# Simple collator
# ----------------------------
class SimpleSeq2SeqCollator:
    """
    Simple collator for seq2seq tasks that:
    - pads inputs via tokenizer.pad(...)
    - pads labels manually into a LongTensor and converts pad_token_id -> -100
    - removes any stray 'label' scalar from features so the model never receives unexpected kwargs
    """
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _coerce_to_list_of_ints(self, lab) -> List[int]:
        # handle torch.Tensor / np.ndarray / numpy scalars / int / list / tuple / generator
        if isinstance(lab, torch.Tensor):
            lab = lab.cpu().numpy()
        if isinstance(lab, np.ndarray):
            # 0-d array yields shape == ()
            if lab.shape == ():
                return [int(lab.item())]
            try:
                lab_list = lab.tolist()
            except Exception:
                return [-100]
            lab = lab_list
        if isinstance(lab, (np.integer,)):
            return [int(lab)]
        if isinstance(lab, int):
            return [int(lab)]
        if isinstance(lab, str):
            # unexpected: treat as bad -> fallback
            return [-100]
        # Try to convert iterable -> list
        try:
            candidate = list(lab)
        except Exception:
            try:
                return [int(lab)]
            except Exception:
                return [-100]
        out = []
        for el in candidate:
            try:
                if isinstance(el, (np.integer,)):
                    out.append(int(el))
                else:
                    out.append(int(el))
            except Exception:
                out.append(-100)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # prepare input features for tokenizer.pad (exclude both 'labels' and raw 'label')
        features_no_label_keys = [{k: v for k, v in f.items() if k not in ("labels", "label")} for f in features]
        batch = self.tokenizer.pad(
            features_no_label_keys,
            padding="longest",
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Build label sequences as tensors (coerce from either 'labels' or scalar 'label')
        pad_id = self.tokenizer.pad_token_id
        label_tensors = []
        for f in features:
            # prefer 'labels' (tokenized target ids). If absent, fall back to 'label' scalar (class)
            lab = f.get("labels", None)
            if lab is None and "label" in f:
                lab = f["label"]
            if lab is None:
                label_tensors.append(torch.tensor([], dtype=torch.long))
                continue
            lab_list = self._coerce_to_list_of_ints(lab)
            label_tensors.append(torch.tensor(lab_list, dtype=torch.long))

        if len(label_tensors) > 0:
            padded = pad_sequence(label_tensors, batch_first=True, padding_value=(pad_id if pad_id is not None else 0))
            # convert pad token id to -100 for loss ignore
            if pad_id is not None:
                padded[padded == pad_id] = -100
            else:
                padded[padded == 0] = -100
            batch["labels"] = padded
        else:
            batch["labels"] = torch.zeros((len(features), 0), dtype=torch.long)

        return batch

# ----------------------------
# Baseline & training
# ----------------------------
def run_discrete_baseline(tokenizer: AutoTokenizer, dataset: Dict[str, Any], args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    def gen_and_score(split):
        inputs = ["sst2 sentence: " + s for s in split["sentence"]]
        enc = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model.generate(**enc, max_length=args.max_target_length)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels = ["positive" if l == 1 else "negative" for l in split["label"]]
        acc = sum(1 if p.strip() == l.strip() else 0 for p, l in zip(preds, labels)) / len(preds)
        return acc, None

    return gen_and_score(dataset["validation"]), gen_and_score(dataset["test"])

def train_prompt_tuning(tokenizer: AutoTokenizer, ds, tokenized_train, tokenized_val, tokenized_test, args, run_id):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    # configure prompt tuning (PEFT)
    prompt_config = PromptTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM, num_virtual_tokens=args.num_virtual_tokens)
    model = get_peft_model(model, prompt_config)

    data_collator = SimpleSeq2SeqCollator(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.output_dir}/{run_id}",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        save_total_limit=2,
        predict_with_generate=False,
        fp16=False,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f"{args.output_dir}/{run_id}")

    # Evaluate via generation (simple)
    def generate_and_score(split):
        model.eval()
        accs = []
        for i in range(0, len(split), args.per_device_eval_batch_size):
            batch = split[i : i + args.per_device_eval_batch_size]
            inputs = tokenizer(["sst2 sentence: " + s for s in batch["sentence"]], truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=args.max_target_length)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels = ["positive" if l == 1 else "negative" for l in batch["label"]]
            for p, l in zip(preds, labels):
                accs.append(1 if p.strip() == l.strip() else 0)
        acc = sum(accs) / len(accs)
        ece = 0.0  # placeholder: computing ECE requires per-example confidence extraction
        return acc, ece

    val_acc, val_ece = generate_and_score(ds["validation"])
    test_acc, test_ece = generate_and_score(ds["test"])
    return {"val_acc": val_acc, "val_ece": val_ece, "test_acc": test_acc, "test_ece": test_ece, "out_dir": f"{args.output_dir}/{run_id}"}

# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    for seed_offset in range(args.num_seeds):
        current_seed = args.seed + seed_offset
        set_seed(current_seed)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"run_{args.mode}_{args.model_name.replace('/', '_')}_vt{args.num_virtual_tokens}_s{current_seed}_{timestamp}"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n=== RUN {run_id} (seed {current_seed}) ===")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds, tokenized_train, tokenized_val, tokenized_test = generate_dataset_splits(tokenizer, args)

        if args.debug:
            # write a few label-type samples for inspection
            dbg_dir = Path(args.output_dir) / run_id
            dbg_dir.mkdir(parents=True, exist_ok=True)
            def sample_labels(tok_dataset, name, n=20):
                entries = []
                for i in range(min(n, len(tok_dataset))):
                    lab = tok_dataset[i].get("labels", None)
                    entries.append({"idx": i, "type": type(lab).__name__, "repr": lab if isinstance(lab, list) and len(str(lab)) < 200 else str(type(lab))})
                with open(dbg_dir / f"label_types_{name}.json", "w") as f:
                    json.dump(entries, f, indent=2)
            sample_labels(tokenized_train, "train")
            sample_labels(tokenized_val, "val")
            sample_labels(tokenized_test, "test")
            print(f"[debug] wrote label_types files to {dbg_dir}")

        if args.mode == "discrete":
            (val_metrics, _), (test_metrics, _) = run_discrete_baseline(tokenizer, ds, args)
            result = {"run_id": run_id, "val_acc": val_metrics, "test_acc": test_metrics}
        elif args.mode == "prompt":
            metrics = train_prompt_tuning(tokenizer, ds, tokenized_train, tokenized_val, tokenized_test, args, run_id)
            result = {"run_id": run_id, **metrics}
        else:
            raise ValueError("unknown mode")

        # save summary
        out_dir = Path(args.output_dir) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(result, f, indent=2)

        print("Result:", result)

    print("\n=== ALL RUNS COMPLETE ===")

if __name__ == "__main__":
    main()