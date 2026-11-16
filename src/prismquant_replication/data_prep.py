from __future__ import annotations
## loading the calibration dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_wikitext_text(split: str) -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    lines = [line for line in ds["text"] if line.strip()]
    return "\n\n".join(lines)

def tokenize_text(tokenizer: AutoTokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids

def calibration_batches(ids: torch.Tensor, samples: int, seqlen: int) -> list[torch.Tensor]:
    needed = samples * seqlen
    if ids.size(1) < needed:
        raise ValueError(f"need {needed} calibration tokens, only {ids.size(1)} available")
    chunk = ids[:, :needed].reshape(samples, seqlen)
    return [chunk[i : i + 1] for i in range(samples)]
