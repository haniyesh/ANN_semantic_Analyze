"""
Standalone embedding worker.

Run as a subprocess from xgboost_train_bert.py to isolate the HuggingFace
tokenizer's Rust allocator from PyTorch's caching allocator; running them in
the same process causes glibc tcache double-free on Linux with PyTorch 2.x.

Checkpoints every CKPT_EVERY rows to --output.ckpt.npy so a restart can
resume rather than re-embedding everything from scratch.
"""

import sys
import json
import argparse
import numpy as np

CKPT_EVERY = 8000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, help="HuggingFace model name or path")
    parser.add_argument("--titles",     required=True, help="Path to JSON file with list of title strings")
    parser.add_argument("--output",     required=True, help="Path to write .npy embedding array")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(args.titles) as f:
        titles = json.load(f)

    import os as _os
    ckpt_path = args.output + ".ckpt.npy"

    # Resume from checkpoint if available
    start_idx = 0
    embs = []
    if _os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        start_idx = len(ckpt)
        embs = [ckpt]
        print(f"  [{args.model}] resuming from checkpoint at row {start_idx}/{len(titles)}", flush=True)

    import torch
    torch.set_num_threads(1)
    from transformers import AutoTokenizer, AutoModel

    print(f"  [{args.model}] loading tokenizer+model on cpu (1 thread)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()

    n = len(titles)
    with torch.no_grad():
        for i in range(start_idx, n, args.batch_size):
            if i % 2000 == 0:
                pct = i * 100 // max(1, n)
                print(f"    {i}/{n} ({pct}%)...", flush=True)
            batch = titles[i : i + args.batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt",
            )
            out = model(**inputs).last_hidden_state[:, 0, :]
            embs.append(out.numpy())

            # Save checkpoint every CKPT_EVERY rows
            total_done = sum(len(e) for e in embs)
            if total_done % CKPT_EVERY < args.batch_size:
                ckpt_arr = np.vstack(embs).astype(np.float32)
                np.save(ckpt_path, ckpt_arr)

    emb = np.vstack(embs).astype(np.float32)
    np.save(args.output, emb)
    print(f"  Saved {emb.shape} → {args.output}", flush=True)

    # Remove checkpoint after successful save
    if _os.path.exists(ckpt_path):
        _os.unlink(ckpt_path)

    # Skip Python GC to avoid glibc tcache double-free when freeing BERT tensors.
    _os._exit(0)


if __name__ == "__main__":
    main()
