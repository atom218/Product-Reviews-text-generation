# Product Reviews Text Generation

Multimodal sequence-to-sequence model that takes a **product image** and **control tokens** (sentiment, persona, aspect) and generates a **natural-language Amazon-style review**. The project is implemented as Jupyter notebooks intended for **Google Colab**, with artifacts stored on Google Drive under a `VisualReview` project folder.

## Overview

| Input | Output |
|-------|--------|
| Product image (`224×224`, ImageNet-normalized) | Token-by-token review text |
| Control prefix, e.g. `[S5] [P2] [A1]` | Review conditioned on rating, writing style, and topic |

The model does **not** use a pretrained vision–language backbone end-to-end. It combines a **custom ResNet-style image encoder**, a **cross-attention bridge**, and a **6-layer causal transformer decoder** (~45M trainable parameters) trained with next-token prediction.

## Repository contents

| File | Purpose |
|------|---------|
| [`dataset download.ipynb`](dataset%20download.ipynb) | Download **Electronics** reviews and metadata from [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) via Hugging Face `datasets` and save JSONL to Drive |
| [`Review_text_generation.ipynb`](Review_text_generation.ipynb) | Full **Appliances** pipeline: preprocessing, image download, BPE tokenizer, labeling, training, inference, evaluation, and Grad-CAM visualization |

There is no separate Python package; logic lives in the notebooks. Run cells in order (or restart the runtime and rerun from model/dataset definitions after edits).

## Data source

Reviews and product metadata come from **[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)** on Hugging Face.

The main training notebook uses the **Appliances** category (local JSONL paths on Drive). The download notebook targets **Electronics** as an alternate category for experimentation.

## Pipeline (`Review_text_generation.ipynb`)

### 1. Data preprocessing

- Load review and metadata JSONL files
- Join on `parent_asin`; extract main product image URL from metadata
- Filter: valid image URL, review length 50–1000 characters, non-null rating
- Keep products with **≥ 3 reviews**
- Save `processed/Appliances_joined.jsonl`

### 2. Image download and validation

- Download one image per product as `{parent_asin}.jpg` (parallel workers, resume-friendly)
- Validate images with PIL; drop rows with missing or corrupt files
- Save `processed/Appliances_final.jsonl`

### 3. Sampling and tokenizer

- Stratified cap: **60,000 reviews per star rating** → **300,000** total (`Appliances_400k.jsonl`)
- Train a **BPE tokenizer** (vocab size **16,000**) on review text with special tokens:
  - `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`
  - Control tokens: `[S1]`–`[S5]`, `[P1]`–`[P3]`, `[A1]`–`[A5]`
- Save `tokenizer/appliances_bpe_tokenizer.json`

### 4. Pseudo-labels (persona and aspect)

Heuristic labels are assigned from review text (for conditioning at train and inference time):

| Token | Meaning |
|-------|---------|
| `[S1]`–`[S5]` | Star rating (1–5) |
| `[P1]` | Budget-focused (price/value keywords) |
| `[P2]` | Enthusiast / technical (specs, performance keywords) |
| `[P3]` | Casual / default |
| `[A1]`–`[A5]` | Dominant aspect: performance, design, value, durability, ease of use |

Saved as `processed/Appliances_labeled.jsonl`.

### 5. Train / validation / test split

- Split by **product** (`parent_asin`), not by review, to avoid leakage  
- **80% / 10% / 10%** → ~240k / ~30k / ~30k reviews  
- Exports: `processed/train.jsonl`, `val.jsonl`, `test.jsonl`

### 6. Model architecture (`VisualReviewModel`)

```
Image (B, 3, 224, 224)
    → ImageEncoder (ResNet-style CNN)     → (B, 49, 512) spatial tokens
    → CrossAttentionBridge                → (B, 49, 512) K/V for decoder
    → Decoder (causal transformer)
         • token + positional embeddings
         • 6 × DecoderLayer (masked self-attention + cross-attention + FFN)
         • tied LM head (weight-shared with token embeddings)
    → logits (B, T, 16000)
```

**Hyperparameters (typical):**

| Setting | Value |
|---------|--------|
| `EMBED_DIM` | 512 |
| `NUM_HEADS` | 8 |
| `NUM_LAYERS` | 4–6 (notebook sections may differ) |
| `FF_DIM` | 2048 |
| `MAX_LENGTH` | 256 |
| `VOCAB_SIZE` | 16,000 |
| `BATCH_SIZE` | 64 |
| `LEARNING_RATE` | 3e-4 |
| `NUM_EPOCHS` | 20 |
| Optimizer | AdamW (`weight_decay=0.01`) |
| Schedule | Linear warmup + cosine decay |
| Mixed precision | CUDA AMP + `GradScaler` |
| Loss | `CrossEntropyLoss(ignore_index=-100)` on shifted labels |

Checkpoints are written to `checkpoints/` (best validation loss).

### 7. Dataset batch format

Each sample provides:

- `image` — `(3, 224, 224)`
- `input_ids` — teacher-forced prefix (control tokens + review), length 256
- `labels` — next-token targets; padding masked with `-100`
- `attention_mask` — `1` for real tokens, `0` for pad (used in decoder self-attention)

Sequences are built as strict autoregressive shifts: predict token `t+1` from tokens `≤ t`, with BOS/EOS from the tokenizer post-processor.

### 8. Inference and analysis

Later notebook sections include:

- `generate_review()` — autoregressive decoding with temperature and repetition penalty
- Side-by-side comparisons across control prefixes
- Training loss curves (`training_curve.png`)
- **Grad-CAM** over the image encoder for interpretability
- Notes toward extending the same pipeline to **Electronics**

## Expected Drive layout

Set `project_dir` in the notebooks (default Colab path):

```text
/content/drive/MyDrive/VisualReview/
├── Appliances.jsonl/              # raw review JSONL (Appliances)
├── meta_Appliances.jsonl/         # raw metadata JSONL
├── data/
│   ├── raw/                       # optional: Electronics from download notebook
│   └── images/                    # {parent_asin}.jpg
├── processed/
│   ├── Appliances_joined.jsonl
│   ├── Appliances_final.jsonl
│   ├── Appliances_400k.jsonl
│   ├── Appliances_labeled.jsonl
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── tokenizer/
│   └── appliances_bpe_tokenizer.json
├── checkpoints/
└── training_curve.png
```

For local runs, some cells use `~/VisualReview` instead of Drive; adjust `project_dir` consistently.

## Getting started (Google Colab)

1. **Mount Drive** and create `VisualReview` (see first cells in either notebook).
2. Place **Appliances** review and metadata JSONL on Drive (or run `dataset download.ipynb` for Electronics and adapt paths).
3. Open **`Review_text_generation.ipynb`** and run sequentially:
   - Preprocessing → images → tokenizer → labels → splits
   - Model definition cells → DataLoaders → sanity-check loss cell
   - Training loop
4. After changing model or dataset code, **Runtime → Restart session** and rerun definition cells so stale class definitions are not reused.
5. Before long training, confirm the sanity check reports a **non-zero** loss on a random batch (expect roughly **9–10** at initialization for fp32 CE over 16k classes).

## Dependencies

Install in Colab (or locally with CUDA):

```bash
pip install torch torchvision pandas numpy scikit-learn pillow requests tqdm tokenizers datasets matplotlib
```

Colab also uses `google.colab` for Drive mount. Hugging Face `datasets` is required for `dataset download.ipynb`.

## Training tips

- **Product-level splits** are required; do not shuffle reviews across products into train and val.
- Use **`attention_mask`** in the forward pass so padding does not affect self-attention.
- Keep loss computation in **float32** even when using AMP (`logits.float()` before `CrossEntropyLoss`).
- Image download is I/O-heavy; re-runs skip existing `{asin}.jpg` files.
- If loss stays at **0.0000** from epoch 1, rerun dataset and model cells after the latest fixes (label shift + mask wiring) and verify `real_tokens` in the debug cell is ≫ 0.

## Limitations

- Persona and aspect labels are **keyword heuristics**, not human annotations.
- Training is notebook-centric; there is no CLI or `requirements.txt` in the repo yet.
- Primary experiments use **Appliances**; Electronics download is a separate path in `dataset download.ipynb`.
- Generation quality depends on image coverage, filter thresholds, and training budget; reported notebook runs are research prototypes, not production models.

## License and attribution

When publishing or redistributing models trained on this data, follow the license and citation requirements of **[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)** and Amazon’s terms of use for review text and product images.
