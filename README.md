# SRLM — Stateful Recurrent Language Model

A discrete diffusion language model with hierarchical recurrent memory (HRM).
Instead of autoregressive token-by-token generation, SRLM denoises entire
sequences in parallel using Score Entropy Discrete Diffusion (SEDD) with an
absorbing-state graph.

The recurrent memory lets the model accumulate context across training steps —
it reads a passage ("study"), then tries to reproduce patterns without looking
("practice"), similar to spaced repetition.

## Why

This project started from curiosity about Hierarchical Reasoning Models (HRM)
and a desire to learn deep learning hands-on by building something real.

The broader question: **can smaller AI models, trained for specific tasks,
run on modern consumer hardware?** Large language models require expensive
infrastructure. A focused model with the right architecture might be able
to do useful work at a fraction of the size.

## Goal

The long-term goal is a **programming assistant / terminal assistant** — a
small model that can help users operate their computer, answer questions
about code, and assist with everyday terminal tasks. Not a general-purpose
chatbot competing with frontier models, but a capable specialist that fits
on a laptop GPU.

## Architecture

```
Tokens → InputLayer (byte embedding + timestep + sinusoidal position)
       → Prior layers (DiT blocks with adaLN-Zero modulation)
       → HRM (hierarchical recurrent memory)
           ├─ FastLayer: cross-attention into pooled slow state, T iterations
           └─ SlowLayer: integrates fast states, updates every T fast steps
       → Posterior layers (DiT blocks + learned routers)
       → OutputLayer → log-score over vocabulary
```

**Diffusion process**: Absorbing-state discrete diffusion. Tokens are
progressively masked (absorbed) during the forward process and recovered
during reverse sampling. The model predicts a score function over the
vocabulary at each masked position.

**HRM**: Two-level recurrent state (fast zL, slow zH). The fast layer
runs T times per slow update, with cross-attention from the fast state
into chunk-mean-pooled slow state. All iterations except the final one
run under `torch.no_grad()` — gradients flow only through the last step.

**Routers**: Each posterior layer has a learned router that scores all
accumulated representations (prior outputs, HRM output, external memories)
via cosine similarity, producing a soft-weighted combination. An auxiliary
entropy loss prevents routing collapse.

**External memory**: Documents can be encoded offline via the prior layers
and injected into the posterior routing. The model learns to retrieve and
use relevant memories for denoising — then consolidates the knowledge
during memory-free practice phases.

## Model sizes

| Preset  | d_model | priors | posteriors | heads | params |
|---------|---------|--------|------------|-------|--------|
| small   | 256     | 3      | 2          | 8     | ~10M   |
| medium  | 384     | 4      | 3          | 12    | ~30M   |
| large   | 1152    | 3      | 2          | 16    | ~350M  |

## Training

The training loop supports multiple interleaved programs:

- **Kalevala** — Finnish epic poetry (567KB). Good for testing memorization.
- **Wikipedia** — Full English Wikipedia (~850MB). Tests generalization.
- **Arithmetic** — `N+M=Y` expressions. Verifiable correctness.
- **Sudoku** — 9x9 puzzles. Constraint satisfaction.
- **QA** — Question-answer pairs from custom datasets.

Programs rotate in round-robin. Each program alternates between **study**
(with memory bank) and **practice** (without), controlled by
`--memory-alternate N`.

### GRPO (Group Relative Policy Optimization)

For tasks with verifiable rewards (arithmetic, sudoku), GRPO generates K
candidate solutions via diffusion sampling, scores them with a reward
function, computes group-relative z-score advantages, and backpropagates
advantage-weighted SEDD loss. Uses a separate Adam optimizer to prevent
momentum corruption with the main SEDD optimizer.

### LoRA

Optional low-rank adaptation for GRPO — freezes base weights and trains
small adapter matrices on attention projections. Useful for protecting
learned representations during reinforcement learning.

## Usage

```bash
# Train a new model on Kalevala + arithmetic
python torch-version/main.py train my-model \
  --kalevala 500 --arithmetic 500 \
  --memory-size 100 --memory-k 2 --memory-alternate 25 \
  --batch-size 32 --seq-len 256

# Train medium model on A100
python torch-version/main.py train my-model \
  --medium --tf32 --batch-size 128 --seq-len 512 \
  --kalevala 500 --wikipedia 500 --arithmetic 500 --sudoku 500

# Resume training
python torch-version/main.py train my-model --kalevala 500

# Interactive generation
python torch-version/main.py eval my-model --steps 50

# With GRPO reinforcement
python torch-version/main.py train my-model \
  --arithmetic 500 --grpo-every 2 --grpo-epochs 1
```

## Key findings

**Cross-task regularization**: Arithmetic training improves Finnish text
generation. Multi-task training produces better results on individual tasks
than single-task training, likely because diverse tasks force more general
representations.

**Memory as scaffolding**: Training with external memories, then removing
them, transfers learned representations. The model needs ~150 readaptation
steps but then continues improving in standalone mode.

**Study/practice cycling**: Alternating between memory-aided study and
memory-free practice prevents the model from becoming dependent on external
memories while still benefiting from the clean gradient signal they provide.

## Implementations

- **torch-version/** — Primary implementation (PyTorch). Active development.
- **jax-version/** — Earlier JAX/Haiku implementation with S5 state space layers.
- **tf-version/** — TensorFlow implementation (broken)

## Pretrained models

Available on Hugging Face: https://huggingface.co/henrituhola/s5hrm-srlm

## Checkpoint format

Each checkpoint is a directory containing:
- `config.json` — model dimensions
- `parameters.pt` — model weights
- `training.txt` — training log (dataset counts, timestamps)
- `loss.txt` — per-step loss values
- `wiki.json` — Wikipedia progress (if applicable)

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for `torch.compile`, `scaled_dot_product_attention`)
- CUDA GPU recommended (A100 for medium/large models)
- `datasets` package (for Wikipedia loading)
- ../data directory with some datasets, (wikipedia cleaned, sudoku dataset, some custom Q/A sets, kalevala.txt cleaned of headers)
