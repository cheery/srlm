# Memory System

## Overview

The SRLM has an external memory mechanism derived from the MSA paper's
encode→compress→route→attend pipeline. Documents are encoded offline via the
prior layers, stored on CPU, and selectively loaded to GPU during training
or inference. The posterior layers' routers naturally score memory blocks
alongside pipeline blocks.

## Architecture

- `model.encode_document(tokens)` — runs tokens through input + prior layers
  at sigma=0 (clean text). Long documents are chunked at context_length,
  encoded separately, and averaged. Returns (B, L, D) on the same device.

- `MemoryBank` — holds encoded memories on CPU. Stores original tokens for
  re-encoding when the model improves.
  - `encode(model, docs, device)` — encode and store
  - `retrieve(query, k)` — cosine similarity routing, returns top-k on GPU
  - `refresh(model, device)` — re-encode all docs with current weights

- `model.forward(..., memories=None)` — optional list of memory tensors.
  Appended to the block list before the posterior loop. The per-layer
  routers score them alongside pipeline blocks via cosine similarity.

## Training: Anki-style flashcard replay

The `--memory-size N` flag enables a rolling memory bank of the last N
training batches. Each step:

1. Fresh batch arrives from training program, gets encoded into the bank
2. A random past batch is selected from the bank as the training target
3. Relevant memories are retrieved via cosine routing (the target's own
   encoded memory will score highest and always be included)
4. Model trains on the replayed batch with memories available

This teaches the model to use encoded memories for denoising — "the answer
is in your flashcards, find and use it."

## Key findings

### The model genuinely learns to use memories

With `--memory-size 50 --memory-k 2`, the memory-aided loss drops rapidly:

    step 160 | loss 381 (mixed, memory batches much lower)
    step 600 | loss 220
    step 660 | loss 175

For comparison, the best memoryless run reached ~530 at step 800. The model
learns to retrieve and exploit encoded memories for denoising.

### Memory dependency: the model forgets how to work without them

Fresh batches (no matching memory) show loss spikes to 780-1400 — WORSE
than a model that never trained with memories. The model reorganizes its
representations around "look up the answer" and loses the ability to
denoise from scratch.

The heron/loss.txt shows a bimodal pattern: individual steps alternate
between ~150 (replayed with memory) and ~800-1400 (fresh without).

### Scaffolding works: memory-trained weights transfer after readaptation

Training with memories for 800 steps, then continuing WITHOUT memories:

    Steps 1-90:   stuck at ~770 plateau (readapting)
    Steps 90-150: slow descent begins
    Steps 150-278: steady descent to ~695, still dropping

The memory-trained representations DID transfer — they just needed ~150
steps to readapt from "memory mode" to "standalone mode." The descent after
readaptation is smooth and steady, unlike the sawtoothy memory-on loss.

### The sawtooth pattern reveals partial memorization

After memory training, the memoryless loss.txt shows high variance between
individual steps (751-814 range). The model memorized specific Kalevala
segments through flashcard replay and has sharp "know this" vs "don't know
this" responses. This smooths out as standalone training continues.

### Memoryless training after scaffold reaches ~560

After the memory-pretrained model was trained without memories, it reached
560 — matching the original log1 baseline. The model produced recognizable
Finnish-like text with Kalevala cadence from just 500KB of training data.
As a diffusion model, generation is parallel and very fast.

## Alternating study/practice

The `--memory-alternate N` flag switches between phases every N steps:

- **Study phase**: replay flashcards with memories (textbook open).
  Clean gradient signal, model learns patterns with assistance.
- **Practice phase**: fresh batches, no memories (textbook closed).
  Model consolidates learned representations into standalone capability.

This mirrors spaced repetition with interleaved retrieval practice —
study, then test yourself, then study again. Each cycle:
- Study builds representations using the clean memory-aided gradient
- Practice forces internalization so the model doesn't become dependent

## CLI flags

    --memory-size N        Rolling bank capacity (0 = disabled)
    --memory-k K           Memories to retrieve per step (default: 2)
    --memory-refresh N     Re-encode bank every N steps (default: 100)
    --memory-alternate N   Switch study/practice every N steps (0 = always study)

## Multi-task interleaving

The `--interleave N` flag switches between training programs every N steps
(e.g. Kalevala and arithmetic). Without interleaving, sequential training
causes catastrophic forgetting — the second task overwrites the first
entirely.

### Catastrophic forgetting (sequential, heron4)

Training arithmetic for 200 steps then Kalevala: both tasks collapse.
Arithmetic outputs garbage (`1+2=8`), Kalevala just echoes the prompt.

### Interleaving preserves both tasks (heron5)

With `--interleave 50 --memory-size 50 --memory-k 2 --memory-alternate 10`:
- Kalevala loss settles to ~490-530 during Kalevala phases
- Arithmetic loss drops to 5-10 during arithmetic study phases
- Model generates plausible Finnish text AND retains arithmetic exposure

### Cross-task regularization

A surprising finding: arithmetic training appears to **improve** Finnish
text generation quality. The interleaved model produces better Finnish than
a Kalevala-only model trained for the same number of steps. Possible reasons:

- **Arithmetic forces precision** — exact token sequences with no room for
  fuzzy approximation. This sharpens the model's general denoising ability.
- **Multi-task regularization** — diverse data prevents overfitting to
  Kalevala's local optima. The model must learn more general representations.
- **Memory bank mixing** — during study phases, the bank contains both
  arithmetic and Kalevala batches. The model learns domain-agnostic memory
  retrieval.

This echoes multi-task training findings in NLP generally — diverse tasks
improve individual task performance even when the tasks are unrelated.

### Implications for scaling

If arithmetic acts as a regularizer that improves Finnish, adding more
diverse tasks should compound the effect. This motivates:

1. **More training** — 400 steps per task is tiny; longer runs should
   let the model fully internalize both tasks
2. **Reinforcement learning** — once form is learned, GRPO/GSPO can
   reward correctness (trivial verifier for arithmetic). The RL signal
   would indirectly sharpen all tasks through shared representations
3. **Task diversity** — add QA, code, structured data to push the model
   toward general-purpose representations

## Open questions

- What is the optimal alternate cycle length?
- Does this help more with structured data (Wikipedia/QA) where context
  memories are naturally different from the training target?
- Can memory-trained models reach lower final loss than pure training,
  or just get there faster?
- Should the memory bank persist across training runs (checkpoint it)?
- How many diverse tasks can a 10M param model juggle before capacity
  limits dominate?
- Does the cross-task improvement scale with model size?
