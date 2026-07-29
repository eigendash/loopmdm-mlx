# loopmdm-mlx

A small independent MLX implementation of *Looped Diffusion Language Models* (arXiv:2605.26106), plus a toy masked diffusion language model trained on a synthetic arithmetic corpus to try the mechanism out. It runs on Apple silicon CPU/GPU through MLX.

The paper observes that masked diffusion models (MDMs) already spend compute across denoising steps, and asks whether looped transformers — a shared block applied repeatedly at fixed parameter cost — help on top of that. Its answer is **LoopMDM**: the denoising transformer is split into a small *head*, an early-middle *mid-block*, and a *tail*, and only the mid-block is applied `S` times with shared weights. The head and tail stay unshared, the parameter count is unchanged, and the effective depth becomes `head + S * mid + tail`. Looping is trained with a stochastic loop count `S ~ U{1..S_max}`, so the shared block sees a range of effective depths and the model can be run at any `S` at inference. The paper reports that LoopMDM matches same-size MDMs with up to 3.3x fewer training FLOPs, beats them on reasoning benchmarks (up to +8.5 points on GSM8K), surpasses deeper non-looped MDMs at comparable per-step compute, and that adaptively stopping the loop when the relative hidden-state change falls below a threshold `eps` recovers most of the quality at roughly half the average loop count. Its attention analysis finds that mask-to-mask attention grows with the loop count.

This is not a reproduction. It is a 312k-parameter model on a synthetic line of arithmetic, trained for a couple of minutes on one machine. Nothing here is comparable to the paper's 170M/125M models, its three pretraining corpora or GSM8K. The experiment below *does not* reproduce the paper's quality gains: with matched total compute the non-looped baseline learns the sum task better than LoopMDM does, and the adaptive schedule never fires. Both are reported plainly in the results section.

## What is in here

**The diffusion process** (`loopmdm/schedule.py`). A `Schedule` holds `alpha_t` (`linear`: `1 - t`, or `cosine`) with `alpha_0 = 1`, `alpha_1 = 0`; `corrupt_tokens` samples the absorbing-state forward process `q(x_t | x_0) = alpha_t * delta_{x_0} + (1 - alpha_t) * delta_mask` per position, and `reverse_transition_probs` implements the reverse step of the paper's Eq. 2: an already-unmasked token is kept, and a masked one stays masked with probability `alpha_{s|t} = (1 - alpha_s) / (1 - alpha_t)` or is drawn from the denoiser. `nelbo_weight` returns the continuous-time weight `-alpha'(t) / (1 - alpha_t)` (which is exactly `1/t` for the linear schedule, clipped at `t_min` because it diverges as `t -> 0`).

**Blocks** (`loopmdm/layers.py`). Pre-norm RMSNorm, bidirectional multi-head self-attention with RoPE on queries and keys only, and a gated-free MLP, with DiT-style adaLN modulation from a sinusoidal embedding of the diffusion time. There is deliberately **no per-loop conditioning** (no iteration-index embedding): the paper does not mention one, and every loop iteration of the mid-block therefore sees exactly the same modulation. Everything is written over an explicit parameter pytree rather than an `nn.Module`, which is what makes weight sharing literal array sharing.

**The looped model** (`loopmdm/model.py`). `forward(params, tokens, t, cfg, loop_count=R)` runs the head once, the mid-block `R` times over the same arrays, then the tail once; `forward_adaptive` replaces the fixed `R` with the paper's stopping rule, comparing `||H^(k) - H^(k-1)||_F / ||H^(k)||_F` against `eps` after each mid-block application and stopping early (at least one application, at most `max_loops`). `unroll_params(params, R)` returns the equivalent deeper model whose `R * n_mid` mid layers are the *same* arrays — the weight-tied deep model that the looped forward must equal. `layer_applications(cfg, R) = head + R * mid + tail` is the compute accounting used everywhere in the experiment, and `count_params` counts distinct arrays, so an unrolled tied tree has the same parameter count as the base model.

**Training** (`loopmdm/train.py`). `mdm_loss` is the masked-token cross-entropy at the corrupted positions with the optional NELBO weighting; `max_loops > 1` samples `S ~ U{1..S_max}` once per step (paper Eq. 5). `train_mdm` is AdamW with linear warmup + cosine decay, global-norm gradient clipping and per-step logging. It also takes `ignore` (positions never corrupted and never scored: padding), `always` (positions always corrupted: the answer span) and `realized_timestep` (hand the model the realised mask rate instead of the sampled one). Those three are how the toy task is made learnable — see the experiment and the differences section.

**Task** (`loopmdm/data.py`). A character-level corpus with `<pad>/<bos>/<eos>` plus digits, `+`, `=`, `?`, `P` and lower-case letters. Two arithmetic tasks and a tiny text slice: `sum` is `BOS a1+a2+...=D EOS` with `D = (a1 + ... + ak) mod 10` (one answer token), `parity` is `BOS a1+a2+...?PD EOS` with `D` the parity of the sum (two answer tokens), and `text` is one of ten short templated sentences. Both arithmetic answers depend on every addend, which is the "globally coupled" flavour the paper cares about; the answer span is recorded per example so evaluation can look only there.

**Sampling** (`loopmdm/sampler.py`). `confidence_sample` is parallel unmasking in the style the paper evaluates with: at each denoising step the still-masked positions are ranked by the model's maximum token probability and the most confident ones are committed, with the number of committed positions following the schedule's own marginal (`round(n_fill * (1 - alpha_t))`) so the sampler and the training-time corruption agree about how many positions should be masked at every `t`. `stochastic_sample` is the formal reverse process instead of the confidence heuristic. Both record the loop count used at every step and the total number of block evaluations. `masked_prompt` builds the corrupted input.

**Metrics** (`loopmdm/metrics.py`). `answer_nll` and `span_exact_match` look only at the answer span; `mask_to_mask_attention` computes the paper's measurement (Appendix C.4), the average attention mass from masked-position queries to masked-position keys, averaged over heads and sequences, and `mid_attention_by_iteration` regroups a `capture=True` forward pass into `[loop iteration][mid layer]`.

## Checks

`pytest` runs 105 tests. The ones that matter most, and the invariants the brief asked for:

- **`R = 1` is the non-looped model, bit for bit.** `forward(params, x, t, cfg, loop_count=1)` is compared with `mx.array_equal` against the same model built as an explicit non-looped stack (`test_model.py`), so the clean limit is exact, not approximate.
- **Looping adds zero parameters.** `count_params(unroll_params(params, R)) == count_params(params) == num_parameters(cfg)` for `R` in 1, 2, 3, 8, and the unrolled tree really does contain the *same* array objects (`params["mid"][0] is unroll_params(params, 3)["mid"][2]`).
- **The looped model equals the tied-weight deeper model.** For `R` in 1..4 the looped forward matches a model whose depth is `head + R * mid + tail` and whose mid layers share weights, to `1e-6` (in practice they are identical computations).
- **Compute scales linearly with `R`.** The number of attention matrices captured in a forward pass equals `layer_applications(cfg, R) = head + R * mid + tail`, and `layer_applications(cfg, R) - layer_applications(cfg, 1) == (R - 1) * mid`.
- **Gradients reach the shared block from every iteration.** The test gives each loop iteration its own *copy* of the mid-block parameters, checks that every copy receives a non-zero, finite gradient, and then checks that the copies' gradients sum to the tied model's gradient (MLX accumulates the cotangents of a shared array over all of its uses).
- **Training and sampling agree about corruption.** With an oracle denoiser that predicts the clean token exactly, one reverse step reproduces the forward marginal `q(x_s | x_0)` to within sampling error, for both schedules, and a four-step reverse chain hits every intermediate marginal and ends fully unmasked (`test_schedule.py`). Separately, the confidence sampler's remaining-mask counts equal `round(n_fill * (1 - alpha_t))` at every step for both schedules (`test_sampler.py`).
- **The NELBO weight is the analytic one.** `w(t) * (1 - alpha_t) == -alpha'(t)` for both schedules, `alpha'(t)` matches finite differences, and `w(t) = 1/t` for the linear schedule.
- **Hand-computed values where they exist.** Attention is compared against `mx.fast.scaled_dot_product_attention` at `1e-5`; RoPE dot products depend only on relative position and preserve norms; the mask-to-mask measure is checked on a 3x3 attention matrix by hand, is 1.0 when every position is masked and 0.0 with no masked query; the sum and parity corpora are checked against Python's own arithmetic.
- **The objective actually trains.** On a learnable synthetic corpus the loss falls by more than a factor of two in 200 steps, the looped variant (`S_max = 3`) learns too, and two runs with the same seed are bit-identical on the CPU device (MLX's GPU reductions are not deterministic; see below).

## Experiment

`scripts/run_experiment.py`. The corpus is generated with a fixed seed: 4096 training examples and 512 held-out examples (sum and parity, 2-3 addends, `seq_len = 12`), plus the text slice in training only. One seed, 2500 training steps for the baseline, looped and deep runs, and 3750 for `mdm-baseline-matched` so that its *total* block evaluations match the looped model's:

| run | layers | params | steps | block evals/step | total block evals | loss first20 | loss last20 | wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mdm-baseline` | 6 unique | 312448 | 2500 | 6.00 | 15000 | 2.875 | 1.343 | 79 s |
| `mdm-baseline-matched` | 6 unique | 312448 | 3750 | 6.00 | 22500 | 2.875 | 1.144 | 85 s |
| `loopmdm` (mid shared, `S_max = 4`) | 6 (2 shared) | 312448 | 2500 | 8.94 | 22338 | 3.385 | 1.353 | 70 s |
| `deep-baseline` | 9 unique | 461632 | 2500 | 9.00 | 22500 | 2.980 | 1.347 | 123 s |

The baseline and LoopMDM have identical parameter counts and the looped model's total block evaluations (22338) match the matched baseline's (22500) and the deep baseline's; the deep baseline matches LoopMDM's per-step compute with three extra layers instead. Loss is the cross-entropy on the answer span (padding is never corrupted), averaged over the first/last 20 steps.

Inference at the answer span (prompt visible, answer masked, confidence unmasking over the answer tokens):

| run | R | sum acc | parity acc | all acc | answer NLL | block evals/example |
|---|---:|---:|---:|---:|---:|---:|
| `mdm-baseline` | 1 | 0.077 | 0.508 | 0.289 | 1.0151 | 12 |
| `mdm-baseline-matched` | 1 | **0.235** | 0.492 | 0.361 | **0.9291** | 12 |
| `loopmdm` | 1 | 0.081 | 0.524 | 0.299 | 1.0270 | 12 |
| `loopmdm` | 2 | 0.081 | 0.480 | 0.277 | 1.0187 | 16 |
| `loopmdm` | 4 | 0.081 | 0.512 | 0.293 | 1.0154 | 24 |
| `deep-baseline` | 1 | 0.081 | 0.552 | 0.312 | 1.0145 | 18 |

Adaptive looping on the same model (`S_max = 4`):

| eps | sum acc | parity acc | all acc | mean R | block evals/example | mean R per denoising step |
|---:|---:|---:|---:|---:|---:|---|
| 0.02 | 0.081 | 0.512 | 0.293 | 4.00 | 24 | 4.00 4.00 |
| 0.05 | 0.081 | 0.512 | 0.293 | 4.00 | 24 | 4.00 4.00 |
| 0.10 | 0.081 | 0.512 | 0.293 | 4.00 | 24 | 4.00 4.00 |
| 0.20 | 0.081 | 0.512 | 0.293 | 4.00 | 24 | 4.00 4.00 |

Mask-to-mask attention at `t = 0.5` (512 held-out sequences with a 50% random mask, averaged over heads, sequences and the two mid layers), by loop iteration:

| run | R | iteration 1 | iteration 2 | iteration 3 | iteration 4 | mean |
|---|---:|---:|---:|---:|---:|---:|
| `mdm-baseline` | 1 | 0.5883 | - | - | - | 0.5883 |
| `loopmdm` | 1 | 0.5507 | - | - | - | 0.5507 |
| `loopmdm` | 2 | 0.5507 | 0.6123 | - | - | 0.5815 |
| `loopmdm` | 4 | 0.5507 | 0.6123 | 0.6348 | 0.6557 | 0.6134 |

Each cell averages the two mid layers; iteration 1 is literally the same computation whatever `R` is, which is why the first column is identical across rows. The per-layer split (0.5322 and 0.5691 at iteration 1, 0.6448 and 0.6666 at iteration 4) is in `results/experiment.json`.

What these numbers mean, and what they do not:

- **The paper's central claim does not hold at this scale.** With matched total compute (22338 vs 22500 block evaluations), the *non-looped* baseline reaches 0.235 exact match on the sum task against a 10% chance rate and an answer NLL of 0.9291, while LoopMDM stays at 0.081 and 1.0270 - worse than a predictor that knows nothing but the format. The looped model is also not cheap: at equal step counts it does 8.94 block evaluations per step against the baseline's 6.00 for the same training loss (1.353 vs 1.343), and its loss starts much higher (3.385 vs 2.875 over the first 20 steps) because the shared block has to serve several depths at once. One seed is a weakness here, but the gap (0.235 vs 0.081) is far outside anything the seed noise could explain at these accuracies.
- **The deep baseline at matched per-step compute is also at chance** (0.081 sum accuracy, NLL 1.0145, 9 layers, 461632 parameters, 18 block evaluations per answer). So the extra depth does not buy the task either, which is at least consistent with the paper's "depth is not the whole story" framing - though here looping does not rescue it.
- **Answer NLL identifies what was learned.** The trivial predictor (constant `P` for the parity format, uniform over the digit/bit) gives `(260 * ln 10 + 504 * ln 2) / 764 = 1.0122` on this held-out set. `mdm-baseline` measures 1.0151 and `loopmdm` 1.0270/1.0154, i.e. the format and nothing else; only the matched-compute baseline gets below it (0.9291).
- **Adaptive looping never fires** for `eps <= 0.2` on this model, so the average loop count stays at 4.00 and 24 block evaluations per answer, with no accuracy or NLL change. The stopping rule as implemented (Frobenius norm of the whole hidden state) sees the hidden state keep moving by more than 20% per iteration. This is a null result for the adaptive schedule at this scale, not a reproduction of the paper's "12 -> ~5 loops at eps = 0.1".
- **Mask-to-mask attention does grow with the loop count**, 0.5507 (one application) -> 0.5815 (two) -> 0.6134 (four), and within `R = 4` it grows monotonically per iteration (0.5507, 0.6123, 0.6348, 0.6557). This is the one qualitative pattern from the paper that the toy reproduces, and it is measured on a model that has not learned the task, so it is not evidence that the extra attention is useful.
- Only block evaluations are counted as compute. There are no FLOPs, no wall-clock speedups and no kernels; the quantity that matters here is exactly `head + R * mid + tail` per forward pass.

## Where this may differ from the paper

- **The training scheme is adapted so the toy task is learnable at all (and even then only the longest run learns it).** The paper trains the plain NELBO objective, corrupting every position with probability `1 - alpha_t`. With that objective the 312k-parameter model here does not learn the task in any budget I could afford (I tried 0.5k-2.5k steps, `t ~ U[0,1]` and `t ~ U[0,0.3]`, both weightings, and a 64-example overfit), so the experiment instead corrupts the **answer span at every step** (`always` in `train_mdm`), never corrupts padding, never scores padding (`ignore`), and hands the model the **realised** mask rate rather than the sampled one (`realized_timestep`). Extra random corruption elsewhere is still available through `--t-max` (the recorded run uses 0.0). The architecture and the loop are untouched by this; the objective is not the paper's.
- **`sum` is modulo 10.** The answer is a single token `(a1 + ... + ak) mod 10` rather than the full sum with carries. This keeps the answer span small enough for the sampler to have something to do and was the version that showed any learning signal at all. The parity task is the paper-flavoured "every addend matters" task.
- **The adaptive stopping rule's norm is my reading.** The paper writes `||H^(k) - H^(k-1)|| / ||H^(k)||` without specifying the norm; `relative_hidden_change` uses the Frobenius norm over the whole hidden state (all positions and channels, not per token). With this norm none of the four thresholds in the recorded sweep ever fires, so the adaptive numbers in the table are identical to a fixed `R = 4`.
- **No per-loop conditioning.** Head and tail are unshared, as the paper says; nothing else is added per iteration. If the paper's implementation injects anything per loop iteration (an iteration embedding, for instance), this does not reproduce it.
- **Loop placement is fixed at layers 1-2** (one head layer, two mid layers, three tail layers), the paper's best configuration, rather than searched. The ablation over position, number of looped layers and `S_max` (their Table 4) is not reproduced.
- **The sampler is answer-region only.** Generation starts from the answer span masked with the prompt visible, which is why the number of denoising steps is 2; the paper's generative evaluation samples whole sequences with hundreds of steps. Full-sequence sampling is not evaluated here.
- **The recorded run is one seed of a 2500-step run**; the paper averages five seeds of 1M-step runs. The accuracy gap between the matched baseline and LoopMDM is large (0.235 vs 0.081) but the *ordering* of the smaller differences (baseline vs deep baseline, `R = 1` vs `R = 4`) should not be trusted from a single seed.
- **Mask-to-mask attention is measured at `t = 0.5`** on 512 held-out sequences with a 50% random mask, whereas the paper samples 10,000 timesteps and reports at `t = 0.5` at the two mid-block layers of a 12-layer OpenWebText model.
- **Loops are not the only difference from the paper's recipe.** There is no RoPE+adaLN DiT at 170M, no sequence packing, no EMA, no zero-shot perplexity suite of seven corpora, no downstream benchmarks, and no GSM8K. The paper's 3.3x FLOP saving and +8.5 GSM8K are not measured or claimed here.
- **Runs are seeded but only bit-reproducible on the CPU.** Every draw (batch, timestep, mask, loop count) comes from an explicit key, so the data stream repeats exactly; the weights do not, because several MLX GPU reductions are non-deterministic. Re-running the same seed on the GPU moves the final loss by ~1e-5 after 25 steps and can move it more over 1000; `tests/test_train.py` checks bit-exactness with the device pinned to CPU. The committed `results/` came from GPU runs.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/run_experiment.py
```

The experiment takes about six minutes for the recorded configuration (one seed, 2500 steps) and rewrites `results/experiment.json` and `results/run_log.txt`. Useful flags: `--seeds`, `--steps`, `--max-loops`, `--inference-loops`, `--adaptive-eps`, `--t-max`, `--n-layers`, `--mid-layers`, `--seq-len`, `--quick`.
