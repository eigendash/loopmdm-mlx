"""Toy experiment for Looped Masked Diffusion Models (arXiv:2605.26106).

Four models are trained on the same synthetic corpus with the same optimiser:

    mdm-baseline           6 unique layers, trained for ``steps``
    mdm-baseline-matched   the same model trained for the same total number of
                           layer applications as the looped model
    loopmdm                6 layers with the 2 mid layers shared, trained with
                           S ~ U{1..S_max}
    deep-baseline          9 unique layers, matched to the looped model's
                           per-step compute (but not its parameter count)

They are then compared on a held-out set at inference time: exact-match
accuracy of the answer tokens, the number of transformer block evaluations each
configuration spends, a sweep over the inference loop count R, the paper's
adaptive stopping rule, and mask-to-mask attention at each loop iteration.

Training corrupts the answer span at every step and, optionally, a random
subset of the remaining positions (``--t-max``); corrupted positions outside the
answer span are part of the loss too, so the model is still a denoiser.  The
timestep handed to the model is the *realised* mask rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopmdm.config import ModelConfig, TrainConfig  # noqa: E402
from loopmdm.data import (  # noqa: E402
    MASK_ID,
    PAD_ID,
    VOCAB_SIZE,
    generate_examples,
    task_counts,
    to_array,
)
from loopmdm.metrics import (  # noqa: E402
    answer_nll,
    loop_profile_by_timestep,
    mask_to_mask_attention,
    mid_attention_by_iteration,
    span_exact_match,
    span_token_accuracy,
    total_layer_applications,
)
from loopmdm.model import forward, forward_adaptive, layer_applications, num_parameters  # noqa: E402
from loopmdm.sampler import confidence_sample, masked_prompt  # noqa: E402
from loopmdm.train import train_mdm  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


class Logger:
    """Print and keep every line so the log file matches what was printed."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self.lines.append(message)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def build_corpora(seq_len: int, n_train: int, n_val: int, max_addends: int) -> dict:
    train = generate_examples(n_train, seq_len, seed=0, max_addends=max_addends)
    val = generate_examples(
        n_val, seq_len, seed=1234, mix={"sum": 0.5, "parity": 0.5}, max_addends=max_addends
    )
    return {
        "train_tokens": to_array(train),
        "train_span": mx.array(np.stack([e.answer_mask() for e in train])),
        "val": val,
        "val_tokens": to_array(val),
        "val_span": mx.array(np.stack([e.answer_mask() for e in val])),
        "tasks": np.array([e.task for e in val]),
    }


def _em(pred, tokens, span, rows) -> float:
    return span_exact_match(
        mx.take(pred, rows, axis=0), mx.take(tokens, rows, axis=0), mx.take(span, rows, axis=0)
    )


def evaluate_inference(
    params: dict,
    cfg: ModelConfig,
    data: dict,
    *,
    loop_count: int = 1,
    max_loops: int | None = None,
    eps: float = 0.1,
    steps: int = 2,
) -> dict:
    """Answer accuracy, NLL and compute for one inference configuration."""
    tokens, span, tasks = data["val_tokens"], data["val_span"], data["tasks"]
    masked = masked_prompt(tokens, span, cfg.mask_id)
    kwargs = (
        {"max_loops": max_loops, "eps": eps}
        if max_loops is not None
        else {"loop_count": loop_count}
    )
    t0 = time.time()
    trace = confidence_sample(params, masked, cfg, span, steps=steps, **kwargs)
    wall = time.time() - t0

    sum_rows = mx.array(np.flatnonzero(tasks == "sum"))
    par_rows = mx.array(np.flatnonzero(tasks == "parity"))
    out = {
        "loop_count": loop_count if max_loops is None else None,
        "max_loops": max_loops,
        "eps": eps if max_loops is not None else None,
        "denoising_steps": steps,
        "exact_match_sum": _em(trace.tokens, tokens, span, sum_rows),
        "exact_match_parity": _em(trace.tokens, tokens, span, par_rows),
        "exact_match_all": span_exact_match(trace.tokens, tokens, span),
        "token_accuracy_all": span_token_accuracy(trace.tokens, tokens, span),
        "mean_loop_count": trace.mean_loop_count,
        "layer_applications_per_example": trace.layer_applications,
        "loop_profile_by_timestep": loop_profile_by_timestep(trace.loop_counts, steps),
        "sampling_wall_s": wall,
    }
    # teacher-forced answer NLL: one forward pass with the answer span masked and
    # the timestep set to the mask rate the answer span implies
    t_row = mx.sum(span.astype(mx.float32), axis=1) / mx.sum(
        (tokens != PAD_ID).astype(mx.float32), axis=1
    )
    if max_loops is None:
        logits = forward(params, masked, t_row, cfg, loop_count=loop_count).logits
    else:
        logits = forward_adaptive(params, masked, t_row, cfg, max_loops=max_loops, eps=eps).logits
    pred = mx.argmax(logits, axis=-1)
    out["answer_nll"] = answer_nll(logits, tokens, span)
    out["teacher_forced_exact_match_sum"] = _em(pred, tokens, span, sum_rows)
    out["teacher_forced_exact_match_parity"] = _em(pred, tokens, span, par_rows)
    return out


def mask_to_mask_analysis(
    params: dict, cfg: ModelConfig, data: dict, loop_count: int, seed: int
) -> dict:
    """Average mask-to-mask attention per mid-layer and loop iteration at t = 0.5."""
    tokens = data["val_tokens"]
    mask = mx.random.uniform(shape=tokens.shape, key=mx.random.key(seed)) < 0.5
    x_t = masked_prompt(tokens, mask, cfg.mask_id)
    out = forward(params, x_t, 0.5, cfg, loop_count=loop_count, capture=True)
    grouped = mid_attention_by_iteration(out.attentions, cfg, loop_count)
    per_iteration = [mask_to_mask_attention(layer_atts, mask) for layer_atts in grouped]
    return {
        "loop_count": loop_count,
        "mask_rate": float(mx.mean(mask.astype(mx.float32))),
        "per_iteration": per_iteration,
        "mean_per_iteration": [float(np.mean(v)) for v in per_iteration],
    }


def run_experiment(args) -> dict:
    log = Logger()
    log("LoopMDM toy experiment - Looped Diffusion Language Models (arXiv:2605.26106)")
    log("=" * 78)
    started = time.time()

    cfg = ModelConfig(
        vocab_size=VOCAB_SIZE,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_layers=args.head_layers,
        mid_layers=args.mid_layers,
        mlp_ratio=args.mlp_ratio,
        time_dim=args.d_model,
    )
    data = build_corpora(cfg.seq_len, args.n_train, args.n_val, args.max_addends)
    log(f"vocab {VOCAB_SIZE} (mask id {MASK_ID}), seq_len {cfg.seq_len}, "
        f"addends 2..{args.max_addends}")
    log(f"train corpus {tuple(data['train_tokens'].shape)}, held-out "
        f"{tuple(data['val_tokens'].shape)} {dict(task_counts(data['val']))}")
    log(f"model: {cfg.n_layers} layers = head {cfg.head_layers} + mid {cfg.mid_layers} "
        f"(shared when looped) + tail {cfg.tail_layers}, d_model {cfg.d_model}, "
        f"{cfg.n_heads} heads")
    log(f"training: {args.steps} steps for the baseline/looped/deep runs, lr {args.lr}, "
        f"batch {args.batch_size}, grad clip {args.grad_clip}, extra random masking "
        f"t ~ U[0, {args.t_max}], padding never corrupted")
    log("")

    expected_s = (args.max_loops + 1) / 2
    loop_per_step = cfg.head_layers + expected_s * cfg.mid_layers + cfg.tail_layers
    matched_steps = int(round(args.steps * loop_per_step / cfg.n_layers))

    runs = [
        {"name": "mdm-baseline", "n_layers": cfg.n_layers, "mid_shared": False,
         "max_loops": 1, "steps": args.steps},
        {"name": "mdm-baseline-matched", "n_layers": cfg.n_layers, "mid_shared": False,
         "max_loops": 1, "steps": matched_steps},
        {"name": "loopmdm", "n_layers": cfg.n_layers, "mid_shared": True,
         "max_loops": args.max_loops, "steps": args.steps},
        {"name": "deep-baseline", "n_layers": args.n_layers + args.extra_depth,
         "mid_shared": False, "max_loops": 1, "steps": args.steps},
    ]
    log("training runs (the first three are iso-parameter; the deep baseline matches "
        "the looped model's per-step compute instead)")
    log(f"{'run':22s} {'layers':>6s} {'params':>9s} {'steps':>6s} {'apps/step':>10s} "
        f"{'total apps':>11s} {'loss first20':>13s} {'loss last20':>12s} {'wall s':>7s}")
    log("-" * 78)

    trained: dict[tuple[str, int], dict] = {}
    training_records = []
    for seed in args.seeds:
        for run in runs:
            run_cfg = ModelConfig(
                vocab_size=VOCAB_SIZE, seq_len=cfg.seq_len, d_model=cfg.d_model,
                n_heads=cfg.n_heads, n_layers=run["n_layers"], head_layers=cfg.head_layers,
                mid_layers=cfg.mid_layers, mlp_ratio=cfg.mlp_ratio, time_dim=cfg.time_dim,
            )
            train_cfg = TrainConfig(
                steps=run["steps"], batch_size=args.batch_size, lr=args.lr,
                warmup=args.warmup, seed=seed, max_loops=run["max_loops"],
                grad_clip=args.grad_clip,
            )
            t0 = time.time()
            result = train_mdm(
                run_cfg, train_cfg, data["train_tokens"], verbose=False,
                ignore=mx.equal(data["train_tokens"], PAD_ID),
                always=data["train_span"],
                realized_timestep=True,
                t_max=args.t_max,
            )
            wall = time.time() - t0
            params = result.params
            loops = [h["loop_count"] for h in result.history]
            apps_per_step = float(np.mean([layer_applications(run_cfg, r) for r in loops]))
            total_apps = total_layer_applications(run_cfg, loops)
            record = {
                "run": run["name"], "seed": seed, "config": run_cfg.as_dict(),
                "params": num_parameters(run_cfg), "steps": run["steps"],
                "max_loops": run["max_loops"],
                "layer_applications_per_step": apps_per_step,
                "total_layer_applications": total_apps,
                "loss_first20": float(np.mean([h["loss"] for h in result.history[:20]])),
                "loss_last20": float(np.mean([h["loss"] for h in result.history[-20:]])),
                "observed_mean_loop_count": float(np.mean(loops)),
                "wall_s": wall,
            }
            training_records.append(record)
            trained[(run["name"], seed)] = {"params": params, "cfg": run_cfg, "record": record}
            log(f"{run['name']:22s} {run['n_layers']:6d} {record['params']:9d} "
                f"{run['steps']:6d} {apps_per_step:10.2f} {total_apps:11d} "
                f"{record['loss_first20']:13.4f} {record['loss_last20']:12.4f} {wall:7.1f}")

    log("")
    log("inference at the answer span (prompt visible, answer masked, confidence "
        "unmasking over the answer tokens)")
    log(f"{'run':22s} {'seed':>4s} {'R':>3s} {'sum acc':>8s} {'parity acc':>10s} "
        f"{'all acc':>8s} {'ans NLL':>8s} {'apps/ex':>8s}")
    log("-" * 78)
    inference_records = []
    sweep = args.inference_loops
    for seed in args.seeds:
        for run in runs:
            entry = trained[(run["name"], seed)]
            for R in (sweep if run["mid_shared"] else [1]):
                out = evaluate_inference(
                    entry["params"], entry["cfg"], data, loop_count=R, steps=args.denoise_steps
                )
                out.update({"run": run["name"], "seed": seed})
                inference_records.append(out)
                log(f"{run['name']:22s} {seed:4d} {R:3d} {out['exact_match_sum']:8.3f} "
                    f"{out['exact_match_parity']:10.3f} {out['exact_match_all']:8.3f} "
                    f"{out['answer_nll']:8.4f} {out['layer_applications_per_example']:8d}")

    log("")
    log("adaptive looping (stop when ||H^(k)-H^(k-1)||/||H^(k)|| < eps)")
    log(f"{'seed':>4s} {'eps':>5s} {'sum acc':>8s} {'parity acc':>10s} {'all acc':>8s} "
        f"{'mean R':>7s} {'apps/ex':>8s}  {'loops by step':>16s}")
    log("-" * 78)
    adaptive_records = []
    for seed in args.seeds:
        entry = trained[("loopmdm", seed)]
        for eps in args.adaptive_eps:
            out = evaluate_inference(
                entry["params"], entry["cfg"], data, max_loops=args.max_loops, eps=eps,
                steps=args.denoise_steps,
            )
            out.update({"run": "loopmdm", "seed": seed})
            adaptive_records.append(out)
            profile = " ".join(f"{v:.2f}" for v in out["loop_profile_by_timestep"])
            log(f"{seed:4d} {eps:5.3f} {out['exact_match_sum']:8.3f} "
                f"{out['exact_match_parity']:10.3f} {out['exact_match_all']:8.3f} "
                f"{out['mean_loop_count']:7.2f} {out['layer_applications_per_example']:8d}  "
                f"{profile:>16s}")

    log("")
    log("mask-to-mask attention at t=0.5 (mass from masked queries to masked keys)")
    log(f"{'run':22s} {'seed':>4s} {'R':>3s} {'per mid layer, per loop iteration':>34s} "
        f"{'mean':>7s}")
    log("-" * 78)
    m2m_records = []
    for seed in args.seeds:
        base = trained[("mdm-baseline", seed)]
        rec = mask_to_mask_analysis(base["params"], base["cfg"], data, 1, seed=7)
        rec.update({"run": "mdm-baseline", "seed": seed})
        m2m_records.append(rec)
        log(f"{'mdm-baseline':22s} {seed:4d} {1:3d} "
            f"{' '.join(f'{v:.4f}' for v in rec['per_iteration'][0]):>34s} "
            f"{rec['mean_per_iteration'][0]:7.4f}")
        entry = trained[("loopmdm", seed)]
        for R in sweep:
            rec = mask_to_mask_analysis(entry["params"], entry["cfg"], data, R, seed=7)
            rec.update({"run": "loopmdm", "seed": seed})
            m2m_records.append(rec)
            flat = " ".join(f"{v:.4f}" for v in np.array(rec["per_iteration"]).reshape(-1))
            log(f"{'loopmdm':22s} {seed:4d} {R:3d} {flat:>34s} "
                f"{np.mean(rec['mean_per_iteration']):7.4f}")

    summary = summarise(training_records, inference_records, adaptive_records, m2m_records)
    log("")
    log("summary (mean over seeds)")
    log(f"{'run':22s} {'params':>9s} {'R':>3s} {'sum acc':>8s} {'parity acc':>10s} "
        f"{'all acc':>8s} {'ans NLL':>8s} {'apps/ex':>8s}")
    log("-" * 78)
    for row in summary["inference"]:
        log(f"{row['run']:22s} {row['params']:9d} {row['loop_count']:3d} "
            f"{row['exact_match_sum']:8.3f} {row['exact_match_parity']:10.3f} "
            f"{row['exact_match_all']:8.3f} {row['answer_nll']:8.4f} "
            f"{row['layer_applications_per_example']:8.1f}")
    log("")
    log("adaptive summary (mean over seeds)")
    for row in summary["adaptive"]:
        log(f"  eps {row['eps']:.3f}: sum {row['exact_match_sum']:.3f} "
            f"parity {row['exact_match_parity']:.3f} all {row['exact_match_all']:.3f} "
            f"mean R {row['mean_loop_count']:.2f} apps {row['layer_applications_per_example']:.1f}")
    log("")
    log(f"total wall clock {time.time() - started:.1f}s")

    payload = {
        "meta": {
            "paper": "Looped Diffusion Language Models (arXiv:2605.26106)",
            "seeds": args.seeds,
            "denoise_steps": args.denoise_steps,
            "inference_loops": sweep,
            "adaptive_eps": args.adaptive_eps,
            "max_loops": args.max_loops,
            "t_max": args.t_max,
            "batch_size": args.batch_size,
            "train_steps": args.steps,
            "matched_baseline_steps": matched_steps,
            "grad_clip": args.grad_clip,
            "lr": args.lr,
            "model": cfg.as_dict(),
            "n_train": args.n_train,
            "n_val": args.n_val,
            "wall_clock_s": time.time() - started,
        },
        "training": training_records,
        "inference": inference_records,
        "adaptive": adaptive_records,
        "mask_to_mask": m2m_records,
        "summary": summary,
    }
    return {"payload": payload, "log": log}


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def summarise(training, inference, adaptive, m2m) -> dict:
    runs = []
    for name in dict.fromkeys(r["run"] for r in training):
        subset = [r for r in training if r["run"] == name]
        runs.append({
            "run": name,
            "params": subset[0]["params"],
            "steps": subset[0]["steps"],
            "layer_applications_per_step": _mean([r["layer_applications_per_step"] for r in subset]),
            "total_layer_applications": _mean([r["total_layer_applications"] for r in subset]),
            "loss_first20": _mean([r["loss_first20"] for r in subset]),
            "loss_last20": _mean([r["loss_last20"] for r in subset]),
            "wall_s": _mean([r["wall_s"] for r in subset]),
        })

    inference_summary = []
    for name, R in dict.fromkeys((r["run"], r["loop_count"]) for r in inference):
        subset = [r for r in inference if r["run"] == name and r["loop_count"] == R]
        inference_summary.append({
            "run": name,
            "loop_count": R,
            "params": next(r["params"] for r in runs if r["run"] == name),
            "exact_match_sum": _mean([r["exact_match_sum"] for r in subset]),
            "exact_match_parity": _mean([r["exact_match_parity"] for r in subset]),
            "exact_match_all": _mean([r["exact_match_all"] for r in subset]),
            "token_accuracy_all": _mean([r["token_accuracy_all"] for r in subset]),
            "answer_nll": _mean([r["answer_nll"] for r in subset]),
            "teacher_forced_exact_match_sum": _mean(
                [r["teacher_forced_exact_match_sum"] for r in subset]),
            "teacher_forced_exact_match_parity": _mean(
                [r["teacher_forced_exact_match_parity"] for r in subset]),
            "layer_applications_per_example": _mean(
                [r["layer_applications_per_example"] for r in subset]),
        })

    adaptive_summary = []
    for eps in dict.fromkeys(r["eps"] for r in adaptive):
        subset = [r for r in adaptive if r["eps"] == eps]
        adaptive_summary.append({
            "eps": eps,
            "exact_match_sum": _mean([r["exact_match_sum"] for r in subset]),
            "exact_match_parity": _mean([r["exact_match_parity"] for r in subset]),
            "exact_match_all": _mean([r["exact_match_all"] for r in subset]),
            "answer_nll": _mean([r["answer_nll"] for r in subset]),
            "mean_loop_count": _mean([r["mean_loop_count"] for r in subset]),
            "layer_applications_per_example": _mean(
                [r["layer_applications_per_example"] for r in subset]),
            "loop_profile_by_timestep": np.mean(
                [r["loop_profile_by_timestep"] for r in subset], axis=0).tolist(),
        })

    m2m_summary = []
    for name in dict.fromkeys(r["run"] for r in m2m):
        subset = [r for r in m2m if r["run"] == name]
        for R in dict.fromkeys(r["loop_count"] for r in subset):
            rows = [r for r in subset if r["loop_count"] == R]
            per_iter = np.mean(
                [np.mean(r["per_iteration"], axis=0) for r in rows], axis=0).tolist()
            m2m_summary.append({
                "run": name,
                "loop_count": R,
                "mean_per_layer": per_iter,
                "mean": float(np.mean(per_iter)),
            })
    return {
        "training": runs,
        "inference": inference_summary,
        "adaptive": adaptive_summary,
        "mask_to_mask": m2m_summary,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=1000, help="training steps per run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--t-max", type=float, default=0.0,
                        help="extra random masking rate; 0 keeps only the answer span corrupted")
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--max-addends", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--head-layers", type=int, default=1)
    parser.add_argument("--mid-layers", type=int, default=2)
    parser.add_argument("--extra-depth", type=int, default=3,
                        help="extra layers for the deep baseline")
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--max-loops", type=int, default=4, help="S_max during training")
    parser.add_argument("--inference-loops", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--denoise-steps", type=int, default=2)
    parser.add_argument("--adaptive-eps", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=512)
    parser.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.seeds = [0]
        args.steps = 30
        args.n_train = 256
        args.n_val = 64
        args.adaptive_eps = [0.1]
    out = run_experiment(args)
    RESULTS.mkdir(exist_ok=True)
    out["log"].save(RESULTS / "run_log.txt")
    (RESULTS / "experiment.json").write_text(json.dumps(out["payload"], indent=2) + "\n")
    print(f"\nwrote {RESULTS/'experiment.json'} and {RESULTS/'run_log.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
