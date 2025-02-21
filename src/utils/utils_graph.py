import pandas as pd
import torch
from torch.nn.functional import kl_div
from functools import partial
from transformers import PreTrainedTokenizer
from typing import Optional, List, Union, Literal, Tuple
from transformer_lens import HookedTransformer
from loguru import logger

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA (NVIDIA GPU)
    print("Using CUDA device:", torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS (Apple Silicon GPU)
    print("Using MPS device")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("Using CPU device")


def get_logit_positions(logits: torch.Tensor, input_length: torch.Tensor):
    batch_size = logits.size(0)
    idx = torch.arange(batch_size, device=logits.device)
    logits = logits[idx, input_length - 1]
    return logits


def js_div(p: torch.tensor, q: torch.tensor):
    p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    m = (0.5 * (p + q)).log()
    return 0.5 * (
        kl_div(m, p.log(), log_target=True, reduction="none").mean(-1)
        + kl_div(m, q.log(), log_target=True, reduction="none").mean(-1)
    )


def divergence(
    logits: torch.Tensor,
    clean_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    divergence_type: Union[Literal["kl"], Literal["js"]] = "kl",
    mean=True,
    loss=True,
):
    logits = get_logit_positions(logits, input_length)
    clean_logits = get_logit_positions(clean_logits, input_length)

    probs = torch.softmax(logits, dim=-1)
    clean_probs = torch.softmax(clean_logits, dim=-1)

    if divergence_type == "kl":
        results = kl_div(
            probs.log(), clean_probs.log(), log_target=True, reduction="none"
        ).mean(-1)
    elif divergence_type == "js":
        results = js_div(probs, clean_probs)
    else:
        raise ValueError(
            f"Expected divergence_type of 'kl' or 'js', but got '{divergence_type}'"
        )
    return results.mean() if mean else results


def logit_diff(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: torch.Tensor,
    mean=True,
    prob=False,
    loss=False,
):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits
    good_bad = torch.gather(cleans, -1, labels.to(device))
    results = good_bad[:, 0] - good_bad[:, 1]

    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean:
        results = results.mean()
    return results


def get_metric(
    metric_name: str,
    task: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model: Optional[HookedTransformer] = None,
):
    if metric_name == "kl_divergence" or metric_name == "kl":
        return partial(divergence, divergence_type="kl")
    elif metric_name == "js_divergence" or metric_name == "js":
        return partial(divergence, divergence_type="js")
    elif metric_name == "logit_diff" or metric_name == "prob_diff":
        prob = metric_name == "prob_diff"
        if "toxicity" in task:
            logit_diff_fn = logit_diff_toxicity
        else:
            logit_diff_fn = logit_diff
        return partial(logit_diff_fn, prob=prob)
    else:
        raise ValueError(f"got bad metric_name: {metric_name}")


def logit_diff_toxicity(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    input_length: torch.Tensor,
    labels: List[torch.Tensor],
    mean=True,
    prob=False,
    loss=False,
):
    clean_logits = get_logit_positions(clean_logits, input_length)
    cleans = torch.softmax(clean_logits, dim=-1) if prob else clean_logits

    results = []
    for i, (ls, corrupted_ls) in enumerate(labels):
        r = (
            cleans[i][ls.to(cleans.device)].sum()
            - cleans[i][corrupted_ls.to(cleans.device)].sum()
        )
        results.append(r)
    results = torch.stack(results)

    if loss:
        # remember it's reversed to make it a loss
        results = -results
    if mean:
        results = results.mean()
    return results
