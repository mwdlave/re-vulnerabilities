from functools import partial
from typing import Callable, List, Union, Optional
from functools import partial
from pathlib import Path
from tqdm import tqdm
from loguru import logger

import torch
from torch.utils.data import DataLoader
from torch import Tensor
from einops import einsum
from transformer_lens import HookedTransformer

from .graph import Graph, AttentionNode, LogitNode, InputNode
from .config_utils import load_config
from .config import Config

allowed_aggregations = {"sum", "mean", "l2"}


def get_npos_input_lengths(model, inputs):
    tokenized = model.tokenizer(
        inputs, padding="longest", return_tensors="pt", add_special_tokens=True
    )
    logger.debug("DEBUG get_npos_input_lengths:")
    logger.debug(f"  tokenized['input_ids'].shape = {tokenized['input_ids'].shape}")
    logger.debug(f"  attention_mask.shape          = {tokenized.attention_mask.shape}")
    logger.debug(f"  attention_mask sums           = {tokenized.attention_mask.sum(1)}")

    n_pos = tokenized.attention_mask.size(1)
    logger.debug(f"  => n_pos = {n_pos}")
    input_lengths = tokenized.attention_mask.sum(1)
    logger.debug(f"  => input_lengths = {input_lengths}")
    logger.debug("-" * 40)

    return n_pos, input_lengths


def make_hooks_and_matrices(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    batch_size: int,
    n_pos: int,
    scores,
):
    activation_difference = torch.zeros(
        (batch_size, n_pos, graph.n_forward, model.cfg.d_model),
        device=config.device,
        dtype=model.cfg.dtype,
    )

    processed_attn_layers = set()
    fwd_hooks_clean = []
    fwd_hooks_corrupted = []
    bwd_hooks = []

    def activation_hook(index, activations, hook, add: bool = True):
        acts = activations.detach()
        # print("acts shape", acts.shape)
        # print("activation_difference shape", activation_difference.shape)
        # print("index", index)
        if not add:
            acts = -acts
        try:
            activation_difference[:, :, index] += acts
        except RuntimeError as e:
            print(hook.name, activation_difference[:, :, index].size(), acts.size())
            raise e

    def gradient_hook(
        fwd_index: Union[slice, int],
        bwd_index: Union[slice, int],
        gradients: torch.Tensor,
        hook,
    ):
        grads = gradients.detach()
        try:
            if isinstance(fwd_index, slice):
                fwd_index = fwd_index.start
            if grads.ndim == 3:
                grads = grads.unsqueeze(2)
            s = einsum(
                activation_difference[:, :, :fwd_index],
                grads,
                "batch pos forward hidden, batch pos backward hidden -> forward backward",
            )
            s = s.squeeze(1)
            scores[:fwd_index, bwd_index] += s
        except RuntimeError as e:
            print(hook.name, activation_difference.size(), grads.size())
            raise e

    for name, node in graph.nodes.items():
        if isinstance(node, AttentionNode):
            if node.layer in processed_attn_layers:
                continue
            else:
                processed_attn_layers.add(node.layer)

        # exclude logits from forward
        fwd_index = graph.forward_index(node)
        if not isinstance(node, LogitNode):
            fwd_hooks_corrupted.append(
                (node.out_hook, partial(activation_hook, fwd_index))
            )
            fwd_hooks_clean.append(
                (node.out_hook, partial(activation_hook, fwd_index, add=False))
            )
        if not isinstance(node, InputNode):
            if isinstance(node, AttentionNode):
                for i, letter in enumerate("qkv"):
                    bwd_index = graph.backward_index(node, qkv=letter)
                    bwd_hooks.append(
                        (
                            node.qkv_inputs[i],
                            partial(gradient_hook, fwd_index, bwd_index),
                        )
                    )
            else:
                bwd_index = graph.backward_index(node)
                bwd_hooks.append(
                    (node.in_hook, partial(gradient_hook, fwd_index, bwd_index))
                )

    return (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference


def strip_trailing_eos(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_id: int = 128001
):
    """
    In-place removal of all trailing <|end_of_text|> tokens from each row.
    If a sequence has multiple EOS in a row, it removes them all.
    """
    batch_size, seq_len = input_ids.size()
    for i in range(batch_size):
        # Keep removing EOS tokens if present at the end
        while True:
            # Count how many tokens are actually present
            length = attention_mask[i].sum().item()  # number of non-pad tokens
            if length == 0:
                # Sequence is entirely empty/pad
                break

            last_idx = length - 1
            if input_ids[i, last_idx] == eos_id:
                # Zero out the final position
                input_ids[i, last_idx] = 0
                attention_mask[i, last_idx] = 0
            else:
                # The last token is not EOS; stop removing
                break


def get_scores(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device=config.device, dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        # print("clean", clean)
        # print("corrupted", corrupted)
        # print("label", label)
        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)
        # print("n_pos", n_pos)
        # print("input_lengths", input_lengths)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(config, model, graph, batch_size, n_pos, scores)
        )

        with model.hooks(fwd_hooks=fwd_hooks_corrupted):
            corrupted_logits = model(corrupted)

        with model.hooks(fwd_hooks=fwd_hooks_clean, bwd_hooks=bwd_hooks):
            logits = model(clean)
            # logits = model(model.to_tokens(clean))
            metric_value = metric(logits, corrupted_logits, input_lengths, label)
            metric_value.backward()

    scores /= total_items

    return scores


def get_scores_ig(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    steps=30,
    quiet=False,
):
    scores = torch.zeros(
        (graph.n_forward, graph.n_backward), device=config.device, dtype=model.cfg.dtype
    )

    total_items = 0
    dataloader = dataloader if quiet else tqdm(dataloader)
    for clean, corrupted, label in dataloader:
        batch_size = len(clean)
        total_items += batch_size
        n_pos, input_lengths = get_npos_input_lengths(model, clean)

        (fwd_hooks_corrupted, fwd_hooks_clean, bwd_hooks), activation_difference = (
            make_hooks_and_matrices(config, model, graph, batch_size, n_pos, scores)
        )

        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_hooks_corrupted):
                _ = model(corrupted)

            input_activations_corrupted = activation_difference[
                :, :, graph.forward_index(graph.nodes["input"])
            ].clone()

            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(clean)

            input_activations_clean = (
                input_activations_corrupted
                - activation_difference[:, :, graph.forward_index(graph.nodes["input"])]
            )

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                new_input = input_activations_corrupted + (k / steps) * (
                    input_activations_clean - input_activations_corrupted
                )
                new_input.requires_grad = True
                return new_input

            return hook_fn

        total_steps = 0
        for step in range(1, steps + 1):
            total_steps += 1
            with model.hooks(
                fwd_hooks=[
                    (graph.nodes["input"].out_hook, input_interpolation_hook(step))
                ],
                bwd_hooks=bwd_hooks,
            ):
                logits = model(clean)
                metric_value = metric(logits, clean_logits, input_lengths, label)
                metric_value.backward()

    scores /= total_items
    scores /= total_steps

    return scores


allowed_aggregations = {"sum", "mean", "l2"}


def attribute(
    config: Config,
    model: HookedTransformer,
    graph: Graph,
    dataloader: DataLoader,
    metric: Callable[[Tensor], Tensor],
    aggregation="sum",
    integrated_gradients: Optional[int] = None,
    quiet=False,
):
    if aggregation not in allowed_aggregations:
        raise ValueError(
            f"aggregation must be in {allowed_aggregations}, but got {aggregation}"
        )

    if integrated_gradients is None:
        scores = get_scores(config, model, graph, dataloader, metric, quiet=quiet)
    else:
        assert (
            integrated_gradients > 0
        ), f"integrated_gradients gives positive # steps (m), but got {integrated_gradients}"
        scores = get_scores_ig(
            config,
            model,
            graph,
            dataloader,
            metric,
            steps=integrated_gradients,
            quiet=quiet,
        )

        if aggregation == "mean":
            scores /= model.cfg.d_model
        elif aggregation == "l2":
            scores = torch.linalg.vector_norm(scores, ord=2, dim=-1)

    scores = scores.cpu().numpy()

    for edge in tqdm(graph.edges.values(), total=len(graph.edges)):
        edge.score = scores[
            graph.forward_index(edge.parent, attn_slice=False),
            graph.backward_index(edge.child, qkv=edge.qkv, attn_slice=False),
        ]
