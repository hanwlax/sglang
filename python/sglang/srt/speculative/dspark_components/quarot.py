"""Align a Kimi K3 ModelSlim target with a DSpark draft in the original basis."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

QUAROT_DRAFT_WEIGHT_UPDATE_ERROR = "Kimi K3 QuaRot DSpark caches derived weights. Restart the server to change weights."


def kimi_k3_quarot_rotation_file(target_model) -> str | None:
    quant_config = getattr(target_model, "quant_config", None)
    rotation_file = getattr(quant_config, "quarot_rotation_file", None)
    if rotation_file is None:
        return None

    from sglang.srt.models.kimi_k3 import (
        KimiK3ForConditionalGeneration,
        KimiK3LinearForCausalLM,
    )

    if isinstance(
        target_model, (KimiK3ForConditionalGeneration, KimiK3LinearForCausalLM)
    ):
        return rotation_file
    return None


def load_quarot_rotation(
    checkpoint: Path, filename: str, hidden_size: int
) -> torch.Tensor:
    relative_path = Path(filename)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Invalid QuaRot rotation path: {filename}")
    path = checkpoint / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Kimi K3 QuaRot rotation is missing: {path}")
    with safe_open(path, framework="pt", device="cpu") as tensors:
        if "global_rotation" not in tensors.keys():
            raise ValueError(f"No global_rotation tensor in {path}")
        rotation = tensors.get_tensor("global_rotation")
    if rotation.shape != (hidden_size, hidden_size) or not rotation.is_floating_point():
        raise ValueError(
            f"Expected a floating-point {hidden_size} x {hidden_size} "
            f"QuaRot rotation in {path}, got {tuple(rotation.shape)} {rotation.dtype}"
        )
    return rotation


@torch.no_grad()
def align_quarot_projection(
    weight: torch.Tensor, rotation: torch.Tensor, num_context_features: int
) -> None:
    hidden_size = rotation.shape[0]
    if (
        weight.shape
        != (
            hidden_size,
            num_context_features * hidden_size,
        )
        or not weight.is_floating_point()
    ):
        raise ValueError(
            f"Cannot align DSpark fc.weight of shape {tuple(weight.shape)} "
            f"and dtype {weight.dtype} with QuaRot"
        )
    # Target activations arrive as H @ R. Fold R into each input block of
    # the draft projection, keeping the draft activations in their own basis.
    for block in weight.split(hidden_size, dim=1):
        block.copy_(block.to(device=rotation.device, dtype=torch.float32) @ rotation)


@torch.no_grad()
def copy_unrotated_vocab_weight(
    destination: torch.Tensor, source: torch.Tensor, rotation: torch.Tensor
) -> None:
    if (
        source.shape != destination.shape
        or source.ndim != 2
        or source.shape[1] != rotation.shape[0]
        or not source.is_floating_point()
    ):
        raise ValueError(
            "Kimi K3 QuaRot requires a compatible floating-point vocab shard"
        )
    if source.numel():
        destination.copy_(
            source.to(device=rotation.device, dtype=torch.float32) @ rotation.T
        )


def _load_embedding_shard(checkpoint: Path, embedding) -> torch.Tensor:
    from sglang.srt.model_loader.weight_utils import get_safetensors_index_path

    index = get_safetensors_index_path(checkpoint)
    if index is None:
        raise ValueError(
            f"QuaRot pipeline parallelism requires a shard index in {checkpoint}"
        )
    with index.open() as f:
        weight_map = json.load(f)["weight_map"]
    for name in (
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ):
        if name in weight_map:
            break
    else:
        raise ValueError(f"No Kimi K3 embedding in {index}")

    with safe_open(checkpoint / weight_map[name], framework="pt", device="cpu") as f:
        weight = f.get_slice(name)
        if weight.get_shape() != [embedding.org_vocab_size, embedding.embedding_dim]:
            raise ValueError(f"Unexpected shape of {name} in {index}")
        shard = embedding.shard_indices
        return weight[shard.org_vocab_start_index : shard.org_vocab_end_index]


def _unrotate_vocab_layer(layer, rotation: torch.Tensor):
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    if not isinstance(layer, VocabParallelEmbedding):
        raise ValueError(
            "Kimi K3 QuaRot requires a vocab-parallel embedding and LM head"
        )
    layer_type = (
        ParallelLMHead if isinstance(layer, ParallelLMHead) else VocabParallelEmbedding
    )
    kwargs = (
        {"output_dtype": layer.output_dtype}
        if layer_type is VocabParallelEmbedding
        else {}
    )
    with torch.device(rotation.device):
        aligned = layer_type(
            layer.num_embeddings,
            layer.embedding_dim,
            params_dtype=layer.weight.dtype,
            org_num_embeddings=layer.org_vocab_size,
            padding_size=layer.padding_size,
            enable_tp=layer.enable_tp,
            use_attn_tp_group=layer.use_attn_tp_group,
            use_presharded_weights=layer.use_presharded_weights,
            **kwargs,
        )
    if (
        aligned.shard_indices != layer.shard_indices
        or aligned.tp_group is not layer.tp_group
    ):
        raise ValueError(
            "Kimi K3 QuaRot draft vocab shard differs from the target shard"
        )
    copy_unrotated_vocab_weight(aligned.weight, layer.weight, rotation)
    return aligned


@torch.no_grad()
def prepare_kimi_k3_quarot_draft(
    draft, target_runner, rotation_file, embed_tokens, lm_head
):
    """Align the draft projection and return independent vocab modules."""
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
        get_embedding_tp_kwargs,
    )
    from sglang.srt.models.dspark import DSparkDraftModel

    if (
        not isinstance(draft, DSparkDraftModel)
        or draft.is_nemotron_35_draft
        or getattr(draft, "uses_own_vocab_modules", False)
    ):
        raise ValueError(
            "Kimi K3 QuaRot currently requires a compatible GQA DSpark draft"
        )

    config = target_runner.model_config
    checkpoint = Path(config.model_path)
    if not checkpoint.is_dir():
        from sglang.srt.speculative.pp_draft_embedding import prepare_checkpoint_files

        folder, _, _ = prepare_checkpoint_files(
            config.model_path,
            revision=config.revision,
            load_config=target_runner.load_config,
        )
        checkpoint = Path(folder)
    hidden_size = config.hf_text_config.hidden_size
    rotation = load_quarot_rotation(checkpoint, rotation_file, hidden_size).float()
    # Match vLLM Ascend's FP32 CPU projection fold. Vocab shards are much
    # smaller per rank and are transformed on their destination device.
    align_quarot_projection(draft.fc.weight, rotation, draft.num_context_features)
    rotation = rotation.to(lm_head.weight.device)
    if isinstance(embed_tokens, PPMissingLayer):
        with torch.device(rotation.device):
            embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                hidden_size,
                params_dtype=lm_head.weight.dtype,
                **get_embedding_tp_kwargs(),
            )
        source = _load_embedding_shard(checkpoint, embed_tokens)
        embed_tokens.weight.zero_()
        copy_unrotated_vocab_weight(
            embed_tokens.weight[: source.shape[0]], source, rotation
        )
    else:
        embed_tokens = _unrotate_vocab_layer(embed_tokens, rotation)
    lm_head = _unrotate_vocab_layer(lm_head, rotation)
    draft._derived_weight_cache_error = QUAROT_DRAFT_WEIGHT_UPDATE_ERROR
    return embed_tokens, lm_head
