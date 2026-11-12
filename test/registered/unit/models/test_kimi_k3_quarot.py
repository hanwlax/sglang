import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from sglang.srt.speculative.dspark_components.quarot import (
    align_quarot_projection,
    copy_unrotated_vocab_weight,
    kimi_k3_quarot_rotation_file,
    load_quarot_rotation,
    prepare_kimi_k3_quarot_draft,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

ROTATION_FILE = "optional/quarot.safetensors"


@pytest.fixture
def rotation():
    return torch.tensor([[0.6, -0.8], [0.8, 0.6]])


def test_load_rotation_tensor(tmp_path, rotation):
    path = tmp_path / ROTATION_FILE
    path.parent.mkdir()
    save_file({"global_rotation": rotation}, path)
    torch.testing.assert_close(
        load_quarot_rotation(tmp_path, ROTATION_FILE, 2), rotation
    )
    with pytest.raises(ValueError, match="floating-point 3 x 3"):
        load_quarot_rotation(tmp_path, ROTATION_FILE, 3)
    with pytest.raises(FileNotFoundError, match="rotation is missing"):
        load_quarot_rotation(tmp_path, "missing.safetensors", 2)


def test_align_all_five_context_features(rotation):
    hidden = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2)
    projection = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    expected = nn.functional.linear(hidden.flatten(1), projection)
    rotated_hidden = (hidden @ rotation).flatten(1)
    align_quarot_projection(projection, rotation, num_context_features=5)
    torch.testing.assert_close(
        nn.functional.linear(rotated_hidden, projection), expected
    )


def test_copy_vocab_weight_does_not_mutate_target(rotation):
    original = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    target = original @ rotation
    draft = torch.empty_like(target)
    copy_unrotated_vocab_weight(draft, target, rotation)
    torch.testing.assert_close(draft, original)
    torch.testing.assert_close(target, original @ rotation)
    assert draft.data_ptr() != target.data_ptr()
    with pytest.raises(ValueError, match="vocab shard"):
        copy_unrotated_vocab_weight(draft[:2], target, rotation)


@pytest.mark.parametrize("pipeline_parallel", [False, True])
def test_prepare_draft_uses_independent_vocab_layers(
    tmp_path, monkeypatch, rotation, pipeline_parallel
):
    from sglang.srt.layers.utils import PPMissingLayer
    from sglang.srt.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )
    from sglang.srt.model_loader.weight_utils import MODELSLIM_SAFETENSORS_INDEX_NAME
    from sglang.srt.models.dspark import DSparkDraftModel

    path = tmp_path / ROTATION_FILE
    path.parent.mkdir()
    save_file({"global_rotation": rotation}, path)
    draft = DSparkDraftModel.__new__(DSparkDraftModel)
    nn.Module.__init__(draft)
    draft.fc = nn.Linear(10, 2, bias=False)
    draft.num_context_features = 5
    draft.is_nemotron_35_draft = False
    head = ParallelLMHead(8, 2, enable_tp=False, padding_size=1)
    embed = VocabParallelEmbedding(8, 2, enable_tp=False, padding_size=1)
    original = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    with torch.no_grad():
        embed.weight.copy_(original @ rotation)
        head.weight.copy_((original + 1) @ rotation)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            model_path=str(tmp_path),
            vocab_size=8,
            hf_text_config=SimpleNamespace(hidden_size=2),
        ),
    )
    if pipeline_parallel:
        key = "language_model.model.embed_tokens.weight"
        shard = "quant_model_weights-00001-of-00001.safetensors"
        save_file({key: embed.weight.detach()}, tmp_path / shard)
        (tmp_path / MODELSLIM_SAFETENSORS_INDEX_NAME).write_text(
            json.dumps({"weight_map": {key: shard}})
        )
        monkeypatch.setattr(
            "sglang.srt.layers.vocab_parallel_embedding.get_embedding_tp_kwargs",
            lambda: {"enable_tp": False, "padding_size": 1},
        )
    draft_embed, draft_head = prepare_kimi_k3_quarot_draft(
        draft,
        runner,
        ROTATION_FILE,
        PPMissingLayer() if pipeline_parallel else embed,
        head,
    )
    assert draft_embed is not embed
    assert draft_head is not head
    torch.testing.assert_close(draft_embed.weight, original)
    torch.testing.assert_close(draft_head.weight, original + 1)
    torch.testing.assert_close(embed.weight, original @ rotation)
    torch.testing.assert_close(head.weight, (original + 1) @ rotation)


def test_modelslim_kda_gate_precision_controls_fusion():
    from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
    from sglang.srt.models.kimi_k3 import _can_fuse_kda_qkvg

    prefix = "model.layers.0.self_attn"
    weights = {
        f"language_model.{prefix}.{name}_proj.weight": "W8A8_DYNAMIC" for name in "qkv"
    }
    gate = f"language_model.{prefix}.g_proj.weight"
    assert not _can_fuse_kda_qkvg(ModelSlimConfig({**weights, gate: "FLOAT"}), prefix)
    assert _can_fuse_kda_qkvg(
        ModelSlimConfig({**weights, gate: "W8A8_DYNAMIC"}), prefix
    )
    assert _can_fuse_kda_qkvg(None, prefix)


def test_detect_rotation_from_kimi_k3_modelslim_descriptor():
    from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
    from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration

    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    model.quant_config = ModelSlimConfig({})
    assert kimi_k3_quarot_rotation_file(model) is None
    model.quant_config = ModelSlimConfig(
        {"optional": {"quarot": {"rotation_map": {"global_rotation": ROTATION_FILE}}}}
    )
    assert kimi_k3_quarot_rotation_file(model) == ROTATION_FILE
    assert (
        kimi_k3_quarot_rotation_file(SimpleNamespace(quant_config=model.quant_config))
        is None
    )


def test_split_kda_loads_int8_qkv_and_float_gate_separately():
    from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM

    prefix = "model.layers.0.self_attn"
    checkpoint = {
        f"{prefix}.q_proj.weight": torch.ones(1, dtype=torch.int8),
        f"{prefix}.k_proj.weight_scale": torch.ones(1),
        f"{prefix}.g_proj.weight": torch.ones(1, dtype=torch.bfloat16),
    }
    params = {}
    for name in ("qkv_proj.weight", "qkv_proj.weight_scale", "g_proj.weight"):
        param = nn.Parameter(torch.empty(1), requires_grad=False)
        param.weight_loader = Mock()
        params[f"{prefix}.{name}"] = param
    owner = SimpleNamespace(
        config=SimpleNamespace(
            linear_attn_config={"use_full_rank_gate": True},
            is_moe=False,
            is_linear_attn=True,
            num_hidden_layers=1,
            is_kda_layer=lambda layer_id: layer_id == 0,
        ),
        model=SimpleNamespace(
            start_layer=0,
            end_layer=1,
            layers=[SimpleNamespace(self_attn=SimpleNamespace(_fuse_qkvg=False))],
        ),
        named_parameters=lambda: params.items(),
        post_load_weights=Mock(),
    )
    loaded = KimiK3LinearForCausalLM.load_weights(owner, checkpoint.items())
    assert loaded == set(params)
    qkv = params[f"{prefix}.qkv_proj.weight"]
    qkv.weight_loader.assert_called_once_with(
        qkv, checkpoint[f"{prefix}.q_proj.weight"], "q"
    )
    scales = params[f"{prefix}.qkv_proj.weight_scale"]
    scales.weight_loader.assert_called_once_with(
        scales, checkpoint[f"{prefix}.k_proj.weight_scale"], "k"
    )
    gate = params[f"{prefix}.g_proj.weight"]
    gate.weight_loader.assert_called_once_with(
        gate, checkpoint[f"{prefix}.g_proj.weight"]
    )


def test_anchor_proposal_uses_aligned_embedding():
    from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer

    proposer = DraftBlockProposer.__new__(DraftBlockProposer)
    proposer.sample_from_anchor = True
    proposer.draft_anchor_embedding = nn.Embedding(8, 2)
    proposer._draft_sampler = None
    proposer._run_forward = Mock(side_effect=RuntimeError("embedding dispatched"))
    target = SimpleNamespace(get_input_embeddings=Mock())
    with pytest.raises(RuntimeError, match="embedding dispatched"):
        proposer.propose(
            batch=None,
            draft_input=None,
            verify_window=None,
            bs=1,
            device="cpu",
            target_model=target,
            sampling_info=None,
        )
    target.get_input_embeddings.assert_not_called()
    assert (
        proposer._run_forward.call_args.kwargs["embed_module"]
        is proposer.draft_anchor_embedding
    )


def test_vision_projector_applies_checkpoint_rotation_after_norm(rotation):
    from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
    from sglang.srt.models.kimi_k3_vl import KimiK3MultiModalProjector

    projector = KimiK3MultiModalProjector(
        SimpleNamespace(
            vt_hidden_size=2,
            merge_kernel_size=(1, 1),
            text_hidden_size=2,
            projector_ln_eps=1e-5,
        ),
        has_rotation=True,
    )
    image = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(language_only=False)
    model.language_model = None
    model.mm_projector = projector
    model.load_weights([("mm_projector.rot_proj.weight", rotation.T)])
    expected = projector.post_norm(projector.proj(image)) @ rotation
    torch.testing.assert_close(projector(image), expected)


def test_modelslim_loader_selects_indexed_shards(tmp_path, monkeypatch):
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.model_loader import loader
    from sglang.srt.model_loader.weight_utils import MODELSLIM_SAFETENSORS_INDEX_NAME

    shard = "quant_model_weights-00001-of-00001.safetensors"
    (tmp_path / shard).touch()
    (tmp_path / "unindexed.safetensors").touch()
    (tmp_path / MODELSLIM_SAFETENSORS_INDEX_NAME).write_text(
        json.dumps({"weight_map": {"weight": shard}})
    )
    monkeypatch.setattr(loader, "get_server_args", lambda: None)
    checkpoint_loader = loader.DefaultModelLoader(LoadConfig())
    _, weights, is_safetensors = checkpoint_loader._prepare_weights(
        str(tmp_path), revision=None, fall_back_to_pt=False
    )
    assert is_safetensors
    assert weights == [str(tmp_path / shard)]
    (tmp_path / shard).unlink()
    with pytest.raises(RuntimeError, match="incomplete download"):
        checkpoint_loader._prepare_weights(
            str(tmp_path), revision=None, fall_back_to_pt=False
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
