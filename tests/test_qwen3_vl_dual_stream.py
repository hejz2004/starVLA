"""Structural acceptance tests for the dual-stream Qwen3-VL vision tower.

Covers the backbone-fusion variant on a small random-config vision tower
(CPU, fp32). Images within a scene interact, while distinct batch samples
are isolated.

1. gamma=0 diagnostic restores the original model outputs
2. view independence before fusion, cross-view influence in subsequent blocks
3. cross-view path exists (other views influence F_i through S)
4. view permutation equivariance (S invariant, F equivariant)
5. variable inputs (mixed resolutions / view counts / single view)
6. gate sanity (shape, (0,1) range, zero-forcing)
7. save/restore of the new config + weights
8. registry regression: original QwenPI unaffected

Run inside the training container:  pytest tests/test_qwen3_vl_dual_stream.py -q
"""

import copy
import json
import tempfile
import unittest

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from starVLA.model.modules.vlm.qwen3_vl_dual.modeling_qwen3_vl import (
    Qwen3VLDualStreamGateMLP,
    Qwen3VLVisionModel,
    _image_counts_from_input_ids,
)

torch.manual_seed(0)

HIDDEN = 64
HEADS = 4
DEPTH = 6
DEEPSTACK = [1, 3]
INTERACTION = sorted(set(DEEPSTACK) | {DEPTH - 1})  # [1, 3, 5]
MERGE = 2
PATCH = 16


def make_vision_config(**overrides):
    cfg = Qwen3VLVisionConfig(
        depth=DEPTH,
        hidden_size=HIDDEN,
        num_heads=HEADS,
        intermediate_size=128,
        out_hidden_size=128,
        patch_size=PATCH,
        spatial_merge_size=MERGE,
        temporal_patch_size=2,
        in_channels=3,
        hidden_act="gelu_pytorch_tanh",
        initializer_range=0.02,
        num_position_embeddings=1024,
        deepstack_visual_indexes=list(DEEPSTACK),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


DUAL_STREAM_CFG = {
    "enabled": True,
    "interaction_indexes": list(INTERACTION),
    "num_shared_tokens": 8,
    "num_heads": HEADS,
    "intermediate_size": 128,
    "gate_hidden_size": 16,
    "gate_init_value": 0.01,
    "gate_output_weight_std": 0.001,
    "attention_dropout": 0.0,
}


def pixels_for(grids, batch=1):
    """Build packed patch pixel_values for the given [(t, h, w)] grids."""
    pixel_values = []
    for t, h, w in grids:
        n = t * h * w
        pixel_values.append(torch.randn(n, 3 * 2 * PATCH * PATCH, batch, 1).squeeze(-1).squeeze(-1))
    return torch.cat(pixel_values, dim=0)


class DualStreamVisionTest(unittest.TestCase):
    def _build_pair(self, **dual_overrides):
        base_cfg = make_vision_config()
        orig = Qwen3VLVisionModel(base_cfg)
        orig.post_init()

        dual_cfg = make_vision_config()
        dual_cfg.dual_stream = dict(DUAL_STREAM_CFG, **dual_overrides)
        dual = Qwen3VLVisionModel(dual_cfg)
        dual.post_init()

        # Copy all shared (upstream) weights so the two towers differ only by
        # the dual-stream modules.
        shared_state = {k: v for k, v in orig.state_dict().items() if not k.startswith("shared_init")}
        missing, unexpected = dual.load_state_dict(shared_state, strict=False)
        expected_missing = {"shared_init"} | {
            k for k in dual.state_dict() if k.startswith("interaction_stages.")
        }
        self.assertEqual(set(missing), expected_missing, f"unexpected missing keys: {missing}")
        self.assertEqual(unexpected, [], f"unexpected keys: {unexpected}")
        return orig, dual

    def _forward(self, model, grids, pixel_values=None, grid_thw=None):
        if pixel_values is None:
            pixel_values = pixels_for(grids)
        if grid_thw is None:
            grid_thw = torch.tensor(grids, dtype=torch.long)
        return model(pixel_values, grid_thw=grid_thw)

    # ---- 1. gamma=0 restores the original model -------------------------
    def test_force_zero_gate_restores_original_output(self):
        orig, dual = self._build_pair()
        grids = [(1, 4, 4), (1, 4, 4)]
        pixel_values = pixels_for(grids)
        grid_thw = torch.tensor(grids, dtype=torch.long)
        orig.set_force_zero_gate(True)
        out_orig, ds_orig = self._forward(orig, grids, pixel_values=pixel_values, grid_thw=grid_thw)
        dual.set_force_zero_gate(True)
        out_dual, ds_dual = self._forward(dual, grids, pixel_values=pixel_values, grid_thw=grid_thw)
        torch.testing.assert_close(out_orig, out_dual, rtol=0, atol=1e-6)
        self.assertEqual(len(ds_orig), len(DEEPSTACK))
        self.assertEqual(len(ds_dual), len(DEEPSTACK))
        for a, b in zip(ds_orig, ds_dual):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)

    def test_gate_bias_init(self):
        _, dual = self._build_pair()
        for stage in dual.interaction_stages.values():
            g0 = 0.01
            expected_bias = torch.tensor(torch.log(torch.tensor(g0 / (1 - g0))).item())
            self.assertTrue(
                torch.allclose(stage.gate_mlp.fc2.bias, expected_bias.expand_as(stage.gate_mlp.fc2.bias)),
                "gate fc2 bias must be log(g0/(1-g0)) after post_init",
            )
            # fc2 weights must be tiny but nonzero
            self.assertGreater(stage.gate_mlp.fc2.weight.abs().max().item(), 0)
            self.assertLess(stage.gate_mlp.fc2.weight.abs().max().item(), 10 * 0.001)

    # ---- 2. cross-view fusion enters subsequent backbone blocks ----------
    def test_fusion_changes_subsequent_backbone_features_and_gradients(self):
        _, dual = self._build_pair()
        dual.eval()
        # Exercise both the batched equal-length path and mixed-length path.
        for grids, scene_counts in [
            ([(1, 4, 4)] * 4, [2, 2]),
            ([(1, 4, 4), (1, 2, 2), (1, 4, 2)], [2, 1]),
        ]:
            with self.subTest(grids=grids):
                captured = {}

                def capture(key):
                    def hook(module, args, output):
                        captured[key] = output
                    return hook

                handles = [
                    dual.blocks[INTERACTION[0]].register_forward_hook(capture("before")),
                    dual.blocks[INTERACTION[0] + 1].register_forward_hook(capture("after")),
                ]
                try:
                    grid_thw = torch.tensor(grids, dtype=torch.long)
                    pixels = pixels_for(grids)
                    n_img1 = grids[0][1] * grids[0][2]
                    scene_end = n_img1 + grids[1][1] * grids[1][2]
                    with torch.no_grad():
                        dual(pixels, grid_thw=grid_thw, scene_image_counts=scene_counts)
                        baseline = {key: value.clone() for key, value in captured.items()}
                        changed = pixels.clone()
                        changed[n_img1:scene_end] += 10.0
                        dual(changed, grid_thw=grid_thw, scene_image_counts=scene_counts)

                    # Before the first interaction, views are still independent.
                    torch.testing.assert_close(
                        baseline["before"][:n_img1], captured["before"][:n_img1], rtol=0, atol=0
                    )
                    # A regular ViT block must now see the other view's information.
                    self.assertFalse(torch.equal(
                        baseline["after"][:n_img1], captured["after"][:n_img1]
                    ))
                    torch.testing.assert_close(
                        baseline["after"][scene_end:], captured["after"][scene_end:], rtol=0, atol=0
                    )

                    dual.zero_grad(set_to_none=True)
                    probe = pixels.clone().requires_grad_(True)
                    dual(probe, grid_thw=grid_thw, scene_image_counts=scene_counts)
                    # No merger/DeepStack loss: gradients must travel through
                    # the regular backbone block and the first fusion stage.
                    captured["after"][:n_img1].square().mean().backward()
                    self.assertTrue(torch.isfinite(probe.grad).all())
                    self.assertGreater(probe.grad[n_img1:scene_end].abs().sum().item(), 0)
                    self.assertTrue(torch.all(probe.grad[scene_end:] == 0))
                    stage_grad = dual.interaction_stages[str(INTERACTION[0])].o3.weight.grad
                    self.assertIsNotNone(stage_grad)
                    self.assertTrue(torch.isfinite(stage_grad).all())
                    self.assertGreater(stage_grad.abs().sum().item(), 0)
                finally:
                    for handle in handles:
                        handle.remove()

    # ---- 3. cross-view path ---------------------------------------------
    def test_attention_matches_explicit_per_head_reference(self):
        _, dual = self._build_pair()
        stage = dual.interaction_stages[str(INTERACTION[0])]
        for batched in (False, True):
            with self.subTest(batched=batched):
                prefix = (2,) if batched else ()
                q = torch.randn(*prefix, 7, HIDDEN)
                k = torch.randn(*prefix, 11, HIDDEN)
                v = torch.randn(*prefix, 11, HIDDEN)
                heads = []
                for head in range(HEADS):
                    channels = slice(head * stage.head_dim, (head + 1) * stage.head_dim)
                    scores = q[..., channels] @ k[..., channels].transpose(-1, -2)
                    weights = (scores * stage.scaling).softmax(dim=-1)
                    heads.append(weights @ v[..., channels])
                expected = torch.cat(heads, dim=-1)
                torch.testing.assert_close(stage._sdpa(q, k, v), expected, rtol=1e-5, atol=1e-6)

    def test_cross_view_path_and_gradients(self):
        _, dual = self._build_pair()
        dual.eval()
        grids = [(1, 4, 4), (1, 4, 4)]
        n_img1 = grids[0][1] * grids[0][2]

        pixel_values = pixels_for(grids)
        grid_thw = torch.tensor(grids, dtype=torch.long)
        baseline_merged, baseline_ds = dual(pixel_values, grid_thw=grid_thw)

        perturbed = pixel_values.clone()
        perturbed[n_img1:] += 10.0
        pert_merged, pert_ds = dual(perturbed, grid_thw=grid_thw)

        n_img1_merged = (grids[0][1] // MERGE) * (grids[0][2] // MERGE)
        changed = not torch.allclose(
            baseline_merged[:n_img1_merged], pert_merged[:n_img1_merged], rtol=0, atol=0
        )
        self.assertTrue(
            changed,
            "with gamma>0, perturbing image 2 must change image 1's fused output via S",
        )

        # Every new parameter must receive a finite gradient from the task loss.
        dual.train()
        probe = pixel_values.clone().requires_grad_(True)
        merged, deepstack = dual(probe, grid_thw=grid_thw)
        loss = merged.sum() + sum(f.sum() for f in deepstack)
        loss.backward()
        self.assertTrue(torch.isfinite(dual.shared_init.grad).all())
        for name, param in dual.interaction_stages.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} received no gradient")
            self.assertTrue(torch.isfinite(param.grad).all(), f"{name} gradient is not finite")

    def test_scene_isolation_and_shared_gradient(self):
        _, dual = self._build_pair()
        dual.eval()
        grids = [(1, 4, 4)] * 4  # two scenes, two views each
        pixel_values = pixels_for(grids)
        grid_thw = torch.tensor(grids, dtype=torch.long)
        scene_patches = 2 * 4 * 4
        scene_merged = 2 * (4 // MERGE) ** 2

        baseline, _ = dual(pixel_values, grid_thw=grid_thw, scene_image_counts=[2, 2])
        changed = pixel_values.clone()
        changed[scene_patches:] += 10.0  # change only scene 2
        perturbed, _ = dual(changed, grid_thw=grid_thw, scene_image_counts=[2, 2])
        torch.testing.assert_close(baseline[:scene_merged], perturbed[:scene_merged], rtol=0, atol=0)

        # A different view of scene 1 must still influence scene 1's output.
        changed = pixel_values.clone()
        changed[16:32] += 10.0
        perturbed, _ = dual(changed, grid_thw=grid_thw, scene_image_counts=[2, 2])
        self.assertFalse(torch.allclose(baseline[:4], perturbed[:4], rtol=0, atol=0))

        probe = pixel_values.clone().requires_grad_(True)
        merged, _ = dual(probe, grid_thw=grid_thw, scene_image_counts=[2, 2])
        merged[:scene_merged].sum().backward()
        self.assertTrue(torch.all(probe.grad[scene_patches:] == 0))
        self.assertIsNotNone(dual.shared_init.grad)
        self.assertTrue(torch.isfinite(dual.shared_init.grad).all())
        self.assertGreater(dual.shared_init.grad.abs().sum().item(), 0)

    def test_variable_scene_lengths_are_isolated(self):
        _, dual = self._build_pair()
        dual.eval()
        grids = [(1, 4, 4), (1, 2, 2), (1, 4, 2)]
        grid_thw = torch.tensor(grids, dtype=torch.long)
        pixels = pixels_for(grids)
        baseline, _ = dual(pixels, grid_thw=grid_thw, scene_image_counts=[1, 2])
        changed = pixels.clone()
        changed[16:] += 10.0
        perturbed, _ = dual(changed, grid_thw=grid_thw, scene_image_counts=[1, 2])
        torch.testing.assert_close(baseline[:4], perturbed[:4], rtol=0, atol=0)
        self.assertEqual(baseline.shape[0], 7)

    def test_image_counts_follow_sample_boundaries(self):
        starts, image = 151652, 151655
        input_ids = torch.tensor([
            [0, starts, image, image, 1, starts, image, image],
            [0, 0, 1, starts, image, image, 2, 2],
        ])
        self.assertEqual(_image_counts_from_input_ids(input_ids, starts, image, 3), [2, 1])
        with self.assertRaises(ValueError):
            _image_counts_from_input_ids(input_ids, starts, image, 2)

    # ---- 4. permutation equivariance -------------------------------------
    def test_view_permutation_equivariance(self):
        _, dual = self._build_pair()
        dual.eval()
        grids = [(1, 4, 4), (1, 2, 2)]
        pixel_values = pixels_for(grids)
        grid_thw = torch.tensor(grids, dtype=torch.long)

        shared_states = {}

        def make_hook(key):
            def hook(module, args, output):
                shared_states.setdefault(key, []).append(output[0].detach().clone())

            return hook

        handles = [
            stage.register_forward_hook(make_hook(name))
            for name, stage in dual.interaction_stages.items()
        ]
        out_ab, _ = dual(pixel_values, grid_thw=grid_thw)
        shared_ab = {k: [t.clone() for t in v] for k, v in shared_states.items()}

        shared_states.clear()
        perm = torch.tensor([1, 0], dtype=torch.long)
        n_img = [grids[0][1] * grids[0][2], grids[1][1] * grids[1][2]]
        pixel_values_ba = torch.cat([pixel_values[n_img[0] :], pixel_values[: n_img[0]]], dim=0)
        grid_ba = grid_thw[perm]
        out_ba, _ = dual(pixel_values_ba, grid_thw=grid_ba)
        for handle in handles:
            handle.remove()

        for name in dual.interaction_stages.keys():
            for a, b in zip(shared_ab[name], shared_states[name]):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

        n_a = (grids[0][1] // MERGE) * (grids[0][2] // MERGE)
        n_b = (grids[1][1] // MERGE) * (grids[1][2] // MERGE)
        # Softmax over the pooled tokens is mathematically permutation
        # invariant, but the fp summation order changes with the packing
        # order, so allow small numeric slack for the fused outputs.
        torch.testing.assert_close(out_ba[:n_b], out_ab[n_a:], rtol=1e-3, atol=1e-4)
        torch.testing.assert_close(out_ba[n_b:], out_ab[:n_a], rtol=1e-3, atol=1e-4)

    # ---- 5. variable inputs ----------------------------------------------
    def test_variable_inputs(self):
        _, dual = self._build_pair()
        dual.eval()
        grids = [(1, 2, 2), (1, 4, 2), (1, 2, 4)]
        merged, deepstack = self._forward(dual, grids)
        n_expected = sum((h // MERGE) * (w // MERGE) for _, h, w in grids)
        self.assertEqual(merged.shape[0], n_expected)
        for f in deepstack:
            self.assertEqual(f.shape[0], n_expected)
            self.assertEqual(f.shape[-1], 128)

        single, deepstack_single = self._forward(dual, grids[:1])
        self.assertEqual(single.shape[0], 1)
        for f in deepstack_single:
            self.assertEqual(f.shape[0], 1)

    # ---- 6. gate sanity ---------------------------------------------------
    def test_gate_mlp_shape_and_range(self):
        gate = Qwen3VLDualStreamGateMLP(hidden_size=HIDDEN, gate_hidden_size=16)
        x = torch.randn(3, 7, 2 * HIDDEN)
        logits = gate(x)
        self.assertEqual(logits.shape, (3, 7, 1))
        gamma = torch.sigmoid(logits)
        self.assertTrue(((gamma > 0) & (gamma < 1)).all())

    def test_interaction_indexes_validation(self):
        for indexes in ([], [-1], [DEPTH], [1, 1], [1.5], [True], "1,3"):
            with self.subTest(indexes=indexes), self.assertRaises(ValueError):
                self._build_pair(interaction_indexes=indexes)
        for tokens in (0, -1, 1.5, True):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self._build_pair(num_shared_tokens=tokens)

    def test_configurable_fusion_sites_and_shared_token_counts(self):
        # None of these sites coincide with DeepStack or the last vision block.
        # Also cover a single early interaction that must survive to the merger.
        for indexes, tokens in (([4, 0, 2], 1), ([0], 16)):
            with self.subTest(indexes=indexes, tokens=tokens):
                orig, dual = self._build_pair(
                    interaction_indexes=indexes, num_shared_tokens=tokens
                )
                orig.eval()
                dual.eval()
                self.assertEqual(dual.dual_stream_interaction_indexes, sorted(indexes))
                self.assertEqual(list(dual.interaction_stages), [str(i) for i in sorted(indexes)])
                self.assertEqual(tuple(dual.shared_init.shape), (tokens, HIDDEN))
                self.assertEqual(dual.deepstack_visual_indexes, DEEPSTACK)
                grid = torch.tensor([(1, 4, 4), (1, 2, 2)])
                pixels = pixels_for(grid.tolist())
                baseline, _ = orig(pixels, grid_thw=grid)
                output, deepstack = dual(pixels, grid_thw=grid)
                self.assertEqual(output.shape, baseline.shape)
                self.assertEqual(len(deepstack), len(DEEPSTACK))
                self.assertFalse(torch.equal(output, baseline))
                output.square().mean().backward()
                for name, param in dual.named_parameters():
                    if name == "shared_init" or name.startswith("interaction_stages."):
                        self.assertIsNotNone(param.grad, name)
                        self.assertTrue(torch.isfinite(param.grad).all(), name)
                        self.assertGreater(param.grad.abs().sum().item(), 0, name)
                with tempfile.TemporaryDirectory() as tmp:
                    dual.config.save_pretrained(tmp)
                    cfg = Qwen3VLVisionConfig.from_pretrained(tmp)
                    restored = Qwen3VLVisionModel(cfg)
                    restored.load_state_dict(dual.state_dict(), strict=True)
                    restored.eval()
                    restored_output, restored_ds = restored(pixels, grid_thw=grid)
                    torch.testing.assert_close(restored_output, output, rtol=0, atol=0)
                    for actual, expected in zip(restored_ds, deepstack):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_default_interaction_sites(self):
        for explicit_null in (False, True):
            cfg = make_vision_config()
            cfg.dual_stream = dict(DUAL_STREAM_CFG)
            cfg.dual_stream.pop("interaction_indexes")
            if explicit_null:
                cfg.dual_stream["interaction_indexes"] = None
            dual = Qwen3VLVisionModel(cfg)
            self.assertEqual(dual.dual_stream_interaction_indexes, INTERACTION)
            self.assertEqual(dual.config.dual_stream["interaction_indexes"], INTERACTION)

    # ---- 7. save / restore -------------------------------------------------
    def test_state_dict_roundtrip_and_config_serialization(self):
        _, dual = self._build_pair()
        state = copy.deepcopy(dual.state_dict())
        self.assertIn("shared_init", state)
        for layer_index in INTERACTION:
            self.assertIn(f"interaction_stages.{layer_index}.q1.weight", state)
            self.assertIn(f"interaction_stages.{layer_index}.gate_mlp.fc2.bias", state)

        cfg2 = make_vision_config()
        cfg2.dual_stream = dict(DUAL_STREAM_CFG)
        dual2 = Qwen3VLVisionModel(cfg2)
        dual2.load_state_dict(state, strict=True)
        for stage, stage2 in zip(dual.interaction_stages.values(), dual2.interaction_stages.values()):
            torch.testing.assert_close(stage.gate_mlp.fc2.bias, stage2.gate_mlp.fc2.bias)

        # The dual_stream block must survive config save/load.
        with tempfile.TemporaryDirectory() as tmp:
            dual.config.save_pretrained(tmp)
            with open(f"{tmp}/config.json") as f:
                raw = json.load(f)
            self.assertTrue(raw["dual_stream"]["enabled"])
            reloaded = Qwen3VLVisionConfig.from_pretrained(tmp)
            self.assertEqual(reloaded.dual_stream["num_shared_tokens"], DUAL_STREAM_CFG["num_shared_tokens"])

        # And a reloaded tower reproduces the outputs bit-exactly.
        dual.eval()
        dual2.eval()
        grids = [(1, 4, 2), (1, 2, 4)]
        pixel_values = pixels_for(grids)
        grid_thw = torch.tensor(grids, dtype=torch.long)
        out1, ds1 = self._forward(dual, grids, pixel_values=pixel_values, grid_thw=grid_thw)
        out2, ds2 = self._forward(dual2, grids, pixel_values=pixel_values, grid_thw=grid_thw)
        torch.testing.assert_close(out1, out2, rtol=0, atol=0)
        for a, b in zip(ds1, ds2):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    # ---- 8. registry regression --------------------------------------------
    def test_qwenpi_registry_unaffected(self):
        from starVLA.model.framework.base_framework import build_framework, _auto_import_framework_modules
        from starVLA.model.tools import FRAMEWORK_REGISTRY

        _auto_import_framework_modules()
        registered = FRAMEWORK_REGISTRY.list()
        self.assertIn("QwenPI", registered)
        self.assertIn("QwenFM", registered)
        self.assertIn("DualStreamQwenPI", registered)
        self.assertIn("DualStreamQwenFM", registered)
        self.assertIs(registered["QwenPI"].__name__, "Qwen_PI" if False else registered["QwenPI"].__name__)
        self.assertEqual(registered["QwenPI"].__name__, "Qwen_PI")
        self.assertEqual(registered["DualStreamQwenPI"].__name__, "DualStreamQwen_PI")
        # the plain interface must still be importable and untouched
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface

        self.assertTrue(callable(_QWen3_VL_Interface))


if __name__ == "__main__":
    unittest.main()
