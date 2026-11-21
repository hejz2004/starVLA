"""Small-width checks of image batching through all four DualStream stages."""
import copy

import pytest
import torch

from starVLA.model.modules.vlm.qwen3_vl_dual.modeling_qwen3_vl import Qwen3VLVisionModel
from tests.test_dualstream_optimizations import tiny_model


def assert_relative(actual, expected, tolerance, label):
    a, b = actual.detach().float(), expected.detach().float()
    error = float((a-b).norm() / b.norm().clamp_min(1e-12))
    assert torch.isfinite(a).all() and error < tolerance, (label, error)
    return error


@pytest.mark.parametrize('mixed_sizes', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_image_batching_preserves_four_stage_dualstream(mixed_sizes, device):
    torch.manual_seed(73)
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    cfg = copy.deepcopy(tiny_model(production_geometry=True).config.vision_config)
    cfg.depth = 24
    cfg.deepstack_visual_indexes = [5, 11, 17]
    cfg.dual_stream['interaction_indexes'] = [5, 11, 17, 23]
    cfg.batch_vision_attention = False
    baseline = Qwen3VLVisionModel(cfg).to(device=device, dtype=dtype).eval()
    batched = copy.deepcopy(baseline)
    batched.config.batch_vision_attention = True
    grids = ([[1,4,4],[1,4,6],[1,4,4]] if mixed_sizes else [[1,4,4]] * 4)
    scene_counts = [2,1] if mixed_sizes else [2,2]
    grid = torch.tensor(grids, device=device)
    first_view = 16
    scene_end = sum(t*h*w for t,h,w in grids[:2])
    pixels = torch.randn(int(grid.prod(-1).sum()), 3*2*16*16, device=device, dtype=dtype)
    captured = {}
    handles = []

    def capture(key):
        def hook(module, args, output):
            if len(key) == 3:
                captured.setdefault(key, []).append(output)
            else:
                captured[key] = output
        return hook

    for prefix, model in [('old',baseline), ('new',batched)]:
        for index, stage in model.interaction_stages.items():
            # Include each stage's shared state and fused patches.
            handles.append(stage.register_forward_hook(capture((prefix,'stage',index))))
        handles.append(model.blocks[5].register_forward_hook(capture((prefix,'before'))))
        handles.append(model.blocks[6].register_forward_hook(capture((prefix,'after'))))
    try:
        old_pixels = pixels.clone().requires_grad_()
        new_pixels = pixels.clone().requires_grad_()
        old = baseline(old_pixels, grid, scene_image_counts=scene_counts)
        new = batched(new_pixels, grid, scene_image_counts=scene_counts)
        tolerance = 1e-5 if device == 'cpu' else .02
        output_errors = []
        for a,b in [(new[0],old[0]), *zip(new[1],old[1])]:
            output_errors.append(assert_relative(a,b,tolerance,'merger/DeepStack'))
        for index in ['5','11','17','23']:
            old_calls = captured[('old','stage',index)]
            new_calls = captured[('new','stage',index)]
            assert len(old_calls) == len(new_calls)
            for new_call,old_call in zip(new_calls,old_calls):
                for a,b in zip(new_call,old_call):
                    assert_relative(a,b,tolerance,'shared state/fused patches')
        old_loss = sum(x.float().square().mean() for x in [old[0], *old[1]])
        new_loss = sum(x.float().square().mean() for x in [new[0], *new[1]])
        old_loss.backward(); new_loss.backward()
        grad_tolerance = 1e-4 if device == 'cpu' else .03
        assert_relative(new_pixels.grad,old_pixels.grad,grad_tolerance,'input gradients')
        old_grads, new_grads = [], []
        for (name,a),(_,b) in zip(baseline.named_parameters(),batched.named_parameters()):
            assert (a.grad is None)==(b.grad is None), name
            if a.grad is not None:
                old_grads.append(a.grad.float().flatten())
                new_grads.append(b.grad.float().flatten())
        gradient_error = assert_relative(torch.cat(new_grads),torch.cat(old_grads),grad_tolerance,'all parameters')

        # Perturb only the second view of scene 1, including a different-size
        # image in the mixed bucket case. Scene 2 must remain exactly unchanged.
        with torch.no_grad():
            batched(pixels,grid,scene_image_counts=scene_counts)
            before = captured[('new','before')].clone()
            after = captured[('new','after')].clone()
            changed = pixels.clone()
            changed[first_view:scene_end] += 2
            batched(changed,grid,scene_image_counts=scene_counts)
            torch.testing.assert_close(captured[('new','before')][:first_view],before[:first_view],rtol=0,atol=0)
            assert not torch.equal(captured[('new','after')][:first_view],after[:first_view])
            torch.testing.assert_close(captured[('new','after')][scene_end:],after[scene_end:],rtol=0,atol=0)

        # A loss on scene 1's first view in the next ordinary ViT block must
        # backpropagate through fusion to view 2, but never into scene 2.
        batched.zero_grad(set_to_none=True)
        probe = pixels.clone().requires_grad_()
        batched(probe,grid,scene_image_counts=scene_counts)
        captured[('new','after')][:first_view].float().square().mean().backward()
        assert torch.isfinite(probe.grad).all()
        assert probe.grad[first_view:scene_end].abs().sum() > 0
        assert torch.count_nonzero(probe.grad[scene_end:]) == 0
        assert batched.interaction_stages['5'].o3.weight.grad.abs().sum() > 0
        print(f'BATCHING {device=} {mixed_sizes=} parameters={sum(p.numel() for p in batched.parameters())} '
              f'max_output_relative_l2={max(output_errors):.6g} gradient_relative_l2={gradient_error:.6g}')
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_batched_vision_fp32_residual_under_bf16_autocast():
    torch.manual_seed(79)
    original=tiny_model(production_geometry=True).model.visual.cuda().eval()
    batched=copy.deepcopy(original)
    batched.config.batch_vision_attention=True
    grid=torch.tensor([[1,4,4],[1,4,6],[1,4,4]],device='cuda')
    pixels=torch.randn(56,3*2*16*16,device='cuda',dtype=torch.float32)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        expected=original(pixels,grid,scene_image_counts=[2,1])
        actual=batched(pixels,grid,scene_image_counts=[2,1])
    for a,b in [(actual[0],expected[0]),*zip(actual[1],expected[1])]:
        assert_relative(a,b,.02,'autocast outputs')
    expected[0].float().square().mean().backward()
    actual[0].float().square().mean().backward()
    ga=[];gb=[]
    for (_,a),(_,b) in zip(original.named_parameters(),batched.named_parameters()):
        assert (a.grad is None)==(b.grad is None)
        if a.grad is not None:ga.append(a.grad.flatten());gb.append(b.grad.flatten())
    assert_relative(torch.cat(gb),torch.cat(ga),.03,'autocast gradients')


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_inference_batch_matches_separate_scenes_and_permutation(device):
    """Real inference mode: packed scenes retain boundaries and output order."""
    torch.manual_seed(83)
    dtype = torch.float32 if device == 'cpu' else torch.bfloat16
    cfg = copy.deepcopy(tiny_model(production_geometry=True).config.vision_config)
    cfg.depth = 24
    cfg.deepstack_visual_indexes = [5, 11, 17]
    cfg.dual_stream['interaction_indexes'] = [5, 11, 17, 23]
    cfg.batch_vision_attention = True
    model = Qwen3VLVisionModel(cfg).to(device=device, dtype=dtype).eval()
    # Three scenes, different numbers of views, interleaved size buckets.
    grids = [torch.tensor(rows, device=device) for rows in (
        [[1,4,4],[1,4,6]], [[1,4,4]], [[1,4,6],[1,4,4]])]
    counts = [len(g) for g in grids]
    pixels = [torch.randn(int(g.prod(-1).sum()), 3*2*16*16,
                          device=device, dtype=dtype) for g in grids]
    output_lengths = [len(p)//4 for p in pixels]
    tolerance = 1e-5 if device == 'cpu' else .02

    def forward(ps, gs, ns):
        main, deep = model(torch.cat(ps), torch.cat(gs), scene_image_counts=ns)
        return [main, *deep]

    with torch.inference_mode():
        batched = forward(pixels, grids, counts)
        separate = [forward([p], [g], [n]) for p,g,n in zip(pixels,grids,counts)]
        for level, actual in enumerate(batched):
            expected = torch.cat([scene[level] for scene in separate])
            assert_relative(actual, expected, tolerance, 'batched versus separate inference')
            assert not actual.requires_grad
        permutation = [2,0,1]
        permuted = forward([pixels[i] for i in permutation], [grids[i] for i in permutation],
                           [counts[i] for i in permutation])
        for actual, original in zip(permuted, batched):
            chunks = original.split(output_lengths)
            assert_relative(actual, torch.cat([chunks[i] for i in permutation]),
                            tolerance, 'scene permutation order')
        changed = [p.clone() for p in pixels]
        changed[0] += 2
        perturbed = forward(changed, grids, counts)
        for actual, original in zip(perturbed, batched):
            torch.testing.assert_close(actual[output_lengths[0]:], original[output_lengths[0]:],
                                       rtol=0, atol=0)
