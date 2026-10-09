# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL a8w8 blockscale bpreshuffle GEMM, layout-API kernel
(gemm_blockscale_preshuffle_layout.py -- TiledMma/TiledCopy/fx.gemm, a from-scratch
rewrite on FlyDSL's layout-algebra API, not the hand-rolled-addressing kernel
test_flydsl_blockscale_preshuffle_gemm.py covers).

Usage:
    python op_tests/flydsl_tests/test_flydsl_blockscale_preshuffle_gemm_layout.py
    pytest -q op_tests/flydsl_tests/test_flydsl_blockscale_preshuffle_gemm_layout.py
"""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)

_GFX = get_gfx()
if not (_GFX.startswith("gfx942") or _GFX.startswith("gfx95")):
    pytest.skip(
        f"blockscale bpreshuffle GEMM (layout API) needs gfx942/gfx95x, got {_GFX}",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.kernels.gemm_blockscale_preshuffle_layout import (
        compile_blockscale_preshuffle_gemm_layout,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_layout
    import flydsl.expr as fx
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL block-scale GEMM (layout API) kernel: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")

# Same tolerances as test_flydsl_blockscale_preshuffle_gemm.py: fp8 quantization on
# both operands dominates the error budget.
DEFAULT_REL_TOL = 2e-2
DEFAULT_TILE_REL_TOL = 1e-2
DEFAULT_INPUT_SEED = 20260401

SCALE_BLOCK_N = 128
SCALE_BLOCK_K = 128

# This kernel requires tile_k == scale_block_k == 128 (see module docstring), so
# every case uses the fixed 64x256x128/4-wave tile; the shape list below instead
# exercises the things THAT constraint still leaves free: ragged M, multiple K
# tiles (even/odd counts), multiple M/N tiles, fp16 output.
PRECISION_CASES = [
    {"name": "m1_n512_k512", "m": 1, "n": 512, "k": 512},
    {"name": "m33_n512_k512", "m": 33, "n": 512, "k": 512},
    {"name": "m100_n512_k512", "m": 100, "n": 512, "k": 512},
    {"name": "m64_n256_k128_single_tile", "m": 64, "n": 256, "k": 128},
    {"name": "m512_n512_k512", "m": 512, "n": 512, "k": 512},
    {"name": "m1024_n1024_k1024", "m": 1024, "n": 1024, "k": 1024},
    {"name": "m64_n512_k512_fp16", "m": 64, "n": 512, "k": 512, "out_dtype": torch.float16},
    {"name": "m513_n768_k896_ragged_odd_ktiles", "m": 513, "n": 768, "k": 896},
    {"name": "m1025_n1280_k640_ragged", "m": 1025, "n": 1280, "k": 640},
    {"name": "m2049_n1024_k1024_ragged", "m": 2049, "n": 1024, "k": 1024},
]


def make_inputs(m: int, n: int, k: int, *, seed: int = DEFAULT_INPUT_SEED):
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_weight

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    scale_n = (n + SCALE_BLOCK_N - 1) // SCALE_BLOCK_N
    scale_k = (k + SCALE_BLOCK_K - 1) // SCALE_BLOCK_K
    x = (torch.rand((m, k), generator=gen, device="cuda", dtype=torch.float32) / 10).to(
        dtypes.fp8
    )
    w = (torch.rand((n, k), generator=gen, device="cuda", dtype=torch.float32) / 10).to(
        dtypes.fp8
    )
    x_scale = torch.rand(
        (m, scale_k), generator=gen, device="cuda", dtype=torch.float32
    )
    w_scale = torch.rand(
        (scale_n, scale_k), generator=gen, device="cuda", dtype=torch.float32
    )
    w_shuffled = shuffle_weight(w, layout=(16, 16))
    x_scale_km = x_scale.transpose(0, 1).contiguous().view(m, scale_k)
    return x, w, x_scale, w_scale, w_shuffled, x_scale_km


def run_torch(x, w, x_scale, w_scale) -> torch.Tensor:
    """fp32 reference: dequantize both operands, then a plain matmul."""
    m, k = x.shape
    n = w.shape[0]
    scale_k = x_scale.shape[1]
    xd = (
        x.float().view(m, scale_k, SCALE_BLOCK_K) * x_scale.float().unsqueeze(-1)
    ).view(m, k)
    scale_n = w_scale.shape[0]
    wd = (
        w.float().view(scale_n, SCALE_BLOCK_N, scale_k, SCALE_BLOCK_K)
        * w_scale.float()[:, None, :, None]
    ).view(n, k)
    return xd @ wd.T


def rel_norm(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref_f = ref.float()
    return ((out.float() - ref_f).norm() / ref_f.norm().clamp_min(1e-30)).item()


def max_tile_rel(ref: torch.Tensor, out: torch.Tensor, tile: int = 16) -> float:
    r, o = ref.float(), out.float()
    m, n = r.shape
    pad = (0, (-n) % tile, 0, (-m) % tile)
    r = torch.nn.functional.pad(r, pad)
    o = torch.nn.functional.pad(o, pad)
    mm, nn = r.shape

    def to_tiles(t):
        return (
            t.view(mm // tile, tile, nn // tile, tile)
            .permute(0, 2, 1, 3)
            .reshape(-1, tile * tile)
        )

    rt, ot = to_tiles(r), to_tiles(o)
    rn = rt.norm(dim=1)
    real = rn > 0
    if not bool(real.any()):
        return 0.0
    floor = rn[real].mean() * 0.05
    denom = rn[real].clamp_min(floor).clamp_min(1e-30)
    return ((ot - rt).norm(dim=1)[real] / denom).max().item()


def run_precision_case(case: dict, *, rel_tol: float = DEFAULT_REL_TOL):
    m, n, k = case["m"], case["n"], case["k"]
    out_dtype = case.get("out_dtype", torch.bfloat16)
    print("=" * 80)
    print(
        f"[flydsl-layout] blockscale bpreshuffle case={case['name']} "
        f"shape=({m}, {n}, {k}) out={out_dtype}"
    )

    x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale)

    out = torch.zeros((m, n), dtype=out_dtype, device="cuda")
    gemm_a8w8_blockscale_bpreshuffle_layout(x, w_shuf, x_scale_km, w_scale, out)
    torch.cuda.synchronize()

    rel = rel_norm(ref, out)
    tile_rel = max_tile_rel(ref, out)
    passed = rel <= rel_tol and tile_rel <= DEFAULT_TILE_REL_TOL
    print(
        f"  rel={rel:.3e} (tol={rel_tol:.1e})  "
        f"max_tile_rel={tile_rel:.3e} (tol={DEFAULT_TILE_REL_TOL:.1e})"
        f"  --> {'PASS' if passed else 'FAIL'}"
    )
    return passed, rel, tile_rel


@pytest.mark.parametrize(
    "case", [pytest.param(c, id=c["name"]) for c in PRECISION_CASES]
)
def test_flydsl_blockscale_layout_precision(case: dict):
    passed, rel, tile_rel = run_precision_case(case)
    assert passed, (
        f"{case['name']}: rel={rel:.3e} (tol {DEFAULT_REL_TOL:.1e}), "
        f"max_tile_rel={tile_rel:.3e} (tol {DEFAULT_TILE_REL_TOL:.1e})"
    )


def test_one_compile_serves_every_m():
    """One compile must serve every M, same contract as the other kernel family.

    The occupancy knobs are pinned here so this tests only what it names: M
    itself is a runtime argument, not part of the compile key. (Their defaults
    are a shape-driven heuristic that CAN pick different values as M grows --
    covered by test_occupancy_cfg_heuristic below.)
    """
    n, k = 512, 512
    compile_blockscale_preshuffle_gemm_layout.cache_clear()
    for m in (64, 100, 1024, 2049):
        x, _, _, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
        out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
        gemm_a8w8_blockscale_bpreshuffle_layout(
            x, w_shuf, x_scale_km, w_scale, out, waves_per_eu=None,
            a_prefetch_depth=1, use_cshuffle_epilog=0,
        )
    info = compile_blockscale_preshuffle_gemm_layout.cache_info()
    assert info.misses == 1, f"expected exactly one compile miss, got {info}"


def test_occupancy_cfg_heuristic():
    """waves_per_eu, a_prefetch_depth and use_cshuffle_epilog are chosen off the
    tile, because the tile is what sets the register count and all three knobs
    are register trades.

    waves_per_eu=2 is only forced at 64x256, the tile on gfx942's 256-register
    cliff; a_prefetch_depth=2 only where there are spare registers to
    double-buffer A staging; CShuffle everywhere."""
    from aiter.ops.flydsl.gemm_kernels import _blockscale_layout_occupancy_cfg

    assert _blockscale_layout_occupancy_cfg(64, 256) == (2, 1, True)
    assert _blockscale_layout_occupancy_cfg(32, 64) == (2, 2, True)
    assert _blockscale_layout_occupancy_cfg(128, 128) == (None, 1, True)
    assert _blockscale_layout_occupancy_cfg(64, 128) == (None, 1, True)


def test_occupancy_knobs_are_numerically_neutral():
    """These knobs only move registers/latency/store-shape around, never results."""
    m, n, k = 1024, 1024, 1024
    x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale)
    for wpe in (None, 2):
        for depth in (1, 2):
            for cs in (0, 1):
                for swz in (0, 1):
                    out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
                    gemm_a8w8_blockscale_bpreshuffle_layout(
                        x, w_shuf, x_scale_km, w_scale, out,
                        waves_per_eu=wpe, a_prefetch_depth=depth,
                        use_cshuffle_epilog=cs, use_xcd_swizzle=swz,
                    )
                    torch.cuda.synchronize()
                    rel = rel_norm(ref, out)
                    assert rel <= DEFAULT_REL_TOL, (
                        f"wpe={wpe} depth={depth} cshuffle={cs} swz={swz}: "
                        f"rel={rel:.3e}"
                    )


def test_tile_and_swizzle_heuristics(monkeypatch):
    """Tile comes from grid size and K; the XCD swizzle follows the tile.

    The swizzle lifts the L2 hit rate on every shape measured, but 64x256 is
    the one candidate at 256 VGPR+AGPR and cannot afford its index math, so it
    is the sole opt-out."""
    import torch as _torch

    from aiter.ops.flydsl import gemm_kernels
    from aiter.ops.flydsl.gemm_kernels import (
        _blockscale_layout_swizzle_cfg,
        _blockscale_layout_tile_cfg,
        _blockscale_layout_wave_m_cfg,
    )

    # These are the gfx942 rules; gfx950 has its own (tested below).
    monkeypatch.setattr(gemm_kernels, "_blockscale_layout_is_gfx950", lambda: False)
    dev = _torch.device("cuda", 0)

    # Deep K with enough workgroups takes the wide tile; shallow K does not.
    assert _blockscale_layout_tile_cfg(8192, 1024, 8192, dev) == (64, 256)
    assert _blockscale_layout_tile_cfg(8192, 1024, 1024, dev) == (128, 128)
    # Grid-starved shapes shrink rather than leave CUs idle.
    assert _blockscale_layout_tile_cfg(128, 4096, 4096, dev) == (32, 64)
    # N that is not a multiple of 128 still yields a legal tile_n.
    tm, tn = _blockscale_layout_tile_cfg(1024, 1024 + 64, 1024, dev)
    assert (1024 + 64) % tn == 0, (tm, tn)

    assert _blockscale_layout_swizzle_cfg(64, 256) is False
    for tile in ((128, 128), (64, 128), (32, 64)):
        assert _blockscale_layout_swizzle_cfg(*tile) is True, tile

    # Only the 128-row tile has both the LDS traffic to save and the registers
    # to pay for it.
    assert _blockscale_layout_wave_m_cfg(128, 128) == 2
    for tile in ((64, 256), (64, 128), (32, 64)):
        assert _blockscale_layout_wave_m_cfg(*tile) == 1, tile


def test_gfx950_tile_and_scale_heuristics(monkeypatch):
    """gfx950 picks 32x64 below one wave of 64x128 workgroups, 64x256 only for
    one or two full waves with N >= 1024, else 64x128; the 64-row tiles stage
    their A-scales in LDS."""
    from aiter.ops.flydsl import gemm_kernels
    from aiter.ops.flydsl.gemm_kernels import (
        _blockscale_layout_scale_cfg,
        _blockscale_layout_tile_cfg_gfx950 as tile_cfg,
    )

    cu = 256
    assert tile_cfg(128, 4096, cu) == (32, 64)
    assert tile_cfg(1024, 1536, cu) == (32, 64)
    assert tile_cfg(4096, 512, cu) == (64, 128)
    assert tile_cfg(8192, 1024, cu) == (64, 256)
    assert tile_cfg(2048, 4096, cu) == (64, 256)
    # A partial last wave, a larger grid, or narrow N falls back to 64x128.
    assert tile_cfg(4096, 1536, cu) == (64, 128)
    assert tile_cfg(32768, 1024, cu) == (64, 128)
    assert tile_cfg(16384, 512, cu) == (64, 128)
    tm, tn = tile_cfg(1024, 1024 + 64, cu)
    assert (1024 + 64) % tn == 0, (tm, tn)

    monkeypatch.setattr(gemm_kernels, "_blockscale_layout_is_gfx950", lambda: True)
    assert _blockscale_layout_scale_cfg(32, 64) == {}
    assert _blockscale_layout_scale_cfg(64, 128) == {
        "stage_a_scales": True,
        "a_prefetch_depth": 2,
    }
    assert _blockscale_layout_scale_cfg(128, 128) == {
        "stage_a_scales": True,
        "use_async_copy": True,
    }
    monkeypatch.setattr(gemm_kernels, "_blockscale_layout_is_gfx950", lambda: False)
    assert _blockscale_layout_scale_cfg(64, 128) == {}


@pytest.mark.skipif(not _GFX.startswith("gfx95"), reason="K128 MFMA is gfx950-only")
@pytest.mark.parametrize(
    "tile_m,tile_n,extra",
    [
        (64, 128, {"a_prefetch_depth": 2}),
        (64, 128, {}),
        (128, 128, {"use_async_copy": True, "wave_m": 2}),
        (64, 256, {"a_prefetch_depth": 2}),
    ],
)
def test_stage_a_scales_matches_oracle(tile_m, tile_n, extra):
    """A-scales DMA'd into the A stage's LDS slot, across the sync (depth 1 and
    2) and async A paths, ragged M and an odd K-tile count."""
    for m, n, k in ((384, 1024, 1024), (513, 768, 896)):
        if n % tile_n:
            continue
        x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
        ref = run_torch(x, w, x_scale, w_scale)
        out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
        exe = compile_blockscale_preshuffle_gemm_layout(
            N=n, K=k, tile_m=tile_m, tile_n=tile_n, tile_k=128, out_dtype="bf16",
            num_waves=4, stage_a_scales=True, **extra,
        )
        semaphore = torch.empty(0, dtype=torch.int32, device="cuda")
        _run_compiled(
            exe, out, out, semaphore,
            x.contiguous(), w_shuf.contiguous(),
            x_scale_km.contiguous().view(-1), w_scale.contiguous().view(-1),
            m, n, fx.Stream(torch.cuda.current_stream()),
        )
        torch.cuda.synchronize()
        rel = rel_norm(ref, out)
        assert rel <= DEFAULT_REL_TOL, (
            f"{tile_m}x{tile_n} {extra} m={m} n={n} k={k}: rel={rel:.3e}"
        )


def test_direct_kernel_entrypoint_matches_oracle():
    """Exercise compile_blockscale_preshuffle_gemm_layout directly (bypassing the
    gemm_op_a8w8 dispatch wrapper), matching how op_tests/bench scripts call it."""
    m, n, k = 1024, 1024, 1024
    x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale)
    out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    exe = compile_blockscale_preshuffle_gemm_layout(
        N=n, K=k, tile_m=64, tile_n=256, tile_k=128, out_dtype="bf16", num_waves=4
    )
    semaphore = torch.empty(0, dtype=torch.int32, device="cuda")
    _run_compiled(
        exe,
        out,
        out,
        semaphore,
        x.contiguous(),
        w_shuf.contiguous(),
        x_scale_km.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()
    rel = rel_norm(ref, out)
    assert rel <= DEFAULT_REL_TOL, f"rel={rel:.3e} exceeds tol {DEFAULT_REL_TOL:.1e}"


@pytest.mark.parametrize("split_k", [2, 4, 8])
def test_split_k_matches_oracle(split_k: int):
    """split_k > 1 (in-kernel fp32 reduction) must match the split_k=1 result."""
    m, n, k = 1024, 1024, 1024
    x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale)
    tile_m, tile_n = 64, 256

    out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    workspace = torch.zeros(split_k * m * n, dtype=torch.float32, device="cuda")
    gx = (m + tile_m - 1) // tile_m
    gy = n // tile_n
    semaphore = torch.zeros(gx * gy, dtype=torch.int32, device="cuda")
    exe = compile_blockscale_preshuffle_gemm_layout(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=128,
        out_dtype="bf16",
        num_waves=4,
        split_k=split_k,
    )
    _run_compiled(
        exe,
        workspace,
        out,
        semaphore,
        x.contiguous(),
        w_shuf.contiguous(),
        x_scale_km.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    torch.cuda.synchronize()
    rel = rel_norm(ref, out)
    assert rel <= DEFAULT_REL_TOL, f"rel={rel:.3e} exceeds tol {DEFAULT_REL_TOL:.1e}"
    assert semaphore.sum().item() == 0, "semaphore must self-reset after the launch"


def test_split_k_via_op_wrapper():
    """split_k through the public op wrapper (gemm_op_a8w8), matching how a real
    caller would request it."""
    m, n, k = 8192, 1024, 1024
    x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
    ref = run_torch(x, w, x_scale, w_scale)
    out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    gemm_a8w8_blockscale_bpreshuffle_layout(
        x, w_shuf, x_scale_km, w_scale, out, split_k=4
    )
    torch.cuda.synchronize()
    rel = rel_norm(ref, out)
    assert rel <= DEFAULT_REL_TOL, f"rel={rel:.3e} exceeds tol {DEFAULT_REL_TOL:.1e}"


def test_async_copy_matches_oracle():
    """use_async_copy is default-off (it measured a loss on gfx942's 4-byte/lane
    DMA) but must stay correct: it is a live candidate for gfx950, where the
    16-byte form needs 4x fewer DMA calls. Covers the hand-rolled LDS swizzle
    on the DMA write side matching the swizzled sA read view."""
    for m, n, k in ((64, 256, 128), (1024, 1024, 1024), (513, 768, 896)):
        x, w, x_scale, w_scale, w_shuf, x_scale_km = make_inputs(m, n, k)
        ref = run_torch(x, w, x_scale, w_scale)
        out = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
        exe = compile_blockscale_preshuffle_gemm_layout(
            N=n, K=k, tile_m=64, tile_n=256, tile_k=128, out_dtype="bf16",
            num_waves=4, use_async_copy=True,
        )
        semaphore = torch.empty(0, dtype=torch.int32, device="cuda")
        _run_compiled(
            exe, out, out, semaphore,
            x.contiguous(), w_shuf.contiguous(),
            x_scale_km.contiguous().view(-1), w_scale.contiguous().view(-1),
            m, n, fx.Stream(torch.cuda.current_stream()),
        )
        torch.cuda.synchronize()
        rel = rel_norm(ref, out)
        assert rel <= DEFAULT_REL_TOL, f"async m={m} n={n} k={k}: rel={rel:.3e}"


def test_rejects_tile_k_not_equal_scale_block_k():
    with pytest.raises(ValueError, match="tile_k == scale_block_k"):
        compile_blockscale_preshuffle_gemm_layout(
            N=512, K=512, tile_m=64, tile_n=256, tile_k=256, scale_block_k=128
        )


if __name__ == "__main__":
    print(f"Running on {_GFX}")
    all_passed = True
    for case in PRECISION_CASES:
        passed, _, _ = run_precision_case(case)
        all_passed &= passed
    test_one_compile_serves_every_m()
    test_direct_kernel_entrypoint_matches_oracle()
    test_occupancy_cfg_heuristic()
    test_occupancy_knobs_are_numerically_neutral()
    test_async_copy_matches_oracle()
    for sk in (2, 4, 8):
        test_split_k_matches_oracle(sk)
    test_split_k_via_op_wrapper()
    test_rejects_tile_k_not_equal_scale_block_k()
    print("=" * 80)
    print("ALL PASSED" if all_passed else "SOME CASES FAILED")
