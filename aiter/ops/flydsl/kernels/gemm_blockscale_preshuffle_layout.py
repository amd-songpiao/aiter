# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Blockscale preshuffle GEMM (layout API): FP8 MFMA, per-[1,128,128]-block scale.

A fourth variant alongside gemm_blockscale_preshuffle.py (2-buffer ping/pong,
hand-rolled addressing) and the two gemm_blockscale_preshuffle_multistage*.py
experiments (N-stage hand-rolled, no net win -- see their own module docstrings).
This one is a from-scratch rewrite on FlyDSL's layout-algebra API (TiledMma,
TiledCopy, fx.gemm, flat_divide, make_fragment_*), modeled directly on
preshuffle_gemm.py's ping-pong pipeline and swizzled-LDS view, NOT a revival of
the deleted manual kernel's addressing style.

Numerics: per-128x128 block scale (same contract as gemm_blockscale_preshuffle.py
-- ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128), so THIS FILE REQUIRES
tile_k == scale_block_k == 128: one main-loop tile is exactly one scale block,
which is what makes the block-scale accumulation tractable without an inner
sub-loop. Unlike preshuffle_gemm.py's row/col scale (constant across K, so it
is applied once in the epilogue), a per-K-block scale must be combined INSIDE
the K loop: each tile's MMA accumulates into a fresh, per-tile fragment (zeroed
every iteration) rather than preshuffle_gemm's single whole-K running
accumulator, then that tile's (A-scale x B-scale) product is FMA'd in software
into a persistent accumulator carried across iterations as device-loop state
(FlyDSL's ``for iv, state in range(...): ... yield [...]`` construct --
preshuffle_gemm.py already uses this for its own single-buffer path; reused
here for every path since every tile needs the carry, not just the last one).

Design notes (status as of this file's first working version):
- 4 waves, tile 64x256x128 by default (not the 8-wave HipKittens-style
  role-split ping/pong: FlyDSL has no `sched_valu` and no two-SIMD-group
  stagger primitive to express that pattern, so it is not attempted here --
  see gemm_blockscale_preshuffle_multistage.py's module docstring and repo
  memory for why a from-scratch 8-wave port was ruled out).
- B stays register-resident (VGPR), double-buffered across tiles, exactly
  like preshuffle_gemm.py -- never touches LDS (its preshuffled layout is
  already per-thread-disjoint, so there is nothing to redistribute via LDS;
  see gemm_blockscale_preshuffle_multistage_full.py's module docstring for the
  measured cost of staging it there anyway on gfx942).
- A LDS ring: 2 stages (ping/pong) in this first version, not yet the 3-stage
  ring with CShuffle alias the design sketch suggested -- correctness first;
  the ring depth and epilogue-buffer aliasing are a mechanical follow-up once
  this version is verified (widening to N stages following the same
  circular-buffer discipline documented in gemm_blockscale_preshuffle_
  multistage.py, which found no stage-count win on the hand-rolled kernel --
  worth re-checking on THIS one before assuming it carries over).
- Scale loads: per-tile, directly from global memory (not yet staged into the
  A LDS ring once per CTA, not yet SGPR-broadcast for B-scale). Marked as a
  follow-up optimization once correctness holds; see _load_tile_scales.
- split_k: in-kernel fp32-partial reduction, reusing splitk_epilogue.py's
  splitk_reduce_epilogue (same mechanism preshuffle_gemm.py uses) -- the last
  arriving split along grid.z reduces the fp32 partials and casts down to the
  final output dtype, in the same launch (no separate reduction kernel).
  Intended for the grid-starved small-M shapes repo memory identified (too
  few workgroups to fill the GPU at the selected tile); not
  useful (and not necessary) for already grid-saturated shapes.
- No bias/activation epilogue. preshuffle_gemm.py has those and they port over
  mechanically. Its XCD swizzle is implemented here too (use_xcd_swizzle) and
  is enabled by the caller-side heuristic for every tile but 64x256.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Float16,
    Float32,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Int32,
)
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.splitk_epilogue import CPOL_COHERENT, splitk_reduce_epilogue

SCALE_BLOCK = 128
NUM_WAVES = 4
# Row bound of the A/C layout views, mirroring preshuffle_gemm.py's own guard.
BLOCKSCALE_M_MAX = 65536
NUM_XCDS = 8  # MI300X/gfx942 accelerator complex dies, each with its own L2.
# An L2 locality group should cover a fixed number of output ROWS, not a fixed
# number of m-tiles: gemm_blockscale_preshuffle.py's 16 was picked against its
# 64-row tiles, and reusing it at tile_m=128 would double the A footprint per
# group and overflow the 4 MiB per-XCD L2.
XCD_GROUP_ROWS = 1024


def _minui(a, b):
    """Unsigned integer min on two index-typed scalars.

    FlyDSL 0.3 stopped unwrapping its Index/ArithValue wrappers at the dialect
    boundary ("Operand 1 ... must be a Value"), so hand the builder raw ir.Values
    and re-wrap the result for the index arithmetic that follows.
    """
    return ArithValue(arith.minui(_raw(a), _raw(b)))


def _xcd_tile_indices(c_num_pid_m, num_pid_n, group_m_tiles):
    """Map this workgroup's linear id to its (m-tile, n-tile) pair.

    Ported from gemm_blockscale_preshuffle.py. Consecutive ids go round-robin
    across the XCDs, each with its own L2, so a tile-linear order scatters
    neighbouring tiles over all of them. Undoing that rotation, then grouping
    group_m_tiles m-tile rows, keeps one XCD's tiles inside a single output
    patch so the B panel they share stays L2-resident.
    """
    wgid = gpu.block_id("x")
    c_num_pid_n = arith.index(num_pid_n)
    c_xcds = arith.index(NUM_XCDS)
    c_group_m = arith.index(group_m_tiles)
    num_wg = c_num_pid_m * c_num_pid_n

    # Undo the round-robin. XCD i owns the ids congruent to i mod NUM_XCDS, and when
    # num_wg is not a multiple of NUM_XCDS the first (num_wg % NUM_XCDS) of them own
    # one workgroup more than the rest. Their destination blocks therefore start
    # min(i, rem) further along; without that term the map stops being a permutation
    # and two workgroups claim one tile.
    xcd = wgid % c_xcds
    rotated = xcd * (num_wg // c_xcds) + _minui(xcd, num_wg % c_xcds) + wgid // c_xcds

    # Group group_m_tiles m-tile rows into one L2 locality patch. The final group is
    # short whenever num_pid_m is not a multiple of it, so it is folded on
    # its real height; folding on the nominal one would send m past the end of the grid.
    group_wgs = c_group_m * c_num_pid_n
    first_m = (rotated // group_wgs) * c_group_m
    group_m = _minui(c_num_pid_m - first_m, c_group_m)
    intra = rotated % group_wgs
    return first_m + (intra % group_m), intra // group_m


@functools.lru_cache(maxsize=1024)
def compile_blockscale_preshuffle_gemm_layout(
    *,
    N: int,
    K: int,
    tile_m: int = 64,
    tile_n: int = 256,
    tile_k: int = 128,
    scale_block_k: int = 128,
    out_dtype: str = "bf16",
    num_waves: int = NUM_WAVES,
    waves_per_eu: int | None = None,
    enable_scheduler: bool = True,
    split_k: int = 1,
    a_prefetch_depth: int = 1,
    use_async_copy: bool = False,
    lds_stages: int = 2,
    use_cshuffle_epilog: bool = False,
    use_xcd_swizzle: bool = False,
    wave_m: int = 1,
):
    """Compile the layout-API blockscale preshuffle GEMM. FP8 input, per-[1,128,128]
    block scale, bf16/fp16 output.
    Signature: fn(C, out, semaphore, A, B, scale_a, scale_b, M, N, stream).

    B must already be preshuffled with shuffle_weight(w, layout=(16, 16)) (same
    on-disk layout gemm_blockscale_preshuffle.py and preshuffle_gemm.py both read).
    scale_a must be K-major ([scale_k, M] flattened); scale_b is [scale_n, scale_k]
    row-major flattened -- same conventions as gemm_blockscale_preshuffle.py.

    For split_k > 1, C is an fp32 partial workspace sized split_k*M*N (caller's
    responsibility to allocate) and the last arriving split reduces into `out`
    (the real output tensor) inside this same launch -- see splitk_epilogue.py.

    a_prefetch_depth sets how far ahead A's GLOBAL load runs (the LDS ring stays
    2-stage either way): 1 closes the global->reg->LDS chain inside one tile, so
    the ds_write waits on that tile's own global load; 2 double-buffers the
    staging register so the load issued for tile t+2 is not consumed until the
    next iteration, giving it a full extra tile of MFMA to land. Costs one more
    A-staging register fragment. Ignored when use_async_copy is set.

    use_async_copy routes A straight from global to LDS with the
    buffer_load_lds DMA (same primitive preshuffle_gemm.py uses), skipping the
    staging register entirely -- no ds_write to stall on, and the staging
    fragment's registers are freed (272 -> 264). Correct but DEFAULT OFF: it
    measured a LOSS on gfx942 (e.g. deep_k 0.932 -> 0.865, m32768 0.799 ->
    0.720 of baseline) because CDNA3 only moves 4 bytes/lane, so A needs 8 DMA
    instructions per tile instead of 2 sixteen-byte loads, and that 4x
    memory-instruction blowup outweighs the saved register hop. Kept because
    CDNA4's 16-byte form would need only 2 calls, where the trade should
    reverse -- UNVERIFIED, no gfx950 available here.

    use_xcd_swizzle remaps a 1D workgroup id to (m-tile, n-tile) so one XCD's
    tiles stay inside one output patch and share an L2-resident B panel (the
    same map gemm_blockscale_preshuffle.py applies unconditionally). It raises
    the measured L2 hit rate (TCC_HIT/(TCC_HIT+TCC_MISS)) on every shape tried
    -- 35.2 -> 79.3%, 50.7 -> 80.5%, 72.6 -> 81.4% -- cutting DRAM reads up to
    4x, worth up to 2.5x where the starting locality was poor.

    Default off HERE because this entry point takes the tile as given. The
    index math costs ~32 registers, which only the small tiles can absorb
    without falling off gfx942's 256-register 2-waves/SIMD cliff, so the
    caller-side heuristic enables it for those alone -- see
    gemm_kernels._blockscale_layout_swizzle_cfg.
    """
    if tile_k != SCALE_BLOCK or scale_block_k != SCALE_BLOCK:
        raise ValueError(
            f"this layout-API kernel requires tile_k == scale_block_k == "
            f"{SCALE_BLOCK} (one main-loop tile is exactly one scale block); "
            f"got tile_k={tile_k}, scale_block_k={scale_block_k}"
        )
    if out_dtype not in ("fp16", "bf16"):
        raise ValueError(f"out_dtype must be 'fp16' or 'bf16', got {out_dtype!r}")
    if a_prefetch_depth not in (1, 2):
        raise ValueError(f"a_prefetch_depth must be 1 or 2, got {a_prefetch_depth}")
    if lds_stages not in (2, 3):
        raise ValueError(f"lds_stages must be 2 or 3, got {lds_stages}")
    if use_async_copy:
        # The DMA writes LDS directly, so there is no staging register to run
        # deeper on; normalise here so the loop/peel bounds stay consistent.
        a_prefetch_depth = 1
    if split_k < 1 or K % split_k != 0:
        raise ValueError(f"split_k must divide K; got split_k={split_k}, K={K}")
    split_k_extent = K // split_k
    if split_k_extent % tile_k != 0:
        raise ValueError(
            f"K/split_k ({split_k_extent}) must be a multiple of tile_k "
            f"({tile_k}); got split_k={split_k}, K={K}, tile_k={tile_k}"
        )
    if N % tile_n != 0:
        raise ValueError(f"N ({N}) must be divisible by tile_n ({tile_n})")
    if num_waves % wave_m:
        raise ValueError(f"wave_m ({wave_m}) must divide num_waves ({num_waves})")
    wave_n = num_waves // wave_m
    if tile_n % (wave_n * 16) != 0:
        raise ValueError(
            f"tile_n ({tile_n}) must split into whole 16-wide MFMA tiles across "
            f"{wave_n} n-waves"
        )
    if tile_m % (wave_m * 16) != 0:
        raise ValueError(
            f"tile_m ({tile_m}) must split into whole 16-high MFMA tiles across "
            f"{wave_m} m-waves"
        )

    gpu_arch = get_rocm_arch()
    is_gfx950 = str(gpu_arch).startswith("gfx95")
    if not (is_gfx950 or str(gpu_arch).startswith("gfx942")):
        raise ValueError(f"blockscale preshuffle GEMM needs gfx942/gfx95x, got {gpu_arch}")
    layout_elem = Float8E4M3FN if is_gfx950 else Float8E4M3FNUZ
    elem_bytes = 1
    final_out_elem_cls = BFloat16 if out_dtype == "bf16" else Float16
    out_elem_cls = Float32 if split_k > 1 else final_out_elem_cls
    out_elem_bytes = 4 if split_k > 1 else 2

    scale_k = K // scale_block_k
    scale_n = (N + 127) // 128  # ScaleBlockN=128, same as gemm_blockscale_preshuffle.py

    tile_K_perm = 64  # fp8 native MFMA grouping (two 32-wide MFMAs per fx.gemm call)
    k_iters = tile_k // tile_K_perm
    num_tiles = split_k_extent // tile_k  # tiles per split
    m_repeat = tile_m // (16 * wave_m)
    n_per_wave = tile_n // wave_n
    num_acc_n = n_per_wave // 16
    acc_size = m_repeat * num_acc_n * 4

    total_threads = num_waves * 64
    a_load_bytes = 16
    bytes_per_thread_a = (tile_m * tile_k * elem_bytes) // total_threads
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"tile_m*tile_k must be divisible by {total_threads * a_load_bytes}: "
            f"tile_m={tile_m}, tile_k={tile_k}, num_waves={num_waves}"
        )
    num_a_loads = bytes_per_thread_a // a_load_bytes
    num_b_loads = (tile_n * tile_k * elem_bytes) // total_threads // 16

    a_lds_elems = tile_m * tile_k

    # A's ring is `lds_stages` deep; B's register ring is always 2, so the main
    # loop must run lcm(lds_stages, 2) tiles per iteration to keep BOTH stage
    # indices compile-time constants (see the main loop's comment).
    tiles_per_iter = lds_stages if lds_stages % 2 == 0 else lds_stages * 2
    # ds_write may run up to lds_stages-1 tiles ahead (the stage it targets was
    # last read that many iterations back, so the per-iteration barrier covers
    # the WAR hazard); the global load runs one further per prefetch depth.
    a_write_ahead = lds_stages - 1
    a_load_ahead = a_write_ahead + (a_prefetch_depth - 1)

    # One contiguous ring, not one fx.Array per stage: the stages have to be
    # addressable by a runtime row index in the CShuffle epilogue, and separate
    # fields are separately allocated (a stage the main loop never touches is
    # also dead-code-eliminated, which silently shrinks LDS under any write that
    # crosses a field boundary).
    _a_fields = {"a": fx.Array[layout_elem, a_lds_elems * lds_stages, 16]}
    if split_k > 1:
        _a_fields["split_flag"] = fx.Array[Int32, 1, 4]
    SharedStorage = fx.struct(
        type("SharedStorage", (), {"__annotations__": _a_fields})
    )

    # CShuffle staging aliases the (by then dead) A ring, so it must fit in it;
    # the tile is walked in row chunks sized to that capacity rather than
    # enlarging LDS.
    # Always store 16B/lane: a narrower tile_n buys fewer n-lanes rather than a
    # shorter vector, since a 2-element store bitcasts to vector<1xi32> and that
    # fails to legalize ("Do not know how to scalarize this operator's operand").
    cs_e_vec = 4
    cs_nlane = min(32, tile_n // cs_e_vec)
    cs_mlane = total_threads // cs_nlane
    cs_rows_cap = (a_lds_elems * lds_stages * elem_bytes) // (tile_n * out_elem_bytes)
    cs_rows_per_mi = 16 * wave_m  # one fragment step covers every m-wave's block
    cs_chunk_rows = min(tile_m, cs_rows_cap)
    cs_mi_per_chunk = cs_chunk_rows // cs_rows_per_mi if cs_chunk_rows else 0
    cs_num_chunks = tile_m // cs_chunk_rows if cs_chunk_rows else 0
    if use_cshuffle_epilog:
        if split_k > 1:
            raise ValueError("use_cshuffle_epilog does not support split_k > 1")
        if (
            cs_chunk_rows < cs_rows_per_mi
            or cs_chunk_rows % cs_rows_per_mi
            or tile_m % cs_chunk_rows
            or cs_chunk_rows % cs_mlane
            or tile_n % (cs_nlane * cs_e_vec)
        ):
            raise ValueError(
                f"cshuffle geometry does not divide: tile_m={tile_m}, "
                f"tile_n={tile_n}, wave_m={wave_m}, rows_cap={cs_rows_cap}, "
                f"chunk_rows={cs_chunk_rows}"
            )

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_out: fx.Tensor,
        arg_semaphore: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        if const_expr(use_xcd_swizzle):
            # Swizzled launch is 1D in x; recover (m-tile, n-tile) from the linear id.
            rt_m = fx.Index(i32_m)
            c_num_pid_m = (rt_m + arith.index(tile_m - 1)) // arith.index(tile_m)
            sw_x, sw_y = _xcd_tile_indices(
                c_num_pid_m, N // tile_n, max(1, XCD_GROUP_ROWS // tile_m)
            )
            # Re-wrap: fx.block_idx hands out Int32-wrapped index values, and everything
            # downstream is typed against that.
            bid_x, bid_y = fx.Int32(sw_x), fx.Int32(sw_y)
        k_off = fx.Int32(bid_z) * num_tiles

        gA = fx.rocdl.make_buffer_tensor(
            arg_a,
            max_size=False,
            num_records_bytes=fx.Int64(i32_m) * fx.Int64(K) * fx.Int64(elem_bytes),
        )
        gB = fx.rocdl.make_buffer_tensor(arg_b)
        c_tensor = arg_c
        if const_expr(split_k > 1):
            c_split_offset = fx.Int64(bid_z) * fx.Int64(i32_m) * fx.Int64(N)
            c_tensor = fx.Tensor(
                fx.make_view(
                    fx.add_offset(fx.get_iter(arg_c), c_split_offset),
                    fx.make_layout((BLOCKSCALE_M_MAX, N), (N, 1)),
                )
            )
        gC = fx.rocdl.make_buffer_tensor(
            c_tensor,
            max_size=False,
            num_records_bytes=fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(out_elem_bytes),
        )

        tA = fx.flat_divide(gA, fx.make_tile(tile_m, tile_k))[None, None, bid_x, None]
        tB = fx.flat_divide(gB, fx.make_tile(tile_n, tile_k))[None, None, bid_y, None]
        tC = fx.flat_divide(gC, fx.make_tile(tile_m, tile_n))[None, None, bid_x, bid_y]

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)

        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_s2r = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        # Swizzled A LDS view -- identical construction to preshuffle_gemm.py's
        # is_8bit branch (verified-correct swizzle, not re-derived by hand here).
        k_blocks16 = (tile_k * elem_bytes) // 16
        swz_bits = k_blocks16.bit_length() - 1
        swz = fx.SwizzleType.get(swz_bits, 4, swz_bits)

        def _make_sA(ptr):
            return fx.make_view(
                ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_ordered_layout((tile_m, tile_k), (1, 0)),
                ),
            )

        lds_a_stage_ptrs = [
            lds.a.ptr if i == 0 else lds.a.ptr + fx.Int32(i * a_lds_elems)
            for i in range(lds_stages)
        ]
        sA_stages = [_make_sA(p) for p in lds_a_stage_ptrs]

        pA_g = thr_g2s.partition_S(tA)
        pA_s_stages = [thr_g2s.partition_D(s) for s in sA_stages]
        pA_s2r_stages = [thr_s2r.partition_S(s) for s in sA_stages]
        pB_g = thr_g2r_B.partition_S(tB)

        frag_copy_A_stages = [
            fx.make_fragment_like(pA_s_stages[0][None, None, None])
            for _ in range(0 if use_async_copy else a_prefetch_depth)
        ]
        frag_A = thr_mma.make_fragment_A(sA_stages[0])
        frag_B_single_layout = thr_mma.partition_B(tB).layout(None, None, None, 0)
        frag_B_stages = [
            fx.make_fragment_like(frag_B_single_layout, layout_elem.ir_type)
            for _ in range(2)
        ]
        frag_A_retile = thr_s2r.retile(frag_A)
        frag_B_retile_stages = [thr_g2r_B.retile(b) for b in frag_B_stages]

        acc_block = thr_mma.make_fragment_C(tC)
        out_cpol = CPOL_COHERENT if split_k > 1 else 0
        copy_op = fx.rocdl.BufferCopy32b if split_k > 1 else fx.rocdl.BufferCopy16b
        buf_copy_out = fx.make_copy_atom(copy_op(out_cpol), out_elem_cls)
        thr_r2g_C = fx.make_tiled_copy_C(buf_copy_out, tiled_mma).get_slice(tid)
        pC_g = thr_r2g_C.partition_S(tC)
        frag_C_out = fx.make_fragment_like(acc_block, out_elem_cls.ir_type)
        frag_C_retile = thr_r2g_C.retile(frag_C_out)

        acc_zero = Vec.filled(acc_size, 0.0, Float32)
        # Running (scale-FMA'd) accumulator: a persistent fragment matching
        # acc_block's (mi, ni) slice layout exactly (both index as
        # frag[None, mi, ni] -> native vector<4xf32>, confirmed via FlyDSL's
        # fragment slicing -- no scalar extract/insert needed anywhere in the
        # combine below, unlike an earlier version that `.load()`'d the WHOLE
        # acc_block into one flat 64-wide Vec and extracted/reassembled 64
        # scalars by hand). Measured via rocprofv3: that scalar-extraction
        # version issued 53.6% more VALU instructions than the hand-rolled
        # baseline kernel (which also combines scale per (mi,ni) at native
        # Vec4 granularity, never decomposing to scalars) -- see repo memory.
        # NOTE: the fragment's own rest-mode index ORDER is [None, mi, ni]
        # (mi first, ni second), the OPPOSITE of the flat-Vec `p = ni*16+
        # mi*4+ii` formula used elsewhere in this file for lane/column
        # mapping -- confirmed empirically (mi=0 correct, mi=1..3 wrong with
        # [None, ni, mi]; all mi correct with the order swapped). The flat
        # formula and this slice order describe the SAME underlying layout
        # from two different access paths; they don't have to agree on which
        # position is "first" in a multi-index tuple.
        frag_running = fx.make_fragment_like(acc_block, Float32.ir_type)

        # ── Per-tile block-scale MMA + scale combine, both at native (mi, ni)
        # Vec4 granularity: zero+MMA acc_block fresh per tile, then for every
        # (mi, ni) slice, FMA (acc_block_slice * s_a[mi] * s_b[ni]) directly
        # into the matching frag_running slice, in place. ──
        # Per-stripe MMA slicing (so each (mi,ni)'s FMA could issue right after
        # its own MFMAs) is NOT expressible here: fx.gemm with a `tiled_mma`
        # iterates the whole m_repeat x n_repeat tiling itself, so a single-atom
        # accumulator slice mismatches it (fails with "'ub.poison' op register
        # operand/result remain after rmem SSA promotion"). It would need the
        # single `mma_atom` plumbed in instead -- not pursued: this kernel is
        # memory-stall-bound, and fully interleaving the FMAs would anyway need
        # a sched_valu primitive FlyDSL does not have.
        def mma_kloop_block(a_stage, cur_frag_B, s_a, s_b):
            acc_block.store(acc_zero)
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_copy,
                    pA_s2r_stages[a_stage][None, None, ki],
                    frag_A_retile[None, None, ki],
                )
                fx.gemm(
                    tiled_mma,
                    acc_block,
                    frag_A[None, None, (None, ki)],
                    cur_frag_B[None, None, (None, ki)],
                    acc_block,
                )
            for ni in range_constexpr(num_acc_n):
                s_b_vec = Vec.filled(4, Float32(s_b[ni]), Float32)
                for mi in range_constexpr(m_repeat):
                    combined = s_a[mi] * s_b_vec
                    block_vec4 = Vec(acc_block[None, mi, ni].load())
                    prev_vec4 = Vec(frag_running[None, mi, ni].load())
                    fma_result = math_dialect.fma(block_vec4, combined, prev_vec4)
                    frag_running[None, mi, ni].store(Vec(fma_result, (4,), Float32))

        # ── Scale staging: load A-scale (tile_m f32s) ONCE per wave per tile
        # into LDS via a direct global->LDS DMA (same rocdl.raw_ptr_buffer_
        # load_lds primitive A's own ring uses), instead of every lane
        # re-reading its vec4 chunk from global every tile. All 4 waves issue
        # the DMA unconditionally and redundantly (each computes the exact
        # same 64 rows via the hardware's automatic per-lane 0..63 fan-out) --
        # deliberately NOT gated behind an `if`: an earlier version only let
        # wave 0 do it (`if tid < tile_m: ...`), which measured a ~20x
        # regression on deep-K shapes (num_tiles=64) because the lane-
        # divergent `if tid < num_b_scale:` B-scale branch broke the
        # instruction scheduler's cross-iteration vmem/mfma interleaving for
        # the whole loop body, not just the branch itself. B-scale is tiny
        # (num_b_scale values) and was never the bottleneck, so it stays a
        # plain per-lane global read every tile (uniform address across all
        # lanes already -- the hardware/compiler treat it as a broadcast).
        #
        # NOTE: staging A-scale into LDS via a per-wave global->LDS DMA (even
        # branch-free, redundant across all 4 waves) was tried and measured
        # WORSE than the plain per-lane global read below: deep_k (K=8192,
        # num_tiles=64) regressed from 68us to 131us (0.52x) and wide_n
        # regressed to 0.81x. Root cause: the staged read is gated behind the
        # loop's existing gpu.barrier(), so the DMA's full global-memory
        # round-trip latency sits on the critical path every iteration,
        # whereas the direct per-lane buffer_load below is interleaved with
        # MFMA by hot_loop_scheduler()'s sched_vmem/sched_dsrd hints (hidden
        # latency, no barrier dependency). This is the third operand-staging
        # variant this session to backfire on gfx942 for this kernel family
        # (see also: A-only multistage, full A+B+scale staging) -- kept here,
        # reverted, as a documented dead end; do not re-attempt without a
        # fundamentally different (non-barrier-gated) delivery mechanism.
        wave_id = tid // 64
        m_wave = wave_id // wave_n
        n_wave = wave_id % wave_n
        lane_id = tid % 64
        lane_div_16 = lane_id // 16
        lane_mod_16 = lane_id % 16
        bx_m = bid_x * tile_m
        by_n = bid_y * tile_n
        row_off_base = lane_div_16 * 4

        scale_a_rsrc = buffer_ops.create_buffer_resource(
            arg_scale_a,
            max_size=False,
            num_records_bytes=fx.Int64(scale_k) * fx.Int64(i32_m) * fx.Int64(4),
        )
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        def _read_sa_vec4(row):
            s_a_vec = buffer_ops.buffer_load(scale_a_rsrc, row, vec_width=4, dtype=T.f32)
            return Vec(s_a_vec).bitcast(Float32)

        def _read_sb_scalar(idx):
            # B-scale is wave-uniform (for a fixed ni the lane's column spans one
            # 16-aligned 16-wide block, which cannot straddle a 128-wide B-scale
            # block), so rocdl.readfirstlane would legally force it to an SGPR.
            # Measured and REJECTED: it does free registers (272 -> 264, and
            # removes the 20B spill at waves_per_eu=2) but is SLOWER on every
            # shape (0.81-0.98x) -- the v_readfirstlane ops add cross-lane
            # dependency stalls worth more than the registers, and SGPR_Count
            # did not even change. Keep the plain per-lane load.
            return buffer_ops.buffer_load(scale_b_rsrc, idx, vec_width=1, dtype=T.f32)

        def load_tile_scales(k_tile):
            kb = fx.Int32(k_tile)
            s_a = [
                _read_sa_vec4(
                    kb * fx.Int32(i32_m)
                    + bx_m
                    + (mi * wave_m + m_wave) * 16
                    + row_off_base
                )
                for mi in range_constexpr(m_repeat)
            ]
            # Column assignment per lane for accumulator slot `ni` is
            # (ni * wave_n + n_wave) * 16 + lane_mod_16 -- the TiledMma wave
            # layout interleaves waves across N rather than giving each wave one
            # contiguous n_per_wave block (verified against preshuffle_gemm.py's
            # own epilogue, which reads scale/bias with this exact formula), and
            # the same interleaving applies down M once wave_m > 1.
            s_b = [
                _read_sb_scalar(
                    (
                        (by_n + (ni * wave_n + n_wave) * 16 + lane_mod_16)
                        // fx.Int32(128)
                    )
                    * fx.Int32(scale_k)
                    + kb
                )
                for ni in range_constexpr(num_acc_n)
            ]
            return s_a, s_b

        # ── Scheduling hints: interleave the per-16x16 MFMA stripes with the
        # next tile's copies, matching preshuffle_gemm.py's own gfx942 branch
        # (no sched_valu, no wave-role stagger -- see module docstring). ──
        def hot_loop_scheduler():
            mfma_group = num_acc_n
            mfma_total = (k_iters * 2) * m_repeat * mfma_group
            mfma_per_iter = 2 * mfma_group
            sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
            rocdl.sched_dsrd(2)
            rocdl.sched_mfma(1)
            rocdl.sched_mfma(1)
            if const_expr(num_acc_n < 4):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                rocdl.sched_mfma(1)
            dswr_tail = num_a_loads
            if const_expr(dswr_tail > sche_iters):
                dswr_tail = sche_iters
            dswr_start = sche_iters - dswr_tail
            for sche_i in range_constexpr(sche_iters):
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(mfma_group)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(mfma_group)
                if const_expr(sche_i >= dswr_start - 1):
                    rocdl.sched_dswr(1)
            rocdl.sched_barrier(0)

        # ── Optional async gmem->LDS DMA for A (skips the staging register) ──
        # Addressing is hand-rolled rather than going through TiledCopy: the DMA
        # writes a wave-uniform LDS base and the hardware fans lanes out across
        # it, so the swizzle the sA view applies on the read side has to be
        # reproduced here on the write side. Same construction (and same
        # k ^ ((m % k_blocks16) * elems_per_16b) formula) as preshuffle_gemm.py.
        if const_expr(use_async_copy):
            # Uses the raw rocdl intrinsic rather than a BufferCopyLDS copy
            # atom: the atom form does not legalize on gfx942 at either width
            # (checked), whereas this is exactly what the hand-rolled kernel
            # issues. CDNA3 moves 4 bytes/lane; CDNA4 added the 16-byte form.
            dma_bytes = 16 if is_gfx950 else 4
            num_dma_loads = bytes_per_thread_a // dma_bytes
            a_rsrc = buffer_ops.create_buffer_resource(
                arg_a,
                max_size=False,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(K) * fx.Int64(elem_bytes),
            )
            lds_a_ptrs = lds_a_stage_ptrs
            k_blocks16_dma = (tile_k * elem_bytes) // 16
            lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")

            def dma_a_to_lds(k_tile_val, stage):
                # The LDS address is wave-uniform and the hardware fans lane L
                # out to +L*dma_bytes, so the swizzle the sA read view applies
                # has to be folded into the GLOBAL address instead.
                base_k = (k_off + k_tile_val) * tile_k
                lds_i64 = None
                for i in range_constexpr(num_dma_loads):
                    pos_bytes = i * total_threads * dma_bytes + tid * dma_bytes
                    m_idx = pos_bytes // tile_k
                    k_idx = pos_bytes % tile_k
                    k_swz = k_idx ^ ((m_idx % k_blocks16_dma) * 16)
                    global_byte = (bx_m + m_idx) * K + base_k + k_swz
                    if const_expr(i == 0):
                        lds_addr = fx.Int64(
                            fx.ptrtoint(lds_a_ptrs[stage])
                        ) + fx.Int64(wave_id * 64 * dma_bytes)
                        lds_i64 = rocdl.readfirstlane(T.i64, lds_addr)
                    else:
                        lds_i64 = lds_i64 + total_threads * dma_bytes
                    rocdl.raw_ptr_buffer_load_lds(
                        a_rsrc,
                        llvm.inttoptr(lds_ptr_ty, lds_i64),
                        fx.Int32(dma_bytes),
                        fx.Int32(global_byte),
                        fx.Int32(0),
                        fx.Int32(0),
                        fx.Int32(1),
                    )

        # ── Prologue ──────────────────────────────────────────────────────
        # Fill the a_write_ahead stages the loop expects to already be live
        # (the loop's own ds_write targets tile t+a_write_ahead), and for
        # prefetch depth 2 also pre-load the tile the loop's first ds_write
        # will consume, so iteration 0 never waits on its own global load.
        if const_expr(use_async_copy):
            for s in range_constexpr(a_write_ahead):
                if const_expr(s < num_tiles):
                    dma_a_to_lds(fx.Int32(s), s)
            fx.copy(buf_copy, pB_g[None, None, None, k_off], frag_B_retile_stages[0])
            frag_running.store(acc_zero)
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
        else:
            for s in range_constexpr(a_write_ahead):
                if const_expr(s < num_tiles):
                    fx.copy(
                        buf_copy,
                        pA_g[None, None, None, k_off + fx.Int32(s)],
                        frag_copy_A_stages[0],
                    )
                    fx.copy(
                        uni_copy,
                        frag_copy_A_stages[0],
                        pA_s_stages[s][None, None, None],
                    )
            if const_expr(a_prefetch_depth == 2 and a_write_ahead < num_tiles):
                fx.copy(
                    buf_copy,
                    pA_g[None, None, None, k_off + fx.Int32(a_write_ahead)],
                    frag_copy_A_stages[1],
                )
            fx.copy(buf_copy, pB_g[None, None, None, k_off], frag_B_retile_stages[0])
            frag_running.store(acc_zero)
            gpu.barrier()
        rocdl.sched_barrier(0)

        # ── Main loop: 2 tiles per device-loop iteration, ping-pong A/B,
        # mirroring preshuffle_gemm.py's lds_stage==2 pipeline_2stage/two_tiles
        # pattern exactly (same proven technique, adapted for a per-tile scale
        # FMA instead of a single whole-K accumulator). Stage selection inside
        # pipeline_2stage/two_tiles is always a HARDCODED 0-then-1 sequence,
        # never the traced device-loop variable `iv` -- that's what lets this
        # be a REAL scf.for (register-reusing) loop despite FlyDSL tracing
        # `iv` as a non-Python-int ArithValue: `iv` only ever appears inside
        # arithmetic (k_off + iv*2 + ...), never as a list index. Replaces the
        # previous range_constexpr(num_tiles) full Python unroll, which forced
        # the backend to allocate num_tiles independent copies of the running
        # accumulator and spilled to scratch past num_tiles=8 (see repo memory).
        #
        # frag_running is mutated in place by mma_kloop_block's combine step
        # (per-(ni,mi) slice store), so pipeline_2stage/two_tiles need no
        # explicit accumulator argument/return -- only the device for-loop's
        # state/yield still threads frag_running's whole-fragment value
        # across real loop iterations (required for scf.for's SSA form, same
        # as preshuffle_gemm.py's frag_C -- storage is the same object either
        # way, this is just what makes the loop-carried dependency explicit).
        def pipeline_tile(pos, cur_k_val, cur_tile, last_tile):
            """One tile. `pos` is the COMPILE-TIME position modulo
            tiles_per_iter, used to derive both ring indices; `cur_tile`/
            `last_tile` decide which prefetches are in range; `cur_k_val` is the
            (possibly traced) tile index actually addressed."""
            read_stage = pos % lds_stages
            a_write_stage = (pos + a_write_ahead) % lds_stages
            b_write_stage = (pos + 1) % 2
            do_a_load = (cur_tile + a_load_ahead) <= last_tile
            do_a_write = (cur_tile + a_write_ahead) <= last_tile
            do_b_load = (cur_tile + 1) <= last_tile
            if const_expr(use_async_copy):
                # One hop: global -> LDS. Nothing to ds_write, so nothing for
                # the next tile's LDS fill to stall on mid-iteration.
                #
                # Scales are issued FIRST here -- the opposite of the sync path
                # below, where doing so cost register pressure. vmcnt retires in
                # order, so if the DMA were issued first the combine's wait on
                # the scales would imply waiting on the DMA too, making the copy
                # async in name only.
                s_a, s_b = load_tile_scales(k_off + cur_k_val)
                if const_expr(do_a_write):
                    dma_a_to_lds(cur_k_val + fx.Int32(a_write_ahead), a_write_stage)
                if const_expr(do_b_load):
                    fx.copy(
                        buf_copy,
                        pB_g[None, None, None, k_off + cur_k_val + fx.Int32(1)],
                        frag_B_retile_stages[b_write_stage],
                    )
                mma_kloop_block(read_stage, frag_B_stages[pos % 2], s_a, s_b)
                if const_expr(enable_scheduler):
                    hot_loop_scheduler()
                if const_expr(do_a_write):
                    # Drain the A DMAs (B's loads, issued after them, may stay
                    # in flight) before publishing the stage to the other waves.
                    rocdl.s_waitcnt(num_b_loads)
                    gpu.barrier()
                return
            if const_expr(do_a_load):
                fx.copy(
                    buf_copy,
                    pA_g[None, None, None, k_off + cur_k_val + a_load_ahead],
                    frag_copy_A_stages[pos % 2 if a_prefetch_depth == 2 else 0],
                )
            if const_expr(do_b_load):
                fx.copy(
                    buf_copy,
                    pB_g[None, None, None, k_off + cur_k_val + fx.Int32(1)],
                    frag_B_retile_stages[b_write_stage],
                )
            # Scales are deliberately loaded AFTER the next tile's A/B
            # prefetches, not before: issuing them first (so they'd return
            # earlier) was measured WORSE -- it extends their live range across
            # the prefetch copies, pushing scratch spill from 20 to 44 bytes and
            # GPU time from 24.2us to 30.4us at M=8192. Register pressure beats
            # issue order for this kernel; do not "optimize" this reorder again.
            s_a, s_b = load_tile_scales(k_off + cur_k_val)
            mma_kloop_block(read_stage, frag_B_stages[pos % 2], s_a, s_b)
            if const_expr(do_a_write):
                fx.copy(
                    uni_copy,
                    frag_copy_A_stages[(pos + 1) % 2 if a_prefetch_depth == 2 else 0],
                    pA_s_stages[a_write_stage][None, None, None],
                )
            if const_expr(enable_scheduler):
                hot_loop_scheduler()
            if const_expr(do_a_write):
                gpu.barrier()

        # Inside the device loop every tile is "interior" (all prefetches in
        # range), so last_tile is passed as a large sentinel; the peeled tail
        # below passes real positions so its prefetches compile out at the end.
        INTERIOR = num_tiles

        def loop_body(k_base):
            for j in range_constexpr(tiles_per_iter):
                pipeline_tile(j, k_base + fx.Int32(j), 0, INTERIOR)

        # The loop may only cover tiles whose deepest prefetch is still in
        # range, so the peel grows with both the ring depth and prefetch depth.
        last_tile = num_tiles - 1
        loop_end = max(0, num_tiles - a_load_ahead) // tiles_per_iter

        if const_expr(loop_end > 0):
            for iv, state in range(0, loop_end, 1, init=[frag_running.load()]):
                frag_running.store(state[0])
                loop_body(fx.Int32(iv * tiles_per_iter))
                results = yield [frag_running.load()]
            frag_running.store(results)
        for t in range_constexpr(loop_end * tiles_per_iter, num_tiles):
            pipeline_tile(t % tiles_per_iter, fx.Int32(t), t, last_tile)

        # ── Epilogue ──────────────────────────────────────────────────────
        # Direct store writes the MFMA's native C layout, where a lane owns 4
        # rows of one column -- i.e. 4 separate out_elem_bytes stores at stride
        # N. CShuffle instead bounces the tile through LDS so each lane can
        # store e_vec contiguous elements, making 32 lanes cover a contiguous
        # run. It reuses (aliases) the A ring, which is dead by now, and walks
        # the tile in row chunks sized to that ring so LDS does NOT grow --
        # growing it would put 2 workgroups/CU at exactly the 64 KiB limit and
        # risk the occupancy win.
        if const_expr(use_cshuffle_epilog):
            c_rsrc = buffer_ops.create_buffer_resource(
                c_tensor,
                max_size=False,
                num_records_bytes=(
                    fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(out_elem_bytes)
                ),
            )
            lds_out = fx.recast_iter(out_elem_cls, lds.a.ptr)
            vec_out_ty = T.vec(cs_e_vec, out_elem_cls.ir_type)
            m_lane = tid // cs_nlane
            n_lane = tid % cs_nlane

            def cs_idx(row, col):
                # Rows sit tile_n elements apart, always a whole number of
                # 32-bank lines, so unswizzled every row would start on bank 0
                # and the lanes covering different rows would collide. XOR keeps
                # cs_e_vec alignment (both operands are multiples of it), so the
                # read side stays a contiguous vector load.
                return row * tile_n + (col ^ ((row % 8) * cs_e_vec))

            for c in range_constexpr(cs_num_chunks):
                # Publishes this chunk's writes, and on later chunks also waits
                # for every lane to finish reading the previous one.
                gpu.barrier()
                for mi_local in range_constexpr(cs_mi_per_chunk):
                    mi = c * cs_mi_per_chunk + mi_local
                    for ni in range_constexpr(num_acc_n):
                        col = (ni * wave_n + n_wave) * 16 + lane_mod_16
                        acc4 = Vec(frag_running[None, mi, ni].load())
                        for ii in range_constexpr(4):
                            row = (
                                (mi_local * wave_m + m_wave) * 16
                                + lane_div_16 * 4
                                + ii
                            )
                            fx.ptr_store(
                                acc4[ii].to(out_elem_cls),
                                lds_out + cs_idx(row, col),
                            )
                gpu.barrier()
                for mr in range_constexpr(cs_chunk_rows // cs_mlane):
                    row_in_chunk = (mr * cs_mlane) + m_lane
                    row_g = bx_m + (c * cs_chunk_rows) + (mr * cs_mlane) + m_lane
                    for nr in range_constexpr(tile_n // (cs_nlane * cs_e_vec)):
                        col = nr * (cs_nlane * cs_e_vec) + n_lane * cs_e_vec
                        frag = fx.ptr_load(
                            lds_out + cs_idx(row_in_chunk, col),
                            result_type=vec_out_ty,
                        )
                        byte_off = (
                            row_g * fx.Int32(N) + by_n + col
                        ) * fx.Int32(out_elem_bytes)
                        buffer_ops.buffer_store(
                            Vec(frag).bitcast(fx.Int32),
                            c_rsrc,
                            byte_off,
                            offset_is_bytes=True,
                        )
        else:
            final_acc = Vec(frag_running.load())
            frag_C_out.store(Vec(final_acc).to(out_elem_cls))
            fx.copy(buf_copy_out, frag_C_retile, pC_g)
        if const_expr(split_k > 1):
            # The store above carries sc0|sc1 (out_cpol), so waiting on it is
            # the whole cross-XCD release -- no extra agent-scope fence needed.
            rocdl.s_waitcnt(0)
            gpu.barrier()
            splitk_reduce_epilogue(
                arg_c,
                fx.Tensor(
                    fx.make_view(
                        fx.get_iter(arg_out),
                        fx.make_layout((BLOCKSCALE_M_MAX, N), (N, 1)),
                    )
                ),
                arg_semaphore,
                lds.split_flag.ptr,
                tile_m,
                tile_n,
                total_threads,
                final_out_elem_cls,
                tid,
                bid_x,
                bid_y,
                i32_m,
                N,
                split_k,
            )

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_out: fx.Tensor,
        arg_semaphore: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
        k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        wave_layout = fx.make_layout((wave_m, wave_n, 1), (wave_n, 1, 0))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, wave_layout, fx.make_tile(None, None, k_perm)
        )

        val_per_thr = a_load_bytes // elem_bytes
        thrs_k = tile_k // val_per_thr
        thrs_m = total_threads // thrs_k
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem),
            fx.make_layout(
                ((thrs_k, thrs_m), (1, val_per_thr)),
                ((thrs_m * val_per_thr, 1), (1, thrs_m)),
            ),
            fx.make_tile(thrs_m, tile_k),
        )

        # Preshuffled B layout (2D hierarchical) -- same construction as
        # preshuffle_gemm.py's launcher, reading the same shuffle_weight(16,16)
        # on-disk format.
        kp_elems = 16  # fp8: 1 byte/elem, 16B kpack
        k_bytes_b = K * elem_bytes
        n0 = N // 16
        k0 = k_bytes_b // 64
        s_nlane = kp_elems
        s_klane = 16 * s_nlane
        s_k0 = 4 * s_klane
        s_n0 = k0 * s_k0
        preshuffle_B = fx.Tensor(
            fx.make_view(
                fx.get_iter(arg_b),
                fx.make_layout(
                    ((16, n0), (kp_elems, 4, k0)), ((s_nlane, s_n0), (1, s_klane, s_k0))
                ),
            )
        )

        arg_a_2d = fx.Tensor(
            fx.make_view(
                fx.get_iter(arg_a), fx.make_layout((BLOCKSCALE_M_MAX, K), (K, 1))
            )
        )
        arg_c_2d = fx.Tensor(
            fx.make_view(
                fx.get_iter(arg_c), fx.make_layout((BLOCKSCALE_M_MAX, N), (N, 1))
            )
        )

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        kernel_gemm(
            arg_c_2d,
            arg_out,
            arg_semaphore,
            arg_a_2d,
            preshuffle_B,
            arg_scale_a,
            arg_scale_b,
            i32_m,
            i32_n,
            tiled_mma,
            tiled_copy_g2s,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(
            grid=(gx * gy, 1, split_k) if use_xcd_swizzle else (gx, gy, split_k),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm


__all__ = ["compile_blockscale_preshuffle_gemm_layout"]
