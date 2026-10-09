# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL GEMM APIs."""

from __future__ import annotations

import functools
import re

import flydsl.expr as fx
import torch
from torch import Tensor

from aiter import logger
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility.graph_alloc import persistent_alloc

from .gemm_a16w16_gfx1250 import gemm_a16w16 as gemm_a16w16_gfx1250
from .kernels.gemm_a16w16_gfx950 import (
    SPLIT_K_SEMAPHORE_MAX_LEN,
    gemm_a16w16,
)
from .kernels.tensor_shim import _run_compiled

# Tile candidates and the tuned-CSV kernelName format live in the tune-metadata
# module so the tuner that writes a row and this op that reads it back cannot
# drift. Order is part of the contract: kernelId indexes into it.
from .gemm_tune.flydsl_gemm_a8w8_blockscale_bpreshuffle_common import (
    default_dsrd_depth,
    default_use_async_copy,
    TILE_CANDIDATES as _BLOCKSCALE_TILE_CANDIDATES,
    tile_is_valid as blockscale_tile_is_valid,
)

__all__ = [
    "SPLIT_K_SEMAPHORE_MAX_LEN",
    "flydsl_hgemm",
    "flydsl_hgemm_kernel_name",
    "flydsl_preshuffle_gemm_a8",
    "get_flydsl_hgemm_kernel_params",
]


_HGEMM_KERNEL_RE = re.compile(
    r"^flydsl_hgemm_"
    r"a(?P<dtype>bf16|fp16|f16)_w(?P=dtype)"
    r"_(?P<out_dtype>bf16|fp16|f16|fp32)_"
    r"t(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)"
    r"x(?P<stages>\d+)_ks(?P<split_k>\d+)_"
    r"w(?P<m_waves>\d+)x(?P<n_waves>\d+)x(?P<k_waves>\d+)_"
    r"bias(?P<has_bias>[01])_ktail(?P<has_k_tail>[01])_"
    r"gm(?P<group_m>\d+)_p(?P<policy>ft|ht|hti)_"
    r"(?P<target_gfx>gfx[0-9a-z]+)$"
)


def _normalize_dtype_name(dtype: str | torch.dtype) -> str:
    if dtype in ("f16", "fp16", torch.float16):
        return "fp16"
    if dtype in ("bf16", torch.bfloat16):
        return "bf16"
    if dtype in ("f32", "fp32", torch.float32):
        return "fp32"
    raise ValueError(f"Unsupported FlyDSL HGEMM dtype: {dtype!r}")


def flydsl_hgemm_kernel_name(
    *,
    dtype: str | torch.dtype,
    out_dtype: str | torch.dtype,
    config: dict,
    has_bias: bool,
    target_gfx: str | None = None,
) -> str:
    """Build the stable kernel name persisted in tuned GEMM CSV files."""

    dtype_name = _normalize_dtype_name(dtype)
    out_dtype_name = _normalize_dtype_name(out_dtype)
    if dtype_name not in ("bf16", "fp16"):
        raise ValueError(f"Unsupported input dtype for HGEMM: {dtype_name}")
    if out_dtype_name not in (dtype_name, "fp32"):
        raise ValueError(
            f"Unsupported output dtype {out_dtype_name} for input {dtype_name}"
        )

    policy = "ht" if config["use_half_tile_interleaved"] else "ft"
    name = (
        f"flydsl_hgemm_a{dtype_name}_w{dtype_name}_{out_dtype_name}_"
        f"t{config['block_m']}x{config['block_n']}x{config['block_k']}"
        f"x{config['stages']}_ks{config['split_k']}_"
        f"w{config['m_waves']}x{config['n_waves']}x{config['k_waves']}_"
        f"bias{int(has_bias)}_ktail0_gm{config['group_m']}_p{policy}_"
        f"{target_gfx or get_gfx()}"
    )
    return name


def get_flydsl_hgemm_kernel_params(name: str) -> dict | None:
    """Parse a tuned FlyDSL HGEMM kernel name into the current config schema."""

    match = _HGEMM_KERNEL_RE.fullmatch(name)
    if match is None or match.group("has_k_tail") != "0":
        return None

    dtype = _normalize_dtype_name(match.group("dtype"))
    out_dtype = _normalize_dtype_name(match.group("out_dtype"))
    if out_dtype not in (dtype, "fp32"):
        return None

    return {
        "block_m": int(match.group("block_m")),
        "block_n": int(match.group("block_n")),
        "block_k": int(match.group("block_k")),
        "stages": int(match.group("stages")),
        "split_k": int(match.group("split_k")),
        "m_waves": int(match.group("m_waves")),
        "n_waves": int(match.group("n_waves")),
        "k_waves": int(match.group("k_waves")),
        "group_m": int(match.group("group_m")),
        "use_half_tile_interleaved": match.group("policy") in ("ht", "hti"),
        "has_bias": match.group("has_bias") == "1",
        "dtype": dtype,
        "out_dtype": out_dtype,
        "target_gfx": match.group("target_gfx"),
    }


def flydsl_hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    m_waves: int = 2,
    n_waves: int = 2,
    k_waves: int = 1,
    group_m: int = 0,
    policy: str = "ft",
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run the A16W16 kernel for this arch with AITER's ``B[N, K]`` convention."""

    if policy not in ("ft", "ht", "hti"):
        raise ValueError(f"Unsupported FlyDSL HGEMM policy: {policy!r}")
    launch_stream = (
        torch.cuda.current_stream(device=a.device) if stream is None else stream
    )
    if launch_stream.device != a.device:
        raise ValueError(f"`stream` must be on {a.device}, got {launch_stream.device}")

    gfx = get_gfx()
    if gfx == "gfx1250":
        if k_waves != 1:
            raise ValueError("The gfx1250 FlyDSL A16W16 kernel supports k_waves=1 only")
        with torch.cuda.stream(launch_stream):
            return gemm_a16w16_gfx1250(
                a,
                b,
                bias=bias,
                dtype=out_dtype or a.dtype,
                y=out,
                tile_m=block_m,
                tile_n=block_n,
                tile_k=block_k,
                m_warp=m_waves,
                n_warp=n_waves,
                num_buffers=stages,
                split_k=split_k,
                main_loop_unroll=policy in ("ht", "hti"),
            )
    if gfx != "gfx950":
        raise RuntimeError(
            "The FlyDSL A16W16 kernel currently supports gfx950 and gfx1250 only"
        )

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    user_kwargs = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": k_waves,
        "group_m": group_m,
        "use_half_tile_interleaved": policy in ("ht", "hti"),
    }
    return gemm_a16w16(
        a,
        b.t(),
        out=out,
        bias=bias,
        user_kwargs=user_kwargs,
        stream=launch_stream,
        layout="nt",
        out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# FlyDSL preshuffle GEMM kernel management
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_compile_fn():
    """Import the preshuffle compiler on first use."""
    from .kernels.preshuffle_gemm import compile_preshuffle_gemm

    logger.info("[FlyDSL] loaded preshuffle GEMM compiler")
    return compile_preshuffle_gemm


# Fixed size rather than one buffer per shape: a shape-keyed cache grows without
# limit and can evict a buffer a captured CUDA graph still points at. The bounds
# come from k_split_candidates, which keeps tile_count under CU_NUM and
# k_split * tile_count at four per CU.
# Mirrors preshuffle_gemm.PRESHUFFLE_M_MAX; duplicated to avoid importing the
# compiler module before the preshuffle path is selected.
PRESHUFFLE_M_MAX = 65536
PRESHUFFLE_FLAT_BUFFER_LIMIT_BYTES = 1 << 32

PRESHUFFLE_SPLIT_K_MAX_TILES = 256
PRESHUFFLE_SPLIT_K_MAX_TILE_ELEMS = 32 * 128
PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS = (
    4 * PRESHUFFLE_SPLIT_K_MAX_TILES * PRESHUFFLE_SPLIT_K_MAX_TILE_ELEMS
)


@functools.lru_cache(maxsize=128)
def _get_preshuffle_split_buffers(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[Tensor, Tensor]:
    # Safe to reuse: launches on a stream are ordered and the reduction hands
    # the semaphore back zeroed.
    with persistent_alloc(device):
        workspace = torch.empty(
            PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS, dtype=torch.float32, device=device
        )
        semaphore = torch.zeros(
            PRESHUFFLE_SPLIT_K_MAX_TILES, dtype=torch.int32, device=device
        )
    return workspace, semaphore


def _check_preshuffle_flat_buffer_capacity(
    m: int,
    n: int,
    k: int,
    a_elem_bytes: int,
    b_elem_bytes: int,
    out_elem_bytes: int,
) -> None:
    """Keep flat AMD buffer descriptors and their i32 offsets below 4 GiB."""
    buffer_bytes = {
        "A": m * k * a_elem_bytes,
        "B": n * k * b_elem_bytes,
        "output": m * n * out_elem_bytes,
    }
    for name, size in buffer_bytes.items():
        if size >= PRESHUFFLE_FLAT_BUFFER_LIMIT_BYTES:
            raise RuntimeError(
                f"[FlyDSL] preshuffle {name} buffer needs {size} bytes; "
                "flat buffer descriptors require fewer than 4 GiB"
            )


def _check_preshuffle_split_capacity(
    m: int, n: int, tile_m: int, tile_n: int, split_k: int
) -> None:
    tiles = ((m + tile_m - 1) // tile_m) * ((n + tile_n - 1) // tile_n)
    if tiles > PRESHUFFLE_SPLIT_K_MAX_TILES:
        raise RuntimeError(
            f"[FlyDSL] split_k needs {tiles} tile semaphores, "
            f"more than {PRESHUFFLE_SPLIT_K_MAX_TILES}"
        )
    elems = split_k * m * n
    if elems > PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS:
        raise RuntimeError(
            f"[FlyDSL] split_k needs a {elems}-element fp32 workspace, "
            f"more than {PRESHUFFLE_SPLIT_K_WORKSPACE_ELEMS}"
        )


def flydsl_preshuffle_gemm_a8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    use_async_copy: int = 0,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    lds_stage: int = 2,
    enable_scheduler: bool = True,
    split_k: int = 1,
) -> Tensor:
    """Compile and run FlyDSL preshuffle GEMM, optionally with fp32 split-K."""
    compile_fn = _get_compile_fn()
    from aiter.utility import dtypes

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0]

    if m > PRESHUFFLE_M_MAX:
        raise RuntimeError(
            f"[FlyDSL] M ({m}) exceeds {PRESHUFFLE_M_MAX}; the preshuffle kernel "
            f"views A and C through a layout bounded by that many rows."
        )
    if n % 16 != 0:
        raise RuntimeError(
            f"[FlyDSL] N ({n}) must be a multiple of 16 for preshuffled B."
        )
    if n % tile_n != 0 and split_k > 1:
        raise RuntimeError(
            f"[FlyDSL] ragged N ({n}) does not support split_k ({split_k})."
        )
    if split_k < 1 or k % split_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K ({k}) must be divisible by split_k ({split_k})."
        )
    if (k // split_k) % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL] K/split_k ({k // split_k}) is not a multiple of "
            f"tile_k ({tile_k}). "
            f"Arguments not supported! Skipping gemm!"
        )

    if XQ.dtype == dtypes.fp8:
        in_dtype = "fp8"
    elif XQ.dtype == torch.int8:
        in_dtype = "int8"
    else:
        raise ValueError(f"[FlyDSL] unsupported input dtype {XQ.dtype}")

    wpe = None if waves_per_eu <= 0 else waves_per_eu

    if Out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif Out.dtype == torch.float16:
        out_dtype = "fp16"
    else:
        raise ValueError(
            f"[FlyDSL] unsupported output dtype {Out.dtype}; "
            "expected torch.bfloat16 or torch.float16"
        )
    _check_preshuffle_flat_buffer_capacity(
        m,
        n,
        k,
        XQ.element_size(),
        WQ.element_size(),
        Out.element_size(),
    )

    exe = compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        use_async_copy=bool(use_async_copy),
        waves_per_eu=wpe,
        enable_scheduler=bool(enable_scheduler),
        xcd_swizzle=int(xcd_swizzle),
        lds_stage=int(lds_stage),
        split_k=int(split_k),
    )

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    out_contig = Out.contiguous()
    # FlyDSL's preshuffle kernel requires an arg_bias slot (used only when
    # epilogue != "none"). Pass an empty tensor as a placeholder for the
    # default epilogue="none" path.
    dummy_bias = torch.empty(0, dtype=Out.dtype, device=Out.device)
    if split_k > 1:
        _check_preshuffle_split_capacity(m, n, tile_m, tile_n, split_k)
        workspace, semaphore = _get_preshuffle_split_buffers(
            Out.device, torch.cuda.current_stream(device=Out.device)
        )
    else:
        workspace = out_contig
        # dtype is part of the executable's cache signature, so this must match
        # what the AOT pre-compile passes or every non-split-K kernel misses it.
        semaphore = torch.empty(0, dtype=torch.int32, device=Out.device)
    # The layout-API launcher takes fx.Tensor args, so pass flat tensors
    # directly rather than raw pointers.
    _run_compiled(
        exe,
        workspace.view(-1),
        out_contig.view(-1),
        semaphore,
        _as_i8(XQ.contiguous()).view(-1),
        _as_i8(WQ.contiguous()).view(-1),
        x_scale.contiguous().view(-1),
        w_scale.contiguous().view(-1),
        dummy_bias,
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not Out:
        Out.copy_(out_contig)

    return Out


# ---------------------------------------------------------------------------
# FlyDSL blockscale bpreshuffle GEMM kernel management
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_blockscale_compile_fn():
    """Import the block-scale compiler on first use."""
    from .kernels import gemm_blockscale_preshuffle as _bs

    logger.info("[FlyDSL] loaded blockscale bpreshuffle GEMM compiler")
    return _bs.compile_blockscale_preshuffle_gemm


@functools.lru_cache(maxsize=1024)
def _blockscale_tile_is_valid(
    tile_m, tile_n, tile_k, n, k, scale_block_k, use_cshuffle_epilog=False, num_waves=4
):
    """Cached tile_is_valid, which builds a real LDS plan and queries the device limit.

    Both flags are part of the key because both change the verdict: cshuffle roughly
    doubles the LDS footprint, and num_waves sets how tile_n must divide and how many
    bytes each thread copies.
    """
    return blockscale_tile_is_valid(
        tile_m,
        tile_n,
        tile_k,
        n,
        k,
        scale_block_k,
        num_waves=num_waves,
        use_cshuffle_epilog=use_cshuffle_epilog,
    )


@functools.lru_cache(maxsize=1024)
def select_blockscale_tile_config(
    m: int,
    n: int,
    k: int,
    scale_block_k: int = 128,
    use_cshuffle_epilog: bool = False,
    num_waves: int = 4,
) -> tuple:
    """Heuristic tile pick for shapes with no tuned row; prefer a tuned kernelName.

    Weights come from FlyDSL's select_tile_config (tests/kernels/
    test_blockscale_preshuffle_gemm.py at 950bed53, deleted by FlyDSL #966). The
    tile_n term below is the one aiter change.
    """
    valid = [
        t
        for t in _BLOCKSCALE_TILE_CANDIDATES
        if _blockscale_tile_is_valid(
            *t,
            n,
            k,
            scale_block_k,
            use_cshuffle_epilog,
            num_waves,
        )
    ]
    if not valid:
        return (64, 128, 128)
    wide_n = get_gfx().startswith("gfx942")

    def _score(t):
        tm, tn, tk = t
        s = 0
        total_blocks = ((m + tm - 1) // tm) * (n // tn)
        s += (
            15
            if total_blocks >= 256
            else (10 if total_blocks >= 128 else (5 if total_blocks >= 64 else 0))
        )
        if m <= 48:
            s += 12 if tm == 16 else (8 if tm == 32 else 0)
        elif m <= 128:
            s += 10 if tm == 32 else (6 if tm == 16 else (4 if tm == 64 else 0))
        elif m <= 512:
            s += 12 if tm == 64 else (8 if tm == 32 else 0)
        else:
            s += 12 if tm == 64 else 0
        if m <= 128:
            s += 6 if tn == 64 else (4 if tn == 128 else (2 if tn == 256 else 0))
        elif m <= 512:
            s += 8 if tn == 128 else (4 if tn == 64 else (4 if tn == 256 else 0))
        elif wide_n:
            # gfx942 measured faster with the wider tile once M is large; other
            # arches keep FlyDSL's preference.
            s += 8 if tn == 256 else (6 if tn == 128 else 2)
        else:
            s += 8 if tn == 128 else (4 if tn == 64 else (4 if tn == 256 else 0))
        s += 6 if tk == 128 else 3
        return s

    return max(valid, key=_score)


@functools.lru_cache(maxsize=1024)
def _compile_flydsl_blockscale(
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    scale_block_k: int,
    out_dtype: str,
    use_cshuffle_epilog: bool = False,
    dsrd_depth: int | None = None,
    use_async_copy: bool | None = None,
    num_waves: int = 4,
    stage_a_scales: bool = True,
):
    """Cached compile. M is not part of the key: the kernel takes it at runtime.

    Every compile flag is, though, since each one selects different code. A None
    dsrd_depth or use_async_copy resolves to the arch default before the call, so the
    key holds the value actually compiled rather than the request.

    waves_per_eu is deliberately not exposed. Leaving it unset lets the compiler
    choose, which measured best on both arches, and a high value collapses occupancy
    rather than trading it."""
    if use_async_copy is None:
        use_async_copy = default_use_async_copy()
    if dsrd_depth is None:
        dsrd_depth = default_dsrd_depth()
    compile_fn = _get_blockscale_compile_fn()
    return compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        out_dtype=out_dtype,
        use_cshuffle_epilog=use_cshuffle_epilog,
        dsrd_depth=dsrd_depth,
        use_async_copy=use_async_copy,
        num_waves=num_waves,
        stage_a_scales=stage_a_scales,
    )


def flydsl_gemm_a8w8_blockscale_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    tile_m: int = 0,
    tile_n: int = 0,
    tile_k: int = 0,
    scale_block_k: int = 128,
    use_cshuffle_epilog: bool = False,
    dsrd_depth: int | None = None,
    use_async_copy: bool | None = None,
    num_waves: int = 4,
    stage_a_scales: bool = True,
) -> Tensor:
    """Compile (cached) and run the FlyDSL blockscale bpreshuffle GEMM.

    Caller supplies the layouts, as gemm_a8w8_blockscale_bpreshuffle already requires:
    WQ preshuffled with shuffle_weight(w, layout=(16, 16)), and x_scale K-major.
    Reshuffling either per call would cost more than the kernel saves.

    Numerics are identical to gemm_a8w8_blockscale.
    """
    compile_fn = _get_blockscale_compile_fn()
    from aiter.utility import dtypes

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0] if WQ.dim() > 1 else w_scale.shape[0] * 128

    # fp8 bytes handed over as torch.uint8 are a supported input of this op: the CK
    # path accepts them and both triton branches normalise the same way. Raising here
    # would make the FlyDSL row the only one that rejects a caller every other backend
    # serves, on exactly the shapes the tuned row covers.
    if XQ.dtype == torch.uint8:
        XQ = XQ.view(dtypes.fp8)
    if WQ.dtype == torch.uint8:
        WQ = WQ.view(dtypes.fp8)
    if XQ.dtype != dtypes.fp8:
        raise RuntimeError(f"[FlyDSL] blockscale GEMM needs fp8 input, got {XQ.dtype}")
    if out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif out.dtype == torch.float16:
        out_dtype = "fp16"
    else:
        raise RuntimeError(
            f"[FlyDSL] unsupported output dtype {out.dtype}; "
            f"expected torch.bfloat16 or torch.float16"
        )

    if use_async_copy is None:
        use_async_copy = default_use_async_copy()
    if dsrd_depth is None:
        dsrd_depth = default_dsrd_depth()
    if not (tile_m and tile_n and tile_k):
        tile_m, tile_n, tile_k = select_blockscale_tile_config(
            m,
            n,
            k,
            scale_block_k,
            use_cshuffle_epilog,
            num_waves,
        )
    if not _blockscale_tile_is_valid(
        tile_m,
        tile_n,
        tile_k,
        n,
        k,
        scale_block_k,
        use_cshuffle_epilog,
        num_waves,
    ):
        raise RuntimeError(
            f"[FlyDSL] tile {tile_m}x{tile_n}x{tile_k} is invalid for N={n}, K={k}, "
            f"scale_block_k={scale_block_k}, num_waves={num_waves}. "
            f"Arguments not supported! Skipping gemm!"
        )

    # Keyword, not positional: the trailing arguments are four interchangeable
    # bool/int flags, so a reorder of either signature would still compile, still
    # run, and silently select a different kernel.
    exe = _compile_flydsl_blockscale(
        n,
        k,
        tile_m,
        tile_n,
        tile_k,
        scale_block_k,
        out_dtype,
        use_cshuffle_epilog=use_cshuffle_epilog,
        dsrd_depth=dsrd_depth,
        use_async_copy=use_async_copy,
        num_waves=num_waves,
        stage_a_scales=stage_a_scales,
    )

    # The kernel indexes scale_a as kb * M + row, so x_scale must reach it K-major.
    # A strided (m, scale_k) view over K-major bytes is a supported input of this op,
    # and .contiguous() would re-materialise it M-major, which is silently wrong rather
    # than an error. Transposing it instead is correct and copy-free.
    if x_scale.dim() == 2 and x_scale.stride(0) == 1 and x_scale.size(1) > 1:
        x_scale_flat = x_scale.t().contiguous().view(-1)
    else:
        x_scale_flat = x_scale.contiguous().view(-1)

    # Unlike flydsl_preshuffle_gemm_a8, this kernel's launcher takes fx.Tensor
    # arguments, so the tensors are handed over directly rather than as pointers.
    out_contig = out if out.is_contiguous() else out.contiguous()
    _run_compiled(
        exe,
        out_contig,
        XQ.contiguous(),
        WQ.contiguous(),
        x_scale_flat,
        w_scale.contiguous().view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not out:
        out.copy_(out_contig)

    return out


# ---------------------------------------------------------------------------
# FlyDSL blockscale bpreshuffle GEMM, layout-API kernel (gemm_blockscale_
# preshuffle_layout.py). A separate, from-scratch kernel on FlyDSL's
# layout-algebra API (TiledMma/TiledCopy/fx.gemm), not another hand-rolled-
# addressing variant -- see that file's module docstring. gfx942/gfx950 only,
# distinct from the gfx1250 MXFP8 family above. Not wired into the tuned-CSV
# auto-dispatch in gemm_op_a8w8.py yet (deliberately: no tuner/AOT support
# until this kernel has matched tests passing -- call it directly via
# gemm_a8w8_blockscale_bpreshuffle_layout in gemm_op_a8w8.py in the meantime).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _get_blockscale_layout_compile_fn():
    """Import the layout-API block-scale compiler on first use."""
    from .kernels import gemm_blockscale_preshuffle_layout as _bsl

    logger.info("[FlyDSL] loaded blockscale bpreshuffle GEMM (layout API) compiler")
    return _bsl.compile_blockscale_preshuffle_gemm_layout


# split_k is for the grid-starved small-M shapes this kernel's fixed 64x256
# tile under-fills 304 CUs on (see repo memory); the bounds below are sized
# for that regime, not for huge, already grid-saturated M.
BLOCKSCALE_SPLIT_K_MAX_TILES = 4096
BLOCKSCALE_SPLIT_K_WORKSPACE_ELEMS = 64 * 1024 * 1024  # fp32 elements (256 MiB)


@functools.lru_cache(maxsize=128)
def _get_blockscale_split_buffers(
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[Tensor, Tensor]:
    # Safe to reuse across calls: launches on a stream are ordered and the
    # reduction hands the semaphore back zeroed (see splitk_epilogue.py).
    with persistent_alloc(device):
        workspace = torch.empty(
            BLOCKSCALE_SPLIT_K_WORKSPACE_ELEMS, dtype=torch.float32, device=device
        )
        semaphore = torch.zeros(
            BLOCKSCALE_SPLIT_K_MAX_TILES, dtype=torch.int32, device=device
        )
    return workspace, semaphore


def _check_blockscale_split_capacity(
    m: int, n: int, tile_m: int, tile_n: int, split_k: int
) -> None:
    tiles = ((m + tile_m - 1) // tile_m) * ((n + tile_n - 1) // tile_n)
    if tiles > BLOCKSCALE_SPLIT_K_MAX_TILES:
        raise RuntimeError(
            f"[FlyDSL] split_k needs {tiles} tile semaphores, "
            f"more than {BLOCKSCALE_SPLIT_K_MAX_TILES}"
        )
    elems = split_k * m * n
    if elems > BLOCKSCALE_SPLIT_K_WORKSPACE_ELEMS:
        raise RuntimeError(
            f"[FlyDSL] split_k needs a {elems}-element fp32 workspace, "
            f"more than {BLOCKSCALE_SPLIT_K_WORKSPACE_ELEMS}"
        )


@functools.lru_cache(maxsize=8)
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _blockscale_layout_occupancy_cfg(
    tile_m: int, tile_n: int
) -> tuple[int | None, int, bool]:
    """Pick (waves_per_eu, a_prefetch_depth, use_cshuffle_epilog) from the tile.

    Keyed on the tile rather than the grid: the tile sets the register count,
    and every one of these knobs is really a register trade. (An earlier
    grid > CU rule was fitted when the kernel only ever ran 64x256 at 272
    registers; it mis-serves the 64-register tiles the tile heuristic now
    picks, costing 1.10x on deep_k alone.)

    waves_per_eu=2 is only worth forcing at 64x256, the one tile sitting on
    gfx942's 256-register 2-waves/SIMD cliff -- 177.5us against 206.6us there.
    Everywhere else the compiler already reaches that occupancy and forcing it
    is within noise (<=1%).

    a_prefetch_depth=2 needs ~16 spare registers to double-buffer the A staging
    fragment, which only the 32-high tile (64 registers) has; it pays there and
    costs 10-20% at 128x128, which has none.

    CShuffle is on for every tile. Its coalesced stores beat the MFMA-native
    layout in all but one measured cell, including the small grids an earlier
    grid > CU gate excluded (smallm256 1.06x).
    """
    if tile_m == 32:
        return 2, 2, True
    if (tile_m, tile_n) == _LAYOUT_TILE_DEEPK:
        return 2, 1, True
    return None, 1, True


def _blockscale_layout_wave_m_cfg(tile_m: int, tile_n: int) -> int:
    """How many of the 4 waves split M rather than N.

    Splitting waves across N only means every wave ds_reads all of A's
    m-fragments per k-step, so a 128-row tile issues twice the LDS traffic of a
    64-row one for the same MFMA count -- measured 2.50x SQ_LDS_IDX_ACTIVE and
    2.27x SQ_WAIT_INST_LDS against the hand-rolled kernel at 128x128, with
    MFMA counts identical and VMEM lower. Giving M two waves halves it.

    It is not free: each wave then covers twice the N columns, so its B
    fragment doubles (232 -> 256 registers at 128x128). That is affordable at
    128x128 and pays (m32768 1.07x, prefill 1.06x), break-even on the small
    tiles, and catastrophic at 64x256, which is already at 256 registers and
    spills 344 bytes for a 0.18x collapse.
    """
    return 2 if (tile_m, tile_n) == _LAYOUT_TILE_LARGE else 1


def _blockscale_layout_swizzle_cfg(tile_m: int, tile_n: int) -> bool:
    """Whether to remap workgroup ids for XCD L2 locality.

    On for every tile but 64x256. The map raises the measured L2 hit rate
    (TCC_HIT/(TCC_HIT+TCC_MISS)) everywhere it was tried -- 35.2 -> 79.3% at
    128x4096x4096, 50.7 -> 80.5% at 1024x8192x1024, 72.6 -> 81.4% at
    2048x4096x4096 -- cutting DRAM reads up to 4x, worth 1.03-1.26x once the
    occupancy knobs stopped forcing waves_per_eu=2 (under that forcing it
    looked like a loss at 128x128; it is not, and with waves_per_eu unset it
    actually allocates FEWER registers there, 240 against 248).

    64x256 remains the exception: at 256 registers it is already on gfx942's
    2-waves/SIMD cliff and the map's runtime div/mod pushes it over, a
    reproducible 0.80x despite its hit rate improving 81.4 -> 88.4%.
    """
    return (tile_m, tile_n) != _LAYOUT_TILE_DEEPK


# Both carry the same 16384-element tile, so they cost the same LDS and do the
# same work per workgroup; 128x128 is strictly the better shape of the two on
# gfx942 (240 VGPR+AGPR and no spill, against 64x256's 256 and 12 bytes of
# scratch) everywhere except deep K -- see _blockscale_layout_tile_cfg.
_LAYOUT_TILE_LARGE = (128, 128)
_LAYOUT_TILE_MID = (64, 128)
_LAYOUT_TILE_SMALL = (32, 64)
_LAYOUT_TILE_DEEPK = (64, 256)
# K at which the main loop dominates enough that the wider N tile's extra reuse
# beats 128x128's lower register pressure. Measured crossover: shapes at
# K <= 4096 prefer 128x128, K >= 8192 prefer 64x256.
_LAYOUT_DEEPK_MIN_K = 8192
# ...but only once 64x256 still makes enough workgroups to be worth it; below
# this the tile has to shrink regardless of K (deep_k, grid 64, wants 32x64).
_LAYOUT_DEEPK_MIN_GRID = 128
# Below this many 64x128 workgroups, halving tile_m to 32 buys more than the
# reuse it gives up. Measured boundary is between 168 (32-high wins 1.17x) and
# 256 (it loses 1.38x).
_LAYOUT_SHRINK_MAX_GRID = 192


def _blockscale_layout_tile_cfg(
    m: int, n: int, k: int, device: torch.device
) -> tuple[int, int]:
    """Pick (tile_m, tile_n) from the grid each candidate yields, plus K.

    Measured over a tile_m x tile_n sweep ({32,64,128} x {64,128,256}) on nine
    shapes and validated on three further disjoint sets (8 shapes each); the
    aggregate landed within 2.4% of the per-shape oracle. tile_m=16 and
    tile_k=256, which the hand-rolled kernel also tunes over, are not
    expressible here: 16 rows do not give every thread a whole 16-byte A load,
    and tile_k is pinned to scale_block_k by the block-scale accumulation.

    Four regimes, in priority order:

    Deep K with enough work: 64x256. The main loop dominates, so the wider N
    tile's reuse outweighs its 12 bytes of spill (m8192_deepk 1.06x and
    1024x4096x8192 1.10x over 128x128).

    Enough work for 128x128: take it. Same 16384-element tile as 64x256 but a
    better shape on gfx942 -- 240 VGPR+AGPR and no spill against 256 and 12
    bytes of scratch (m8192 1.03x, m32768 1.21x, prefill_m2048 1.05x over the
    old fixed 64x256).

    Too little work even for 64x128: 32x64, which trades reuse for roughly
    twice the workgroups (1.13-1.59x over 64x128 across 15 such shapes).

    Otherwise 64x128, which was within 8% of the best candidate everywhere in
    that middle band while no other single tile was.

    Known costs of keeping this to four rules: wide-N shapes at K <= 4096 and
    one workgroup per CU would rather have 64x256 and lose ~3% (wide_n,
    6144x1536x4096), and shapes whose M is an exact multiple of 64 just above
    the shrink boundary lose ~12% to the 32-high tile (320x2048x2048).
    """
    cu = _cu_count(device.index or 0)
    tm, tn = _LAYOUT_TILE_DEEPK
    if k >= _LAYOUT_DEEPK_MIN_K and n % tn == 0:
        if ((m + tm - 1) // tm) * (n // tn) >= _LAYOUT_DEEPK_MIN_GRID:
            return tm, tn
    tm, tn = _LAYOUT_TILE_LARGE
    if n % tn == 0 and ((m + tm - 1) // tm) * (n // tn) >= cu:
        return tm, tn
    tm, tn = _LAYOUT_TILE_MID
    if n % tn:
        return tm, 64
    if ((m + tm - 1) // tm) * (n // tn) < _LAYOUT_SHRINK_MAX_GRID:
        return _LAYOUT_TILE_SMALL
    return tm, tn


def flydsl_gemm_a8w8_blockscale_bpreshuffle_layout(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    out: Tensor,
    tile_m: int = 0,
    tile_n: int = 0,
    tile_k: int = 128,
    scale_block_k: int = 128,
    num_waves: int = 4,
    split_k: int = 1,
    waves_per_eu: int | None = -1,
    a_prefetch_depth: int = -1,
    use_cshuffle_epilog: int = -1,
    use_xcd_swizzle: int = -1,
    wave_m: int = -1,
) -> Tensor:
    """Compile (cached) and run the FlyDSL blockscale bpreshuffle GEMM, layout-API
    variant. Same calling convention, layouts, and numerics as
    flydsl_gemm_a8w8_blockscale_bpreshuffle (WQ preshuffled with
    shuffle_weight(w, layout=(16, 16)), x_scale K-major) -- this is a different
    kernel implementation, not a different op contract.

    tile_k must equal scale_block_k (128): see gemm_blockscale_preshuffle_layout.py's
    module docstring for why. split_k > 1 reduces fp32 partials in-kernel -- see
    that file's module docstring and splitk_epilogue.py; use it for grid-starved
    small-M shapes, not large/already grid-saturated ones.

    waves_per_eu and a_prefetch_depth default to a joint shape-driven heuristic
    (see _blockscale_layout_occupancy_cfg); pass explicit values to override.
    tile_m/tile_n default to 0, meaning pick by shape (see
    _blockscale_layout_tile_cfg); pass both to override. use_xcd_swizzle
    likewise defaults to a tile-driven choice (see
    _blockscale_layout_swizzle_cfg).
    """
    compile_fn = _get_blockscale_layout_compile_fn()
    from aiter.utility import dtypes

    m, k = XQ.shape[0], XQ.shape[-1]
    n = WQ.shape[0] if WQ.dim() > 1 else w_scale.shape[0] * 128

    if XQ.dtype == torch.uint8:
        XQ = XQ.view(dtypes.fp8)
    if WQ.dtype == torch.uint8:
        WQ = WQ.view(dtypes.fp8)
    if XQ.dtype != dtypes.fp8:
        raise RuntimeError(f"[FlyDSL] blockscale GEMM needs fp8 input, got {XQ.dtype}")
    if out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif out.dtype == torch.float16:
        out_dtype = "fp16"
    else:
        raise RuntimeError(
            f"[FlyDSL] unsupported output dtype {out.dtype}; "
            f"expected torch.bfloat16 or torch.float16"
        )

    if not (tile_m and tile_n):
        tile_m, tile_n = _blockscale_layout_tile_cfg(m, n, k, out.device)
    heur_wpe, heur_depth, heur_cs = _blockscale_layout_occupancy_cfg(tile_m, tile_n)
    # CShuffle aliases the A ring for its staging and has no split_k path.
    if split_k > 1:
        heur_cs = False
    heur_swz = _blockscale_layout_swizzle_cfg(tile_m, tile_n)
    heur_wm = _blockscale_layout_wave_m_cfg(tile_m, tile_n)
    exe = compile_fn(
        N=n,
        K=k,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        out_dtype=out_dtype,
        num_waves=num_waves,
        split_k=split_k,
        waves_per_eu=heur_wpe if waves_per_eu == -1 else waves_per_eu,
        a_prefetch_depth=heur_depth if a_prefetch_depth == -1 else a_prefetch_depth,
        use_cshuffle_epilog=(
            heur_cs if use_cshuffle_epilog == -1 else bool(use_cshuffle_epilog)
        ),
        use_xcd_swizzle=(
            heur_swz if use_xcd_swizzle == -1 else bool(use_xcd_swizzle)
        ),
        wave_m=heur_wm if wave_m == -1 else wave_m,
    )

    if x_scale.dim() == 2 and x_scale.stride(0) == 1 and x_scale.size(1) > 1:
        x_scale_flat = x_scale.t().contiguous().view(-1)
    else:
        x_scale_flat = x_scale.contiguous().view(-1)

    out_contig = out if out.is_contiguous() else out.contiguous()
    if split_k > 1:
        _check_blockscale_split_capacity(m, n, tile_m, tile_n, split_k)
        workspace, semaphore = _get_blockscale_split_buffers(
            out.device, torch.cuda.current_stream(device=out.device)
        )
    else:
        workspace = out_contig
        # dtype is part of the executable's cache signature, so this must match
        # what a split_k>1 compile passes or the split_k=1 kernel misses it.
        semaphore = torch.empty(0, dtype=torch.int32, device=out.device)
    _run_compiled(
        exe,
        workspace.view(-1),
        out_contig,
        semaphore,
        XQ.contiguous(),
        WQ.contiguous(),
        x_scale_flat,
        w_scale.contiguous().view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )
    if out_contig is not out:
        out.copy_(out_contig)

    return out
