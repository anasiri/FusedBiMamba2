# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from src.mamba_ssm.ops.triton_bi.layernorm_gated import RMSNorm as RMSNormGated
from src.mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from src.mamba_ssm.ops.triton_bi.ssd_combined_bi import mamba_chunk_scan_combined_bi, mamba_split_conv1d_scan_combined_bi
from src.mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=3,
        conv_init=None,
        expand=2,
        headdim=12,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=128,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        bimamba_type=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        assert bimamba_type in [None, "v0", "v1", "v1_shared", "fused"], f"Invalid bimamba_type: {bimamba_type}"
        self.bimamba_type = bimamba_type

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
        dim = self.nheads * self.headdim
        self.d_nonssm = (d_in_proj - 2 * dim - 2 * self.ngroups * self.d_state - self.nheads) // 2

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.bimamba_type == "v1":
            if self.process_group is None:
                self.in_proj_b = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
            else:
                self.in_proj_b = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                      process_group=self.process_group,
                                                      sequence_parallel=self.sequence_parallel,
                                                      **factory_kwargs)

            A_b = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
            A_b_log = torch.log(A_b).to(dtype=dtype)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            if self.conv_init is not None:
                nn.init.uniform_(self.conv1d_b.weight, -self.conv_init, self.conv_init)

            self.dt_bias_b = nn.Parameter(inv_dt)
            self.dt_bias_b._no_weight_decay = True
            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True


        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.bimamba_type!='fused' and self.use_mem_eff_path and inference_params is None:
        # if False and self.bimamba_type!='fused' and self.use_mem_eff_path and inference_params is None:
            if self.bimamba_type in ["v0", "v1_shared", "v1"]:
                out = mamba_split_conv1d_scan_combined(
                    zxbcdt.contiguous(),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.dt_bias,
                    A,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    chunk_size=self.chunk_size,
                    seq_idx=seq_idx,
                    activation=self.activation,
                    rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                    rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                    outproj_weight=None,
                    outproj_bias=None,
                    headdim=None if self.D_has_hdim else self.headdim,
                    ngroups=self.ngroups,
                    norm_before_gate=self.norm_before_gate,
                    **dt_limit_kwargs,
                )
                if self.bimamba_type == "v0":
                    out = out + out.flip([1])
                elif self.bimamba_type == "v1_shared":
                    out_b = mamba_split_conv1d_scan_combined(
                        zxbcdt.flip([1]).contiguous(),
                        # zxbcdt.contiguous(),
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias,
                        self.dt_bias,
                        A,
                        D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                        chunk_size=self.chunk_size,
                        seq_idx=seq_idx,
                        activation=self.activation,
                        rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                        rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                        outproj_weight=None,
                        outproj_bias=None,
                        headdim=None if self.D_has_hdim else self.headdim,
                        ngroups=self.ngroups,
                        norm_before_gate=self.norm_before_gate,
                        **dt_limit_kwargs,
                    )
                    out = out + out_b
                elif self.bimamba_type == "v1":
                    A_b = -torch.exp(self.A_b_log.float())
                    zxbcdt_b = self.in_proj_b(u.flip([1]))  # (B, L, d_in_proj) or (B * L, d_in_proj)
                    if seqlen_og is not None:
                        zxbcdt_b = rearrange(zxbcdt_b, "(b l) d -> b l d", l=seqlen)

                    out_b = mamba_split_conv1d_scan_combined(
                        zxbcdt_b.contiguous(),
                        rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                        self.conv1d_b.bias,
                        self.dt_bias_b,
                        A_b,
                        D=rearrange(self.D_b, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                        chunk_size=self.chunk_size,
                        seq_idx=seq_idx,
                        activation=self.activation,
                        rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                        rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                        outproj_weight=None,
                        outproj_bias=None,
                        headdim=None if self.D_has_hdim else self.headdim,
                        ngroups=self.ngroups,
                        norm_before_gate=self.norm_before_gate,
                        **dt_limit_kwargs,
                    )
                    out = out + out_b
            else:
                raise ValueError(f"Only Mamba type v0, v1, v1s, v2 are supported.")
            outproj_weight = self.out_proj.weight
            outproj_bias = self.out_proj.bias
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_gpu_dtype()
                out, outproj_weight = out.to(dtype), outproj_weight.to(dtype)
                outproj_bias = outproj_bias.to(dtype) if outproj_bias is not None else None
            out = F.linear(out, outproj_weight, outproj_bias)
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            if self.bimamba_type == "v1":
                d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
                z0, x0, z, xBC, dt = torch.split(
                    zxbcdt,
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                    dim=-1
                )
                if conv_state is not None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                assert self.activation in ["silu", "swish"]
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2).contiguous()
                )
                # This one is slower than the one above
                # xBC = self.act(
                #     self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                # )
                x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                      dim=-1)
                y = mamba_chunk_scan_combined(
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
                # backward
                A_b = -torch.exp(self.A_b_log.float())
                zxbcdt_b = self.in_proj_b(u.flip([1]))  # (B, L, d_in_proj) or (B * L, d_in_proj)
                if seqlen_og is not None:
                    zxbcdt_b = rearrange(zxbcdt_b, "(b l) d -> b l d", l=seqlen)

                d_mlp = (zxbcdt_b.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
                z0_b, x0_b, z_b, xBC_b, dt_b = torch.split(
                    zxbcdt_b,
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                    dim=-1
                )
                if conv_state is not None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC_b, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                assert self.activation in ["silu", "swish"]
                if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                    xBC_b = self.act(
                        self.conv1d(xBC_b.transpose(1, 2)).transpose(1, 2)
                    )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
                else:
                    xBC_b = causal_conv1d_fn(
                        xBC_b.transpose(1, 2),
                        rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                        bias=self.conv1d_b.bias,
                        activation=self.activation,
                    ).transpose(1, 2)
                x_b, B_b, C_b = torch.split(xBC_b,
                                            [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                            dim=-1)
                y_b = mamba_chunk_scan_combined(
                    rearrange(x_b, "b l (h p) -> b l h p", p=self.headdim),
                    dt_b,
                    A_b,
                    rearrange(B_b, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C_b, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D_b, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z_b, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias_b,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
                y = y + y_b.flip([1])
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b l h p -> b l (h p)")
                if self.rmsnorm:
                    y = self.norm(y, z)
                if d_mlp > 0:
                    y = torch.cat([F.silu(z0_b) * x0_b, y], dim=-1)
                if seqlen_og is not None:
                    y = rearrange(y, "b l d -> (b l) d")
            elif self.bimamba_type == "v1_shared":
                d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
                z0, x0, z, xBC, dt = torch.split(
                    zxbcdt,
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                    dim=-1
                )
                if conv_state is not None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                assert self.activation in ["silu", "swish"]
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2).contiguous()
                )
                # This one is slower than the one above
                # xBC = self.act(
                #     self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                # )
                x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                      dim=-1)
                y = mamba_chunk_scan_combined(
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
                y_b = mamba_chunk_scan_combined(
                    rearrange(x.flip([1]).contiguous(), "b l (h p) -> b l h p", p=self.headdim),
                    dt.flip([1]).contiguous(),
                    A,
                    rearrange(B.flip([1]).contiguous(), "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C.flip([1]).contiguous(), "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z.flip([1]).contiguous(), "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
                y = y + y_b.flip([1])
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b l h p -> b l (h p)")
                if self.rmsnorm:
                    y = self.norm(y, z)
                if d_mlp > 0:
                    y = torch.cat([F.silu(z0) * x0, y], dim=-1)
                if seqlen_og is not None:
                    y = rearrange(y, "b l d -> (b l) d")
            elif self.bimamba_type == "fused":
                d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
                z0, x0, z, xBC, dt = torch.split(
                    zxbcdt,
                    [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                    dim=-1
                )
                if conv_state is not None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2).contiguous()
                )
                x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state],
                                      dim=-1)
                y = mamba_chunk_scan_combined_bi(
                    rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                    dt,
                    A,
                    rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                    rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                    chunk_size=self.chunk_size,
                    D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                    z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                    dt_bias=self.dt_bias,
                    dt_softplus=True,
                    seq_idx=seq_idx,
                    **dt_limit_kwargs,
                    return_final_states=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b l h p -> b l (h p)")
                if self.rmsnorm:
                    y = self.norm(y, z)
                if d_mlp > 0:
                    y = torch.cat([F.silu(z0) * x0, y], dim=-1)
                if seqlen_og is not None:
                    y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.conv1d.weight.shape[0], self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.conv1d.weight.shape[0],
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
