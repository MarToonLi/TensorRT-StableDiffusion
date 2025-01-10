from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

# from ldm.modules.diffusionmodules.util import checkpoint


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention3(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        print("--q.shape: {}".format(q.shape))
        print("--k.shape: {}".format(k.shape))
        print("--v.shape: {}".format(v.shape))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttention2(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads; context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5; self.heads = heads
        self.to_qkv = nn.Linear(query_dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        with torch.no_grad():
            if query_dim == context_dim:
                q_weight = self.to_q.weight
                k_weight = self.to_k.weight
                v_weight = self.to_v.weight
                print("q_weight: ",q_weight.shape)
                print("k_weight: ",k_weight.shape)
                print("v_weight: ",v_weight.shape)
                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                print("qkv_weight: ",qkv_weight.shape)
                print("-----------")

                self.to_qkv.weight.copy_(qkv_weight)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, context=None, mask=None):
        print("x.shape: {}".format(x.shape))
        h = self.heads
        if context is None and torch.onnx.is_in_onnx_export():
            qkv = self.to_qkv(x)
            print("qkv.shape: {}".format(qkv.shape))
            q, k, v = torch.chunk(qkv, 3, dim=-1)

            print("--q.shape: {}".format(q.shape))
            print("--k.shape: {}".format(k.shape))
            print("--v.shape: {}".format(v.shape))

        else:
            q = self.to_q(x); context = default(context, x)
            k = self.to_k(context); v = self.to_v(context)

            print("++q.shape: {}".format(q.shape))
            print("++k.shape: {}".format(k.shape))
            print("++v.shape: {}".format(v.shape))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        sim = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttention4(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        context_dim_copy = context_dim

        inner_dim = dim_head * heads; context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5; self.heads = heads
        self.to_qkv = nn.Linear(query_dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.head_dim = dim_head

        print("context_dim: {};".format(context_dim))
        with torch.no_grad():
            if query_dim == context_dim:
                q_weight = self.to_q.weight
                k_weight = self.to_k.weight
                v_weight = self.to_v.weight

                print("q_weight: {}; k_weight: {}; v_weight: {};".format(q_weight.shape, k_weight.shape, v_weight.shape))

                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                print("qkv_weight: ",qkv_weight.shape)
                print("-----------")

                self.to_qkv.weight.copy_(qkv_weight)
            else:
                print("nothing to do!")
        print("==================================================")
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, context=None, mask=None):
        print("x.shape: {}".format(x.shape))

        batch_size, seq_length, embed_dim = x.size()
        mark = context.shape if context is not None else 0
        print("context: {}".format(mark))
        print("is_in_onnx_export: {}".format(torch.onnx.is_in_onnx_export()))

        h = self.heads
        # if context is None:
        if context is None:
            qkv = self.to_qkv(x)  # (batch_size, seq_length, embed_dim * 3)
            qkv = qkv.reshape(batch_size, seq_length, 3, self.heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)

            # 分离 Q, K, V
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = torch.squeeze(q, dim=0)
            k = torch.squeeze(k, dim=0)
            v = torch.squeeze(v, dim=0)

            print("--qkv.shape: {}; --q.shape: {}; --k.shape: {}; --v.shape: {}".format(qkv.shape, q.shape, k.shape, v.shape))

        else:
            q = self.to_q(x); context = default(context, x)
            k = self.to_k(context); v = self.to_v(context)

            # q_weight: torch.Size([512, 64])
            # k_weight: torch.Size([512, 64])
            # v_weight: torch.Size([512, 64])
            # qkv_weight: torch.Size([1536, 64])
            # -----------
            # x.shape: torch.Size([1, 10, 64])
            # ++q.shape: torch.Size([8, 10, 64])
            # ++k.shape: torch.Size([8, 15, 64])
            # ++v.shape: torch.Size([8, 15, 64])
            # output.shape: torch.Size([1, 10, 64])
            print("++q.shape: {}; ++k.shape: {}; ++v.shape: {}".format(q.shape, k.shape, v.shape))

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            print("++q.shape: {}; ++k.shape: {}; ++v.shape: {}".format(q.shape, k.shape, v.shape))

        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        del q, k
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
        sim = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # add
        if query_dim == context_dim:
            self.qkv_w = torch.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight]).transpose(0,1).detach()
        else:
            self.kv_w = torch.cat([self.to_k.weight, self.to_v.weight]).transpose(0,1).detach()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        if torch.onnx.is_in_onnx_export():
            if context is None:
                qkv = torch.matmul(x, self.qkv_w)
                q,k,v = qkv.chunk(3,dim=-1)
            else:
                q = self.to_q(x)
                kv = torch.matmul(context, self.kv_w)
                k, v = kv.chunk(2, dim=-1)
        else:
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return self._forward(x, context)
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

if __name__ == '__main__':
    # Example usage
    query_dim = 64
    context_dim = 64
    heads = 8
    dim_head = 64
    dropout = 0.1
    batch = 2

    # model = CrossAttention3(query_dim, context_dim, heads, dim_head, dropout)
    model = CrossAttention(query_dim, context_dim, heads, dim_head, dropout)

    x = torch.randn(batch, 10, query_dim)
    context = torch.randn(batch, 15, context_dim)
    mask = torch.ones(batch, 15).bool()

    output = model(x, context, mask)
    # output = model(x, None, None)
    print("output.shape: {}".format(output.shape))
