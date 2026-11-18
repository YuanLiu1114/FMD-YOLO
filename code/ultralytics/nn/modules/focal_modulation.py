import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class LayerNorm2d(nn.Module):
    """
    用 BatchNorm2d 替代 LayerNorm，推理最快（可与Conv融合）

    数学等价：GroupNorm(G=1) 对每个样本的所有C个通道做归一化
           与LayerNorm(normalized_shape=C) 在 channels-first 上等价
    
    速度优势：不需要permute / contiguous
    若训练中出现 loss 震荡，改回 GroupNorm 版本
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # self.norm = nn.BatchNorm2d(dim, eps=eps)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class FocalModulation(nn.Module):
    """
    focal modulation 核心
    """

    def __init__(
        self,
        dim: int,
        focal_levels: int = 3,
        focal_window: int = 3,
        focal_factor: int = 2,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.focal_levels = focal_levels
        self.use_sigmoid = use_sigmoid

        self.f = nn.Conv2d(
            dim, 
            2 * dim + (focal_levels + 1),
            kernel_size=1, 
            bias=True
        )

        self.focal_layers = nn.ModuleList()
        for lvl in range(focal_levels):
            k = focal_window + focal_factor * lvl
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=k, stride=1, padding=k//2, groups=dim, bias=False),
                    nn.GELU()
                )
            )
        
        self.h = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape

        xp = self.f(x)
        q, z, gates = torch.split(xp, [C, C, self.focal_levels + 1], dim=1)

        z = self.focal_layers[0](z)
        ctx = z * gates[:, :1]
        for lvl in range(1, self.focal_levels):
            z = self.focal_layers[lvl](z)
            ctx.add_(z * gates[:, lvl : lvl + 1])
        
        ctx_global = F.gelu(z.mean(dim=(2,3), keepdim=True))
        ctx.add_(ctx_global * gates[:, self.focal_levels :])
        
        modulator = self.h(ctx)
        if self.use_sigmoid:
            modulator = torch.sigmoid(modulator)
        
        out = q * modulator
        return self.proj(out)
    

class FocalModulationBlock(nn.Module):
    """
    LayerScale 保留：训练时稳定梯度，推理前用fold_layerscale()折叠消除
    """
    def __init__(
        self,
        dim: int,
        focal_levels: int = 3,
        focal_window: int = 3,
        mlp_ratio: float = 4.0,
        use_sigmoid: bool = False,
        init_gamma: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.modulation = FocalModulation(
            dim, focal_levels, focal_window, use_sigmoid=use_sigmoid
        )
        self.gamma1 = nn.Parameter(init_gamma * torch.ones(dim))

        self.use_ffn = mlp_ratio > 0
        if self.use_ffn:
            hidden = int(dim * mlp_ratio)
            self.norm2 = LayerNorm2d(dim)
            self.ffn = nn.Sequential(
                nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            )
            self.gamma2 = nn.Parameter(init_gamma * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1 = self.gamma1.view(1, -1, 1, 1)
        x = x + g1 * self.modulation(self.norm1(x))
        if self.use_ffn:
            g2 = self.gamma2.view(1, -1, 1, 1)
            x = x + g2 * self.ffn(self.norm2(x))
        return x
    

def fold_layerscale(model: nn.Module) -> nn.Module:
    """
    将 FocalModulationBlock 中已学习的 LayerScale gamma
    折叠进 modulation.proj(和ffn最后一层)的权重中，
    从而在推理时完全消除gamma乘法开销

    调用时机：训练结束、模型评估/导出之前。
    调用后不降低模型精度，提高推理速度

    使用用例：
    mdoel = YOLO("best").model
    fold_layerscale(model)
    model.eval()
    """
    for m in model.modules():
        if not isinstance(m, FocalModulationBlock):
            continue
        
        # 折叠 gamma1 -> modulation.proj
        with torch.no_grad():
            g1 = m.gamma1.data.view(-1, 1, 1, 1)
            m.modulation.proj.weight.data.mul_(g1)
            if m.modulation.proj.bias is not None:
                m.modulation.proj.bias.data.mul_(m.gamma1.data)
        
        # 用恒等 gamma(全1) 替换，forward 里乘1不改变结果
        m.gamma1.data.fill_(1.0)

        # 折叠gamma2 -> ffn最后一个conv
        if m.use_ffn and hasattr(m, 'gamma2'):
            with torch.no_grad():
                g2 = m.gamma2.data.view(-1, 1, 1, 1)
                last_conv = m.ffn[-1]
                last_conv.weight.data.mul_(g2)
                if last_conv.bias is not None:
                    last_conv.bias.data.mul_(m.gamma2.data)
            m.gamma2.data.fill_(1.0)
    
    return model


class SPPF_FM(nn.Module):
    def __init__(self, c1, c2, k=5, focal_levels=2, focal_window = 3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.fm = FocalModulationBlock(c2, focal_levels, focal_window, mlp_ratio=0.0)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.fm(self.cv2(torch.cat([x, y1, y2, y3], 1)))

class Bottleneck_FM(nn.Module):
    def __init__(self, c1, c2, shortcut=True, focal_levels=3, focal_window=3, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.fm = FocalModulationBlock(c_, focal_levels, focal_window, mlp_ratio=0.0)
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and (c1 == c2)
    
    def forward(self, x):
        out = self.cv2(self.fm(self.cv1(x)))
        return x + out if self.add else out


class C2f_FM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, focal_levels=3, focal_window=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_FM(self.c, self.c, shortcut, focal_levels, focal_window) for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c),1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


__all__ = (
    "LayerNorm2d",
    "FocalModulation",
    "FocalModulationBlock",
    "SPPF_FM",
    "Bottleneck_FM",
    "C2f_FM",
    "fold_layerscale",
)