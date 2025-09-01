# moo_pinn_gradnorm.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# -----------------------
# 0) 基本設定與工具函式
# -----------------------
torch.manual_seed(42)
np.random.seed(42)

def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32)).to(device)

def true_solution(x: np.ndarray) -> np.ndarray:
    # u*(x) = sin(pi x)
    return np.sin(np.pi * x).astype(np.float32)

@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_pde: int = 256          # 每步 PDE 采樣點數
    n_bc_each: int = 64       # 每端邊界點數
    iters: int = 6000
    lr_model: float = 1e-3
    lr_weights: float = 1e-3
    alpha: float = 0.5        # GradNorm 超參數
    log_every: int = 200


# -----------------------
# 1) PINN 網路（簡單 FNN）
# -----------------------
class FNN(nn.Module):
    def __init__(self, in_dim: int = 1, hidden: int = 64, depth: int = 3, out_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------
# 2) PDE 殘差與損失
# -----------------------
def pde_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ -u_xx - pi^2 sin(pi x) = 0  => residual = -u_xx - source """
    x.requires_grad_(True)
    u = model(x)                              # (N,1)
    du_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0]
    source = (math.pi ** 2) * torch.sin(math.pi * x)
    res = -d2u_dx2 - source
    return res

def bc_loss(model: nn.Module, x_bc: torch.Tensor) -> torch.Tensor:
    u = model(x_bc)
    return torch.mean(u ** 2)


# -----------------------
# 3) GradNorm 主程式
# -----------------------
def train_gradnorm(config: TrainConfig) -> Tuple[FNN, dict]:
    dev = torch.device(config.device)
    model = FNN().to(dev)

    # 兩組優化器：一個給模型參數、一個給可訓練的 loss 權重
    opt_model = torch.optim.Adam(model.parameters(), lr=config.lr_model)

    # w = [w_pde, w_bc]，初始化為 1.0，並可學習
    loss_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True, device=dev)
    opt_w = torch.optim.Adam([loss_weights], lr=config.lr_weights)

    # 記錄用
    hist = {
        "L_pde": [],
        "L_bc": [],
        "L_total": [],
        "W_pde": [],
        "W_bc": [],
        "G_pde": [],
        "G_bc": [],
    }

    # 預存初始 L0 作為「相對進度」基準
    x_pde0 = to_tensor(np.random.rand(config.n_pde, 1) * 2 - 1, dev)  # [-1,1]
    r0 = pde_residual(model, x_pde0)
    L0_pde = torch.mean(r0 ** 2).detach()

    x_bc0 = to_tensor(np.array([[-1.0], [1.0]]).repeat(config.n_bc_each, axis=0), dev)
    L0_bc = bc_loss(model, x_bc0).detach()

    L0 = torch.stack([L0_pde, L0_bc]) + 1e-12  # 防 0

    # 訓練迴圈
    for it in range(1, config.iters + 1):
        model.train()
        opt_model.zero_grad()
        opt_w.zero_grad()

        # ---- 個別 loss ----
        # PDE collocation
        x_pde = to_tensor(np.random.rand(config.n_pde, 1) * 2 - 1, dev)
        r = pde_residual(model, x_pde)
        L_pde = torch.mean(r ** 2)

        # BC points
        x_bc_left = to_tensor(-np.ones((config.n_bc_each, 1)), dev)
        x_bc_right = to_tensor(np.ones((config.n_bc_each, 1)), dev)
        x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
        L_bc = bc_loss(model, x_bc)

        losses = torch.stack([L_pde, L_bc])

        # ---- 總損並對模型做第一次反傳 ----
        weighted = (losses * loss_weights).sum()
        # 保留圖給 GradNorm 權重更新使用
        weighted.backward(retain_graph=True)

        # ---- 計算最後一層權重的梯度範數 G_i ----
        last_W = model.net[-1].weight  # 最後一層
        G_pde = torch.norm(torch.autograd.grad(loss_weights[0] * L_pde, last_W, retain_graph=True)[0])
        G_bc = torch.norm(torch.autograd.grad(loss_weights[1] * L_bc, last_W, retain_graph=True)[0])
        G = torch.stack([G_pde, G_bc])
        G_avg = torch.mean(G)

        # ---- 相對進度 r_i = (L_i / L0_i)^alpha ----
        L_ratio = (losses.detach() / L0).clamp_min(1e-12)
        target = G_avg * (L_ratio ** config.alpha)

        # ---- GradNorm 權重損失與更新（只更新 w_i）----
        gradnorm_loss = torch.sum(torch.abs(G - target))
        gradnorm_loss.backward()        # 只對 loss_weights 產生梯度
        opt_w.step()

        # ---- 權重重正規化：sum(w_i) = num_tasks ----
        with torch.no_grad():
            loss_weights.data = loss_weights.data * (2.0 / loss_weights.data.sum().clamp_min(1e-12))

        # ---- 更新模型參數 ----
        opt_model.step()

        # ---- 紀錄 ----
        hist["L_pde"].append(L_pde.item())
        hist["L_bc"].append(L_bc.item())
        hist["L_total"].append(weighted.item())
        hist["W_pde"].append(loss_weights[0].item())
        hist["W_bc"].append(loss_weights[1].item())
        hist["G_pde"].append(G_pde.item())
        hist["G_bc"].append(G_bc.item())

        if it % config.log_every == 0:
            print(f"[{it:5d}] "
                  f"Total={weighted.item():.3e} | "
                  f"L_pde={L_pde.item():.3e}, L_bc={L_bc.item():.3e} | "
                  f"W_pde={loss_weights[0].item():.3f}, W_bc={loss_weights[1].item():.3f}")

    return model, hist


# -----------------------
# 4) 視覺化
# -----------------------
def visualize(model: FNN, hist: dict, device: str = "cpu"):
    model.eval()
    dev = torch.device(device)

    # (A) u(x) 與解析解
    xs = np.linspace(-1, 1, 400).reshape(-1, 1).astype(np.float32)
    xt = to_tensor(xs, dev)
    with torch.no_grad():
        up = model(xt).cpu().numpy()
    ua = true_solution(xs)

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ua, label="Analytical")
    plt.plot(xs, up, label="PINN (GradNorm)")
    plt.title("Solution Comparison")
    plt.xlabel("x"); plt.ylabel("u(x)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # (B) Loss 曲線
    iters = np.arange(1, len(hist["L_total"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(iters, hist["L_total"], label="Total Loss")
    plt.plot(iters, hist["L_pde"], label="L_pde")
    plt.plot(iters, hist["L_bc"], label="L_bc")
    plt.title("Loss Curves")
    plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()

    # (C) 權重演化
    plt.figure(figsize=(7, 5))
    plt.plot(iters, hist["W_pde"], label="w_pde")
    plt.plot(iters, hist["W_bc"], label="w_bc")
    plt.title("GradNorm Weights Evolution")
    plt.xlabel("Iteration"); plt.ylabel("Weight")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()


# -----------------------
# 5) 進入點
# -----------------------
if __name__ == "__main__":
    cfg = TrainConfig()
    model, hist = train_gradnorm(cfg)
    visualize(model, hist, cfg.device)
