**結論先行**：進階損失設計（gPINN、Causal PINN、MOO）＝用「更多物理訊息」「時間因果課表」「帕雷托權衡」三件事，補上標準 PINN 在梯度/激波/多目標衝突下的弱點，顯著提升穩定度與精度。

---

# 5.1 Theory of Advanced and Specialized Loss Functions（教科書版）

## 0) 讀前心智圖

* **gPINN**：把「梯度/通量/應變」等**導數量測**納入損失 → 讓解不只數值對、**斜率也對**。
* **Causal PINN**：在時間維度實施**因果課表** → 先學早期，再學後期，避免「先知道未來」。
* **MOO for PINN**：把 PINN 視為**多目標最佳化** → 用帕雷托/梯度調和動態配重，取代拍腦袋的權重。

---

## 1) 簡介（Knowledge Area）

標準 PINN 的總損失

\[
\mathcal L_{\text{PINN}}
= \lambda_{\text{PDE}}\mathcal L_{\text{PDE}}
+ \lambda_{\text{BC}}\mathcal L_{\text{BC}}
+ \lambda_{\text{IC}}\mathcal L_{\text{IC}}
+ \lambda_{\text{data}}\mathcal L_{\text{data}}
\]

在含激波、剛性方程、噪聲量測或多約束衝突時，常出現**收斂慢、震盪、偏移**。本章三種增強路徑分別對應三類瓶頸。

---

## 2) Gradient-Enhanced PINNs（gPINNs）

### 2.1 第一性原理

物理定律往往**直接支配導數/通量**（例：Fourier \(q=-k\nabla T\)、Hooke \(\sigma = C:\varepsilon\)）。若手上有導數量測或可可靠計算的導數「真值」\(\nabla u_{\text{data}}\)，就應把它做為**等權目標**。

### 2.2 基本形式

在一組導數監督點 \(\{\mathbf x^{(g)}_i\}_{i=1}^{N_g}\) 上加一項：

\[
\mathcal L_{\text{grad}}
=\frac{1}{N_g}\sum_{i=1}^{N_g}
\left\|\nabla u_\theta\!\big(\mathbf x^{(g)}_i\big)
-\nabla u_{\text{data}}\!\big(\mathbf x^{(g)}_i\big)\right\|_2^2,
\quad
\mathcal L_{\text{total}}=\mathcal L_{\text{PINN}}+ \omega_{\text{grad}}\mathcal L_{\text{grad}}.
\]

> 若是**只知道通量** \(q=-k\nabla T\)，可改用 \(\|q_\theta - q_{\text{data}}\|^2\)。

### 2.3 何時有效

* 逆問題（從內部感測器回推係數/源項）
* 強耦合場（熱–應力、多物理耦合）
* 邊界層/梯度跨越大之區域（薄板導熱、黏性邊界層）

### 2.4 最小實作（PyTorch-style；展示觀念）

```python
class PINN(nn.Module):
    def forward(self, x):  # x: (N,d)
        return net(x)      # uθ(x)

def grad_u(u, x):
    (du_dx,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
    return du_dx

# L_grad
u_pred_g = model(x_g).sum(dim=1, keepdim=True)
du_pred_g = grad_u(u_pred_g, x_g)               # ∇uθ(xg)
L_grad = ((du_pred_g - du_data)**2).mean()

L_total = L_pinn + w_grad * L_grad
```

### 2.5 DeepXDE 小技巧

* 有**導數目標**：可用 `OperatorBC` 寫成 \(\partial u/\partial n - d_{\text{data}}(x)=0\) 型式。
* 有**通量目標**：把算子改為 \(q(u,\nabla u)-q_{\text{data}}(x)\)。

**口訣**：**「值對不夠，要把斜率也拉齊。」**

---

## 3) Causality-Informed PINNs（時間因果）

### 3.1 第一性原理：時間箭頭

對時間演化系統，解在 \(t\) 只能依賴 \(t'<t\)。一次性隨機抽整個 \([0,T]\) 的點，容易「先擬合後期、再回頭修前期」→ 不穩定、非因果。

### 3.2 兩種實作路徑

**(A) Time-Marching（嚴格因果）**
切分 \(0=t_0<t_1<\cdots<t_M=T\)，逐段訓練：

1. 在 \([t_k,t_{k+1}]\) 上以 \(u(\cdot,t_k)\) 為「IC」訓練至收斂；
2. 用學得的 \(u(\cdot,t_{k+1})\) 當下一段的「IC」。

> 穩但成本高，適用剛性或強非線性。

**(B) Causal Loss Weighting（高效率課表）**
把 PDE 殘差在時間上加權，讓**早期更重、後期漸次接手**：

\[
w(t)=\exp\!\big(-\lambda\,(T-t)\big),\quad
\mathcal L_{\text{PDE}}^{\text{causal}}
=\frac{1}{N}\sum_{i=1}^N w(t_i)\,\| \mathcal R_\theta(x_i,t_i)\|^2 .
\]

\(\lambda\) 可線上調整（例如隨訓練遞減），或採分段常數。

**DeepXDE 實務**：在 `pde(x, y)` 中取得 \(t=x[:,1:2]\) 後直接回傳 `w(t)*residual`（等價於在該點平方損失乘權重）。

**口訣**：**「先學早期，再學後期；時間課表保因果。」**

---

## 4) 把 PINN 視為多目標最佳化（MOO）

### 4.1 第一性原理：帕雷托最優

\(\mathcal L = \sum_i w_i \mathcal L_i\) 本質是把多目標（PDE/BC/IC/Data/Grad/Interface…）線性匯總。固定 \(w_i\) 往往**不魯棒**。MOO 要找的是**帕雷托前緣**上的解：改善某一目標不會讓另一個變差。

### 4.2 實用演算法（訓練時自動配重/調和梯度）

* **GradNorm**：調整 \(w_i\) 使各任務**梯度規模**均衡。

  \[
  w_i \leftarrow w_i \Big(\frac{G_i}{\bar G}\Big)^\alpha,\quad
  G_i=\big\|\nabla_\theta (w_i\mathcal L_i)\big\|.
  \]
* **MGDA（Multiple Gradient Descent Algorithm）**：求一組係數 \(\alpha_i\ge0,\sum\alpha_i=1\)，使 \(\sum\alpha_i g_i\) 成為**共同下降方向**（\(g_i=\nabla_\theta \mathcal L_i\)）。
* **PCGrad**：若 \(g_i\cdot g_j<0\)，把 \(g_i\) 在 \(g_j\) 上的衝突分量投影掉。
* **不確定性加權（Kendall–Gal）**：把各 \(\mathcal L_i\) 乘 \(\frac{1}{2\sigma_i^2}\) 並加 \(\log\sigma_i\) 正則，學習 \(\sigma_i\)。
* **NTK-based AdaLoss**：動態調整權重，使不同損失在**函數空間**的下降率相近。

> 這些方法在 PINN 很實用：如 PDE 殘差與 BC/IC 損失在早期階段尺度相差數個量級。

### 4.3 簡要 PyTorch 片段（PCGrad 概念）

```python
losses = [L_pde, L_bc, L_ic, w_grad*L_grad]
grads = []
for L in losses:
    opt.zero_grad()
    L.backward(retain_graph=True)
    grads.append([p.grad.detach().clone() for p in model.parameters()])

# PCGrad: 兩兩檢查，移除相互衝突分量
for i in range(len(grads)):
    for j in range(len(grads)):
        if i == j: continue
        dot = sum((gi*gj).sum() for gi, gj in zip(grads[i], grads[j]))
        if dot < 0:
            # gi <- gi - proj(gi on gj)
            norm = sum((gj*gj).sum() for gj in grads[j]) + 1e-12
            for k, gi in enumerate(grads[i]):
                grads[i][k] = gi - dot / norm * grads[j][k]

# 聚合梯度再更新
opt.zero_grad()
for p, *gs in zip(model.parameters(), *grads):
    p.grad = sum(gs) / len(gs)
opt.step()
```

**口訣**：**「權重別死扣，讓梯度彼此不打架。」**

---

## 5) 綜合策略與調參 SOP

1. **先做標準 PINN 的尺度對齊**：輸入/輸出正規化、損失初值同量級。
2. **加 gPINN**：若有導數/通量資訊，先嘗試小權重 \(\omega_{\text{grad}}\in[0.01,0.1]\)，觀察殘差曲線是否更穩。
3. **加 Causal**：時變問題先用 causal weighting（\(\lambda\) 取 \(1/T\sim5/T\)），嚴重剛性再考慮 time-marching。
4. **用 MOO 穩健化**：先上 GradNorm（穩定、易實作），再視情況換 PCGrad/MGDA。
5. **優化器**：**Adam 探路 → L-BFGS 精修**，遇到多目標時 L-BFGS 可在**後段**再開。
6. **除錯清單**：梯度爆/消失？不同損失曲線是否「此消彼長」？權重是否導致某一項永遠主導？

---

## 6) 迷你示例（觀念級）

### 6.1 gPINN：1D 導熱（已知局部熱通量）

\[
T_t=\kappa T_{xx},\quad q=-k T_x,\quad \text{在 } S_g \text{有 } q_{\text{meas}}.
\]

加上
\(\mathcal L_{\text{flux}}=\frac{1}{|S_g|}\sum_{x\in S_g}\| -k\,T_x(x)-q_{\text{meas}}(x)\|^2\)。

### 6.2 Causal Weighting：波動/對流

對殘差 \(\mathcal R(x,t)\) 用 \(w(t)=e^{-\lambda (T-t)}\) 乘權，\(\lambda\) 隨 epoch 緩慢下降。

### 6.3 MOO：\(\{\mathcal L_{\text{PDE}},\mathcal L_{\text{BC}},\mathcal L_{\text{IC}},\mathcal L_{\text{grad}}\}\)

用 GradNorm 或 PCGrad；觀察四條曲線是否「同向下降且無互相拉扯」。

---

## 7) 總結

* **gPINN**：把導數/通量當一等公民 → 在邊界層、逆問題特別有效。
* **Causal PINN**：在時間上「從近到遠」學習 → 收斂更穩、避免非因果。
* **MOO 視角**：權重動態化、梯度去衝突 → 減少手調權重、提高泛化。

**口訣記憶**

* **gPINN**：值準＋斜率準。
* **Causal**：早重晚輕，按時推進。
* **MOO**：不拗單一權重，讓梯度同向。

---

## 8) 參考文獻（精選）

1. Raissi, Perdikaris, Karniadakis. *Physics-Informed Neural Networks*, JCP, 2019.
2. Yu, Lu, Meng, Karniadakis. *Gradient-Enhanced PINNs*, arXiv:2111.02801.
3. Wang, Teng, Perdikaris. *Understanding and Mitigating Gradient Flow in PINNs*, JCP, 2021（含 NTK/自適應權重）。
4. Krishnapriyan et al. *Characterizing Failures of PINNs*, NeurIPS 2021.
5. Byun et al. *Causal PINNs for Advection-Dominated PDEs*, 2022（時間加權/課表思路）。
6. Chen et al. *PCGrad: Gradient Surgery for Multi-Task Learning*, NeurIPS 2020.
7. Sener & Koltun. *Multi-Task Learning as Multi-Objective Optimization*, NeurIPS 2018（MGDA）。
8. Kendall & Gal. *Uncertainty Weighing in Multi-Task Learning*, CVPR 2018（學習式權重）。

> 需要我把其中任一方法做成 **DeepXDE 最小可跑樣例**（含程式碼與可視化）嗎？我可以直接以 1D heat/Burgers 展示 gPINN＋Causal＋PCGrad 的整合骨架。
