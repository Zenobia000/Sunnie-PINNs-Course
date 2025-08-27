# Physics-Informed Neural Networks (PINNs) PyTorch 速成課程

## 課程概述

本課程旨在提供一個基於 PyTorch 的 Physics-Informed Neural Networks (PINNs) 速成指南。我們將從 PINNs 的核心思想出發，深入其損失函數的構成與優化挑戰，並透過一系列經典案例與先進框架的實作，學習如何使用 `DeepXDE` 等工具解決實際的科學計算問題。

## 課程目標

-   理解 PINN 作為代理模型的範式轉移，及其與傳統數值方法的區別。
-   能夠解構標準 PINN 的複合損失函數，並理解各組成部分的物理意義。
-   掌握利用自動微分 (Automatic Differentiation) 構建 PDE 殘差的核心技術。
-   分析 PINN 訓練中的優化挑戰，如梯度失衡、頻譜偏置與病態損失景觀。
-   學習並應用 SA-PINNs、VPINNs 等先進框架以應對特定的優化難題。
-   獲得解決正向問題、反向問題以及非線性、不連續問題的實戰經驗。

---

## 課程大綱

### Part 1: 第一原理 (First Principles) - 構成 PINNs 的基石

本部分將回歸問題的本質，介紹構成 PINNs 的兩大核心基礎：物理定律（微分方程）與神經網絡。

1.  **物理系統的語言：微分方程入門**
    *   何謂常微分方程 (ODEs) 與偏微分方程 (PDEs)？
    *   為何微分方程是描述物理現象的通用語言？
    *   經典物理方程概覽：熱傳導方程、波動方程、伯格斯方程 (Burgers' Equation)。
2.  **通用函數逼近器：神經網絡**
    *   前饋神經網絡的基本結構：權重、偏差、激活函數。
    *   萬能逼近定理 (Universal Approximation Theorem) 的直觀理解。
    *   如何利用神經網絡 `u_θ(x, t)` 來表示一個連續函數（方程的解）。

### Part 2: PINN 核心概念 (Core Concepts)

本部分將介紹 PINN 的基礎，闡述其如何將物理定律嵌入神經網路，並解析其損失函數的通用結構。

1.  **範式轉移**：從傳統求解器到可微分的物理代理模型。
2.  **損失函數解構**：深入解析 `L_total = w_pde*L_pde + w_bc*L_bc + w_ic*L_ic + w_data*L_data`。
3.  **自動微分的角色**：實現無網格計算的關鍵技術。
4.  **基礎實作**：以一個簡單的常微分方程（ODE）為例，展示 PINN 的基本工作流程。

### Part 3: 標準 PINNs 實戰：正向問題求解 (Forward Problems)

本部分將應用標準的 PINN 框架來求解幾個經典的偏微分方程（PDEs）。

1.  **拋物線型 PDE**：以一維熱傳導方程為例，學習處理時間演化問題。
2.  **橢圓型 PDE**：以二維泊松方程為例，學習處理複雜邊界條件。
3.  **非線性 PDE**：以一維伯格斯方程為例，探討非線性項 `u*u_x` 帶來的挑戰。

### Part 4: 進階應用：反向問題 (Inverse Problems)

本部分將展示 PINN 在數據同化與參數辨識中的強大能力。

1.  **問題定義**：如何利用稀疏觀測數據反推未知的物理參數。
2.  **案例研究**：從帶噪聲的溫度數據中，反向推斷熱傳導方程的擴散係數 `α`。

### Part 5: 優化挑戰與解決方案 (Optimization Challenges & Solutions)

本部分將深入探討 PINN 訓練困難的根源，並介紹為克服這些挑戰而設計的先進框架。

1.  **挑戰根源**：梯度戰爭、病態條件與神經網路的「頻譜偏置」。
2.  **解決方案一 (動態權重)**：用於處理剛性問題的自適應 PINNs (SA-PINNs)。
3.  **解決方案二 (馴服導數)**：利用變分弱形式改善穩定性的 VPINNs。
4.  **解決方案三 (捕捉不連續性)**：為求解帶有激波的守恆律而設計的 cPINNs。

### Part 6: 專門化與前沿主題 (Specialized & Advanced Topics)

1.  **專門化損失組件**：處理帶噪聲數據、發現多重解、梯度增強正規化 (gPINNs)。
2.  **決策框架**：如何為您的問題選擇合適的損失策略。
3.  **課程總結與未來展望**。

---

## 專案結構

本課程的程式碼與文檔將依循以下重新設計的結構進行組織，以反映從基礎到進階的完整學習路徑。

`PINNs-course/`
`├── 00-Course_Setup/`
`│   ├── README.md`
`│   └── pyproject.toml`
`│`
`├── 01-PINN_Core_Concepts/`
`│   ├── README.md`
`│   └── 1.1-Simple_ODE_Example/`
`│       ├── README.md`
`│       └── simple_ode.py`
`│`
`├── 02-Standard_PINNs_for_Forward_Problems/`
`│   ├── README.md`
`│   ├── 2.1-Parabolic_PDE-1D_Heat_Equation/`
`│   │   ├── README.md`
`│   │   └── heat_equation.py`
`│   ├── 2.2-Elliptic_PDE-2D_Poisson_Equation/`
`│   │   ├── README.md`
`│   │   └── poisson_equation.py`
`│   └── 2.3-Nonlinear_PDE-1D_Burgers_Equation/`
`│       ├── README.md`
`│       └── burgers_equation.py`
`│`
`├── 03-Advanced_Applications-Inverse_Problems/`
`│   ├── README.md`
`│   └── 3.1-Heat_Equation_Parameter_Discovery/`
`│       ├── README.md`
`│       └── heat_inverse.py`
`│`
`├── 04-Optimization_Challenges_and_Solutions/`
`│   ├── README.md`
`│   ├── 4.1-Theory-Gradient_Pathologies/`
`│   │   └── README.md`
`│   ├── 4.2-Adaptive_Weights-SA-PINN/`
`│   │   ├── README.md`
`│   │   └── sa_pinn_example.py`
`│   ├── 4.3-Variational_Formulation-VPINN/`
`│   │   ├── README.md`
`│   │   └── vpinn_example.py`
`│   └── 4.4-Discontinuous_Solutions-cPINN/`
`│       ├── README.md`
`│       └── cpinn_example.py`
`│`
`├── 05-Specialized_and_Frontier_Topics/`
`│   ├── README.md`
`│   ├── 5.1-Theory-Advanced_Loss_Functions/`
`│   │   └── README.md`
`│   └── 5.2-Example-Gradient_Enhanced_PINN/`
`│       ├── README.md`
`│       └── gpinn_example.py`
`│`
`├── 06-Capstone_Project-Navier_Stokes/`
`│   ├── README.md`
`│   └── 6.1-2D_Lid_Driven_Cavity_Flow/`
`│       ├── README.md`
`│       └── navier_stokes.py`
`│`
`├── Utils/`
`│   ├── README.md`
`│   └── plotters.py`
`│`
`└── README.md`