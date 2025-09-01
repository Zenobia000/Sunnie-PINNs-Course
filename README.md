# 物理資訊神經網路：一部融合第一性原理與實踐導向的課程

## 前言：課程哲學

歡迎來到這門關於物理資訊神經網路 (PINNs) 的綜合課程。本課程專為期望超越對 PINNs 的淺層理解，並希望將其作為強大科學計算工具的研究生、研究人員及工程師而設計。

我們的核心哲學是彌合抽象理論與實際應用之間的鴻溝。為此，整個課程圍繞一套用於學習和解決問題的**標準作業程序 (SOP)** 來建構。此方法論為應對 PINNs 的任何新挑戰提供了一個清晰、可重複的框架。

---

## PINN 精通之路：標準作業程序 (SOP)

本課程的每個模組都建立在以下五個步驟的 SOP 之上：

1.  **掌握第一性原理 (Grasp the First Principle)**：在編寫任何程式碼之前，首先理解其背後的基礎物理和數學。系統遵循哪些守恆定律？控制微分方程的數學形式是什麼？
2.  **問題公式化 (Formulate the Problem)**：將物理定律和邊界條件轉化為 PINN 複合損失函數的各個組成部分 (`L_pde`, `L_bc`, `L_ic`, `L_data`)。這是設計一個成功的 PINN 最關鍵的步驟。
3.  **程式碼實作 (Implement the Code)**：編寫程式碼以執行公式化後的模型。這包括定義神經網路架構、指定問題域，並使用 PyTorch 或 DeepXDE 等框架實現損失函數。
4.  **訓練與驗證 (Train and Validate)**：運行優化過程以訓練網路參數。對比分析解、實驗數據或已建立的基準來驗證結果。
5.  **分析與迭代 (Analyze and Iterate)**：診斷訓練過程中的病態問題（例如，收斂緩慢、結果不準確）。如有必要，返回步驟 2 或 3，應用更進階的公式化或優化策略。

---

## 專案結構說明

本專案的目錄結構經過精心設計，以模組化的方式組織課程內容，確保學習路徑的清晰度。

```
PINNs-course/
├── 00-Course_Setup/              # 課程環境設定
├── 01-PINN_Core_Concepts/       # 核心概念與從零開始的實作
├── 02-Standard_PINNs_for_Forward_Problems/ # 使用 DeepXDE 解決標準正向問題
├── 03-Advanced_Applications-Inverse_Problems/ # 進階應用：反向問題
├── 04-Optimization_Challenges_and_Solutions/ # 優化挑戰與進階模型
├── 05-Specialized_and_Frontier_Topics/      # 特殊化與前沿主題
├── 06-Capstone_Project-Navier_Stokes/       # 畢業專案：納維-斯托克斯方程式
├── Utils/                        # 共用工具函式 (如繪圖)
└── README.md                     # 本說明檔案
```
每個模組資料夾內都包含一個 `README.md` 提供該模組的詳細說明，以及對應的 Python 腳本 (`.py`) 或 Jupyter Notebook (`.ipynb`) 實作。

---

## 課程大綱

### **模組 0：環境設定標準作業程序 (SOP)**

-   **目標**：為整個課程建立一個一致、可重現的 Python 環境。
-   **SOP**：
    1.  **系統驗證**：確保已安裝 Python 3.10+。
    2.  **安裝管理器**：安裝 `Poetry` 依賴性管理器。
    3.  **執行設定**：導航至專案根目錄並運行 `poetry install`。此命令會讀取 `pyproject.toml` 檔案，創建一個虛擬環境，並安裝所有必要的函式庫（PyTorch, DeepXDE 等）。
    4.  **驗證**：運行一個簡單的測試腳本，確認環境已正確設定。
-   **對應檔案**：
    - `00-Course_Setup/pyproject.toml`

### **模組 1：核心概念基礎 SOP**

-   **目標**：從零開始理解並實現一個 PINN，內化其核心機制。
-   **第一性原理**：萬能逼近定理 (The Universal Approximation Theorem)；作為可微分損失函數的物理定律。
-   **第一性原理 PINN 之 SOP (使用 PyTorch)**：
    1.  **定義架構**：創建一個 `torch.nn.Module` 作為通用函數逼近器 `u_θ(x, t)`。
    2.  **定義配置點**：從時空域中採樣訓練點。
    3.  **建構損失函數**：
        -   **PDE 損失 (`L_pde`)**：使用 `torch.autograd.grad` 計算網路輸出對其輸入的精確導數，進而構建 PDE 殘差。
        -   **邊界/初始損失 (`L_bc`/`L_ic`)**：通過懲罰網路預測與已知值之間的差異來強制執行問題的約束條件。
    4.  **實例化優化器**：選擇一個優化器，通常是 `torch.optim.Adam`。
    5.  **執行訓練迴圈**：重複執行 `optimizer.zero_grad()`、`loss.backward()` 和 `optimizer.step()` 的循環。
-   **對應檔案**：
    - `01-PINN_Core_Concepts/1.1-Simple_ODE_Example/simple_ode.py`
    - `01-PINN_Core_Concepts/1.1-Simple_ODE_Example/simple_ode.ipynb`

### **模組 2：正向問題標準 SOP (使用 DeepXDE)**

-   **目標**：掌握高階 `DeepXDE` 函式庫，以高效解決標準的正向 PDE 問題。
-   **第一性原理**：將 PINN 工作流程抽象為宣告式組件。
-   **DeepXDE 正向問題之 SOP**：
    1.  **定義域**：使用 `dde.geometry` 物件（`Interval`, `Rectangle`, `TimeDomain`）定義問題的幾何形狀。
    2.  **定義 PDE 殘差**：創建一個函數 `pde(x, u)`，返回控制方程式的殘差。
    3.  **定義條件**：使用 `dde.IC`、`dde.DirichletBC`、`dde.NeumannBC` 等指定所有約束。
    4.  **組裝數據**：從上述組件創建 `dde.data.TimePDE` 或 `dde.data.PDE` 物件。
    5.  **定義模型**：實例化一個 `dde.Model`，傳入數據物件和網路架構。
    6.  **訓練與預測**：使用優化器和學習率編譯模型，然後調用 `model.train()`。
-   **對應檔案**：
    - `02-Standard_PINNs_for_Forward_Problems/2.1-Parabolic_PDE-1D_Heat_Equation/heat_equation.py`
    - `02-Standard_PINNs_for_Forward_Problems/2.2-Elliptic_PDE-2D_Poisson_Equation/poisson_equation.py`
    - `02-Standard_PINNs_for_Forward_Problems/2.3-Nonlinear_PDE-1D_Burgers_Equation/burgers_equation.py`

### **模組 3：反向問題 SOP**

-   **目標**：利用 PINN 進行參數發現和數據同化。
-   **第一性原理**：擴展優化過程，不僅求解場變數，同時也找出 PDE 內部未知的物理參數。
-   **參數發現之 SOP**：
    1.  **識別未知數**：明確定義待發現的物理參數（例如，熱擴散係數 `α`）。
    2.  **定義為變數**：將未知數宣告為帶有初始猜測值的 `dde.Variable`。
    3.  **整合至 PDE**：在 `pde(x, u)` 函數中使用此 `dde.Variable`，使 PDE 殘差與之相關。
    4.  **提供觀測數據**：使用 `dde.PointSetBC` 提供稀疏的、已知的系統狀態測量值。
    5.  **訓練與監控**：訓練模型。優化器將同時更新網路權重和未知參數的值。監控 `dde.Variable` 的收斂情況。
-   **對應檔案**：
    - `03-Advanced_Applications-Inverse_Problems/3.1-Heat_Equation_Parameter_Discovery/heat_inverse.py`

### **模組 4：克服優化挑戰之 SOP**

-   **目標**：診斷並解決常見的 PINN 訓練病態問題。
-   **第一性原理**：理解梯度不平衡、頻譜偏置和損失景觀的剛性問題。
-   **穩健訓練之 SOP (兩階段優化)**：
    1.  **第一階段 (探索)**：使用 `Adam` 優化器進行大量迭代訓練。Adam 的自適應特性非常適合在複雜的非凸損失景觀中導航，並快速找到一個有希望的區域（一個寬而平的谷底）。
    2.  **第二階段 (精煉)**：切換到 `L-BFGS` 優化器。L-BFGS 作為一種擬牛頓法，一旦進入一個良好的吸引盆地，就能高效地收斂到一個精確的最小值。此兩階段過程結合了 Adam 的探索能力和 L-BFGS 的高精度收斂性。
-   **進階模型與對應檔案**：
    - **兩階段優化**: `04-Optimization_Challenges_and_Solutions/4.2-Strategy-Adam_and_LBFGS/optimization_strategy_example.py`
    - **cPINN**: `04-Optimization_Challenges_and_Solutions/4.4-Discontinuous_Solutions-cPINN/cpinn_example.py`
    - **FO-PINN**: `04-Optimization_Challenges_and_Solutions/4.5-First_Order_System-FO-PINN/fo_pinn_example.py`
    - 其他進階模型 (SA-PINN, VPINN, BPINN) 在 `README.md` 中提供理論和偽代碼。

### **模組 5：前沿問題 SOP**

-   **目標**：應用專門化和研究級別的技術，進一步提升 PINN 的性能。
-   **第一性原理**：通過更多物理資訊豐富損失函數 (gPINN)、尊重時間箭頭 (Causal PINN) 以及平衡相互競爭的目標 (MOO PINN)。
-   **進階應用之 SOP**：
    1.  **初步評估**：從標準 PINN 開始。它是否失敗？如果是，原因為何？（例如，在高梯度區域不準確、對時間演化不穩定、損失項之間存在衝突）。
    2.  **應用 gPINN**：如果您有關於導數（通量、應變）的數據，或者高梯度區域至關重要，請使用 `dde.PointSetOperatorBC` 添加梯度增強損失項。
    3.  **應用 Causal PINN**：對於時間相關問題，特別是對流主導的問題，在 PDE 殘差中實施因果加權方案，以強制執行學習課程。
    4.  **應用 MOO**：如果不同的損失項（PDE, BC, IC）難以和諧收斂，考慮使用如 GradNorm 或 PCGrad 的多目標優化策略，以動態平衡它們的影響。
-   **對應檔案**：
    - **gPINN**: `05-Specialized_and_Frontier_Topics/5.2-Example-Gradient_Enhanced_PINN/gpinn_example.py`
    - **Causal PINN**: `05-Specialized_and_Frontier_Topics/5.3-Example-Causal_PINN/causal_pinn_example.py`
    - **MOO PINN**: `05-Specialized_and_Frontier_Topics/5.4-Example-MOO_PINN/moo_pinn_example.py` (概念性偽代碼)

### **模組 6：畢業專案 SOP - 納維-斯托克斯方程式**

-   **目標**：綜合所有已學技能，解決一個複雜、耦合的非線性 PDE 系統。
-   **第一性原理**：不可壓縮流體的質量守恆（連續性方程）和動量守恆（納維-斯托克斯方程）。
-   **耦合 PDE 系統之 SOP**：
    1.  **定義多輸出網路**：網路必須為每個場變數（例如 `u`, `v`, `p`）設置一個輸出神經元。
    2.  **定義 PDE 系統**：`pde` 函數必須返回一個殘差列表，每個殘差對應一個控制方程式。
    3.  **應用綜合邊界條件**：為每個變數在每個相關邊界上定義 `dde.DirichletBC`。使用 `dde.PointSetBC` 在單個點上固定壓力，以確保解的唯一性。
    4.  **訓練與可視化**：訓練複雜模型。使用標準的 CFD 技術（例如，速度的箭頭圖、壓力的等高線圖）將結果可視化，以驗證解的物理真實性。
-   **對應檔案**：
    - `06-Capstone_Project-Navier_Stokes/6.1-2D_Lid_Driven_Cavity_Flow/navier_stokes.py`