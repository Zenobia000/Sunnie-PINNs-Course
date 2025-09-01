"""
causal_pinn_example_refined.py

Causal PINN for 1D inviscid Burgers' equation with curriculum-style
time weighting (重前輕後). Structured, typed, and extendable.

IMPORTANT:
- Set T <= ~0.318 (1/pi) to avoid shock; default T=0.3 for stable demo.
"""

from __future__ import annotations
import numpy as np
import deepxde as dde
from typing import Callable, Literal, Optional, Tuple

# -----------------------------
# 0. Global Config & Utilities
# -----------------------------
dde.config.set_random_seed(42)
dde.config.set_default_float("float32")

RealFn = Callable[[np.ndarray], np.ndarray]
WeightMode = Literal["exp_decay", "linear_decay"]

def isclose(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# -----------------------------------
# 1. Causal Weight (重前輕後) Schedules
# -----------------------------------
class CausalWeight:
    """Causal weight w(t): larger at small t, smaller at large t."""

    def __init__(self, T: float, mode: WeightMode = "exp_decay", lam: float = 5.0):
        """
        Args:
            T: final time horizon
            mode: 'exp_decay' uses w(t) = exp(-lam * t)
                  'linear_decay' uses w(t) = clip(1 - t/T, 0, 1)
            lam: decay rate for exponential mode
        """
        self.T = float(T)
        self.mode = mode
        self.lam = float(lam)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (N, 2) with columns [x, t]
        return shape: (N, 1)
        """
        t = x[:, 1:2]
        if self.mode == "exp_decay":
            w = np.exp(-self.lam * t)
        elif self.mode == "linear_decay":
            w = np.clip(1.0 - t / self.T, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown weight mode: {self.mode}")
        return w.astype(np.float32)


class CausalWeightScheduler(dde.callbacks.Callback):
    """
    Optional: anneal lambda during training (e.g., start stronger causal emphasis,
    then relax).
    """
    def __init__(self, cw: CausalWeight, lam_start: float, lam_end: float, max_steps: int):
        self.cw = cw
        self.lam_start = lam_start
        self.lam_end = lam_end
        self.max_steps = max_steps

    def on_epoch_end(self):
        step = self.model.train_state.step
        ratio = min(1.0, step / max(1, self.max_steps))
        self.cw.lam = self.lam_start * (1 - ratio) + self.lam_end * ratio


# -----------------------------------------
# 2. Problem Definition (Burgers, inviscid)
# -----------------------------------------
class BurgersCausalPINN:
    def __init__(
        self,
        T: float = 0.3,  # <= 1/pi advisable
        num_domain: int = 2500,
        num_boundary: int = 100,
        num_initial: int = 100,
        weight_mode: WeightMode = "exp_decay",
        lam: float = 5.0,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        solution_fn: Optional[RealFn] = None,  # set if you have a reference
    ):
        self.T = float(T)
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_initial = num_initial
        self.causal = CausalWeight(T=self.T, mode=weight_mode, lam=lam)
        self.loss_weights = loss_weights
        self.solution_fn = solution_fn  # None by default (no analytic ref)

        # Geometry in space-time
        self.geom = dde.geometry.Interval(-1.0, 1.0)
        self.timedomain = dde.geometry.TimeDomain(0.0, self.T)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)

        # IC: u(x,0) = -sin(pi x)
        self.ic = dde.icbc.IC(
            self.geomtime,
            lambda X: -np.sin(np.pi * X[:, 0:1]),
            lambda _, on_initial: on_initial,
        )

        # BC: u(-1,t)=u(1,t)=0, only on spatial boundary (exclude t=0/T)
        def boundary_x_only(X, on_boundary):
            if not on_boundary:
                return False
            x, t = X[0], X[1]
            # Only x = -1 or x = 1, not t boundaries
            return isclose(x, -1.0) or isclose(x, 1.0)

        self.bc = dde.icbc.DirichletBC(self.geomtime, lambda X: 0.0, boundary_x_only)

        # PDE residual with causal weight
        def pde_causal(X, u):
            """
            X: (x, t), with j=0 -> x, j=1 -> t
            u: network output
            """
            du_dx = dde.grad.jacobian(u, X, i=0, j=0)
            du_dt = dde.grad.jacobian(u, X, i=0, j=1)
            residual = du_dt + u * du_dx
            # Apply causal weight (重前輕後)
            w = self.causal(X)
            return w * residual

        self.pde = pde_causal

    def build_data(self) -> dde.data.TimePDE:
        return dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [self.bc, self.ic],
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_initial=self.num_initial,
            solution=self.solution_fn,  # None -> no built-in L2 metric
            # num_test can be set if solution_fn is provided
        )

    def build_net(self) -> dde.nn.NN:
        # 2 inputs (x, t) -> hidden -> 1 output
        return dde.nn.FNN([2] + [64, 64, 64] + [1], "tanh", "Glorot normal")

    def build_model(self) -> dde.Model:
        data = self.build_data()
        net = self.build_net()
        model = dde.Model(data, net)
        return model


def main_train():
    problem = BurgersCausalPINN(
        T=0.3,                  # keep <= 1/pi to avoid shock
        num_domain=2500,
        num_boundary=200,
        num_initial=200,
        weight_mode="exp_decay",  # or "linear_decay"
        lam=5.0,                # causal decay rate
        loss_weights=(1.0, 1.0, 1.0),
        solution_fn=None,       # set to a callable if you have reference
    )
    model = problem.build_model()

    # If you DO have a reference solution, also set metrics=["l2 relative error"]
    # and optionally num_test in build_data()
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=list(problem.loss_weights),
        # metrics=["l2 relative error"]  # enable only when solution_fn is not None
    )

    # Optional: anneal lambda from strong causal (e.g., 8) to mild (e.g., 2)
    cw_sched = CausalWeightScheduler(
        cw=problem.causal, lam_start=8.0, lam_end=2.0, max_steps=5000
    )

    # Stage 1: Adam
    model.train(iterations=8000, callbacks=[cw_sched], display_every=1000)

    # Stage 2: L-BFGS (refine)
    model.compile("L-BFGS")
    model.train()

    # If you want to evaluate manually on a grid (no built-in metrics):
    xs = np.linspace(-1, 1, 201).reshape(-1, 1)
    ts = np.full_like(xs, 0.1)  # a time slice t=0.1
    XT = np.hstack([xs, ts]).astype(np.float32)
    u_pred = model.predict(XT)
    print("Prediction slice t=0.1 -> u shape:", u_pred.shape)
    print("Causal PINN training finished.")


if __name__ == "__main__":
    main_train()
