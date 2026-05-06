"""
Synthetic ICL regression datasets for LTV.

Tasks:
  linear_regression : LinearRegressionTask  (Garg et al. 2022, w ~ N(0, I_d))
  relu_regression   : TwoLayerReLUTask

Per-run usage pattern (in the runner):
  1. Sample one w for the whole run.
  2. Build demonstration (n_shot context pairs from w).
  3. Build train / test queries from the same w.
"""

import numpy as np

from .basetask import BaseTask


# ---------------------------------------------------------------------------
# Sampling / formatting helpers
# ---------------------------------------------------------------------------

def sample_x(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(n, d))


def fmt_x(x: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in x) + "]"


def fmt_example(x: np.ndarray, y: float, include_y: bool = True) -> str:
    if include_y:
        return f"x: {fmt_x(x)}\ny: {y:.4f}"
    else:
        return f"x: {fmt_x(x)}\ny:"


def build_demo(ctx_xs: np.ndarray, ctx_ys: np.ndarray, sep: str = "\n\n") -> str:
    return sep.join(fmt_example(x, float(y)) for x, y in zip(ctx_xs, ctx_ys)) + sep


def build_query(x: np.ndarray) -> str:
    return fmt_example(x, 0.0, include_y=False)


# ---------------------------------------------------------------------------
# Base regression task
# ---------------------------------------------------------------------------

class RegressionTask(BaseTask):
    """
    Abstract base for synthetic regression tasks.

    Regression data is generated on-the-fly via regenerate_with_w().
    all_data is left empty; the runner uses the per-run sampling protocol.
    """

    def __init__(self, task_name: str, d: int = 20, **kwargs):
        super().__init__(task_name=task_name, **kwargs)
        self.task_type = "regression"
        self.d = d
        self.class_num = None
        self.all_labels = None
        self.all_data = []
        print(f"[{self.__class__.__name__}] task={task_name}  d={d}")

    def _task_fn(self, w: np.ndarray, x: np.ndarray) -> float:
        raise NotImplementedError

    def _sample_w(self, d: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=d)

    def sample_w_for_run(self, d: int, rng: np.random.Generator) -> np.ndarray:
        return self._sample_w(d, rng)

    def regenerate_with_w(self, w: np.ndarray, n: int, seed: int = 0):
        """Return n (x, y) pairs all sharing the same w."""
        rng = np.random.default_rng(seed)
        xs = sample_x(n, self.d, rng)
        ys = np.array([self._task_fn(w, x) for x in xs])
        return xs, ys

    # BaseTask interface
    def get_dmonstration_template(self):
        return {"input": "x: {x}\ny:", "options": [], "format": ["x:", "y:"]}

    def get_task_instruction(self):
        return (
            "You are given input-output pairs from an unknown function. "
            "Predict the output for the final input.\n\n"
        )

    def apply_template(self, data):
        x_str = fmt_x(np.array(data["x"]))
        input_str = f"x: {x_str}\ny:"
        return input_str, [f"{data['y']:.4f}"], data["y"]

    def get_max_demonstration_token_length(self, tokenizer):
        tokens_per_example = 30
        name = getattr(tokenizer, "name_or_path", "").lower()
        if "llama-3" in name:
            cxt_max = 8192
        elif "llama-2" in name:
            cxt_max = 4096
        else:
            cxt_max = int(getattr(tokenizer, "model_max_length", 32768))
        return cxt_max - tokens_per_example


# ---------------------------------------------------------------------------
# Concrete tasks
# ---------------------------------------------------------------------------

class LinearRegressionTask(RegressionTask):
    """f(x) = w^T x,  w ~ N(0, I_d),  x ~ N(0, I_d)  (Garg et al., 2022)"""

    def _task_fn(self, w, x):
        return float(np.dot(w, x))


class TwoLayerReLUTask(RegressionTask):
    """
    f(x) = sum_i alpha_i * ReLU(w_i^T x)

    w is packed as [W.ravel(), alpha]:
      W     ~ N(0, I),   shape (r, d)
      alpha ~ N(0, 1/sqrt(r)), shape (r,)
    """

    def __init__(self, task_name: str, d: int = 20, r: int = 100, **kwargs):
        super().__init__(task_name=task_name, d=d, **kwargs)
        self.r = r

    def _sample_w(self, d: int, rng: np.random.Generator) -> np.ndarray:
        W = rng.normal(0.0, 1.0, size=(self.r, d))
        alpha = rng.normal(0.0, 1.0 / np.sqrt(self.r), size=self.r)
        return np.concatenate([W.ravel(), alpha])

    def _task_fn(self, w_flat: np.ndarray, x: np.ndarray) -> float:
        r, d = self.r, self.d
        W = w_flat[: r * d].reshape(r, d)
        alpha = w_flat[r * d:]
        return float(np.dot(alpha, np.maximum(0.0, W @ x)))
