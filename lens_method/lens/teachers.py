"""Teacher factory contracts and locked tree/neural/kernel implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

TaskType = Literal["regression", "binary"]
TeacherName = Literal["xgboost", "softplus_mlp"]


class FittedTeacher(Protocol):
    def predict_margin(self, X: np.ndarray) -> np.ndarray: ...


class TeacherFactory(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> FittedTeacher: ...


def _validated_regression_arrays(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("X and y must have aligned shapes (n,p) and (n,)")
    if len(features) < 4 or not np.all(np.isfinite(features)):
        raise ValueError("teacher training requires at least four finite rows")
    if not np.all(np.isfinite(target)):
        raise ValueError("teacher labels must be finite")
    return features, target


def _validated_binary_arrays(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features, target = _validated_mlp_arrays(X, y)
    unique = np.unique(target)
    if not np.array_equal(unique, np.asarray([0.0, 1.0])):
        raise ValueError("binary teacher labels must contain exactly {0,1}")
    counts = np.asarray([(target == value).sum() for value in unique])
    if np.any(counts < 2):
        raise ValueError("binary teacher training needs at least two rows per class")
    return features, target


def _validated_mlp_arrays(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate neural-Teacher arrays while preserving imputable NaNs."""

    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("X and y must have aligned shapes (n,p) and (n,)")
    if len(features) < 4 or np.any(np.isinf(features)):
        raise ValueError(
            "MLP teacher training requires at least four rows without infinite features"
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("teacher labels must be finite")
    return features, target


def _stratified_validation_indices(
    target: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation indices with both classes in each."""

    rng = np.random.default_rng(seed)
    validation_parts: list[np.ndarray] = []
    training_parts: list[np.ndarray] = []
    for value in (0.0, 1.0):
        indices = np.flatnonzero(target == value)
        indices = indices[rng.permutation(len(indices))]
        n_validation = max(1, int(round(fraction * len(indices))))
        n_validation = min(n_validation, len(indices) - 1)
        validation_parts.append(indices[:n_validation])
        training_parts.append(indices[n_validation:])
    validation = np.concatenate(validation_parts)
    training = np.concatenate(training_parts)
    return training[rng.permutation(len(training))], validation[
        rng.permutation(len(validation))
    ]


@dataclass
class _XGBoostTeacher:
    model: Any
    device: str = "cpu"

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X)
        if self.device.startswith("cuda"):
            try:
                import cupy as cp
            except Exception as exc:
                raise RuntimeError(
                    "CUDA XGBoost inference requires CuPy; refusing the slower "
                    "NumPy/device-mismatch fallback"
                ) from exc
            prediction = self.model.get_booster().inplace_predict(
                cp.asarray(matrix),
                predict_type="margin",
            )
            prediction = cp.asnumpy(prediction)
        else:
            prediction = self.model.predict(matrix, output_margin=True)
        return np.asarray(prediction, dtype=np.float64).reshape(-1)


@dataclass
class XGBoostTeacherFactory:
    """Deterministically seeded XGBoost factory.

    Confirmation and every permutation refit must receive the same ``seed``.
    The label vector is the only varying input to that exchangeable training map.
    """

    task: TaskType
    params: dict[str, Any] = field(default_factory=dict)
    name: str = "xgboost"

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> _XGBoostTeacher:
        from xgboost import XGBClassifier, XGBRegressor

        common: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": 1,
            **self.params,
        }
        if self.task == "binary":
            model = XGBClassifier(eval_metric="logloss", **common)
        elif self.task == "regression":
            model = XGBRegressor(objective="reg:squarederror", **common)
        else:
            raise ValueError(f"unknown task: {self.task!r}")
        model.fit(np.asarray(X), np.asarray(y))
        return _XGBoostTeacher(model, device=str(common.get("device", "cpu")))


@dataclass
class _SoftplusMLPTeacher:
    torch_model: Any
    core_model: Any
    device: str
    batch_size: int
    fit_metadata: dict[str, Any]

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        import torch

        features = np.asarray(X, dtype=np.float32)
        if features.ndim != 2 or np.any(np.isinf(features)):
            raise ValueError(
                "MLP prediction input must be 2-D and may contain NaN but not inf"
            )
        outputs: list[np.ndarray] = []
        self.torch_model.eval()
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                batch = torch.from_numpy(features[start : start + self.batch_size]).to(
                    self.device
                )
                values = self.torch_model(batch).detach().cpu().numpy().reshape(-1)
                outputs.append(values)
        prediction = np.concatenate(outputs).astype(np.float64, copy=False)
        if not np.all(np.isfinite(prediction)):
            raise ValueError("MLP teacher emitted NaN or inf")
        return prediction


@dataclass
class SoftplusMLPTeacherFactory:
    """Locked Softplus MLP for the formal teacher-family sensitivity track."""

    task: TaskType = "regression"
    hidden: tuple[int, ...] = (256, 128, 64)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    prediction_batch_size: int = 65536
    max_epochs: int = 200
    min_epochs: int = 30
    patience: int = 15
    validation_fraction: float = 0.15
    tolerance: float = 1e-5
    device: str = "cuda"
    name: str = "softplus_mlp"

    def __post_init__(self) -> None:
        if self.task not in ("regression", "binary"):
            raise ValueError(f"unknown task: {self.task!r}")
        if not self.hidden or any(width < 1 for width in self.hidden):
            raise ValueError("hidden widths must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid MLP optimizer settings")
        if self.batch_size < 1 or self.prediction_batch_size < 1:
            raise ValueError("MLP batch sizes must be positive")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0,0.5)")
        if not 1 <= self.min_epochs <= self.max_epochs or self.patience < 1:
            raise ValueError("invalid MLP epoch or patience settings")

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, seed: int
    ) -> _SoftplusMLPTeacher:
        import torch
        from torch import nn

        if self.task == "binary":
            features, target = _validated_binary_arrays(X, y)
        else:
            features, target = _validated_mlp_arrays(X, y)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Softplus MLP but is unavailable")
        torch_device = torch.device(self.device)
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)

        if self.task == "binary":
            training_indices, validation_indices = _stratified_validation_indices(
                target, fraction=self.validation_fraction, seed=seed
            )
        else:
            permutation = np.random.default_rng(seed).permutation(len(features))
            n_validation = max(
                1, int(round(self.validation_fraction * len(features)))
            )
            if len(features) - n_validation < 2:
                raise ValueError("MLP internal training split is too small")
            validation_indices = permutation[:n_validation]
            training_indices = permutation[n_validation:]
        X_training = features[training_indices]
        y_training = target[training_indices]
        x_impute = np.empty(features.shape[1], dtype=np.float64)
        all_missing_columns: list[int] = []
        for column in range(features.shape[1]):
            observed = X_training[~np.isnan(X_training[:, column]), column]
            if observed.size:
                x_impute[column] = float(np.median(observed))
            else:
                x_impute[column] = 0.0
                all_missing_columns.append(column)
        X_imputed = np.where(np.isnan(features), x_impute[None, :], features)
        X_training_imputed = X_imputed[training_indices]
        x_mean = X_training_imputed.mean(axis=0)
        x_scale = X_training_imputed.std(axis=0)
        x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
        X_normalized = ((X_imputed - x_mean) / x_scale).astype(np.float32)
        if self.task == "binary":
            n_negative = int(np.count_nonzero(y_training == 0.0))
            n_positive = int(np.count_nonzero(y_training == 1.0))
            positive_class_weight = float(n_negative / n_positive)
            # Weighted BCE targets balanced discrimination.  Its population
            # optimum is shifted by log(pos_weight), so remove that constant
            # from the exported raw margin.  The correction is learned from
            # the internal training subset only and cannot alter pairwise
            # interaction contrasts by itself.
            y_mean = float(-np.log(positive_class_weight))
            y_scale = 1.0
            y_model_target = target.astype(np.float32)
            loss_name = "class_balanced_BCEWithLogitsLoss"
        else:
            n_negative = None
            n_positive = None
            positive_class_weight = None
            y_mean = float(y_training.mean())
            y_scale = float(y_training.std())
            if not np.isfinite(y_scale) or y_scale <= 1e-12:
                y_scale = 1.0
            y_model_target = ((target - y_mean) / y_scale).astype(np.float32)
            loss_name = "MSELoss"

        layers: list[Any] = []
        previous = features.shape[1]
        for width in self.hidden:
            layers.extend((nn.Linear(previous, width), nn.Softplus()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        core = nn.Sequential(*layers).to(torch_device)
        optimizer = torch.optim.AdamW(
            core.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_function = (
            nn.BCEWithLogitsLoss(
                pos_weight=torch.as_tensor(
                    positive_class_weight,
                    dtype=torch.float32,
                    device=torch_device,
                )
            )
            if self.task == "binary"
            else nn.MSELoss()
        )
        X_tensor = torch.from_numpy(X_normalized).to(torch_device)
        y_tensor = torch.from_numpy(y_model_target[:, None]).to(torch_device)
        train_index_tensor = torch.as_tensor(
            training_indices, dtype=torch.long, device=torch_device
        )
        validation_index_tensor = torch.as_tensor(
            validation_indices, dtype=torch.long, device=torch_device
        )
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(seed + 1)
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            core.train()
            local_order = torch.randperm(len(training_indices), generator=order_generator)
            for start in range(0, len(training_indices), self.batch_size):
                local = local_order[start : start + self.batch_size].to(torch_device)
                indices = train_index_tensor[local]
                prediction = core(X_tensor[indices])
                loss = loss_function(prediction, y_tensor[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            core.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(
                        core(X_tensor[validation_index_tensor]),
                        y_tensor[validation_index_tensor],
                    )
                    .detach()
                    .cpu()
                )
            if validation_loss < best_loss - self.tolerance:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in core.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
            if epoch + 1 >= self.min_epochs and bad_epochs >= self.patience:
                break
        if best_state is None:
            raise RuntimeError("MLP training did not produce a finite validation state")
        core.load_state_dict(best_state)
        core.eval()

        class _RawScaleModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.core = core
                self.register_buffer(
                    "x_impute", torch.as_tensor(x_impute, dtype=torch.float32)
                )
                self.register_buffer(
                    "x_mean", torch.as_tensor(x_mean, dtype=torch.float32)
                )
                self.register_buffer(
                    "x_scale", torch.as_tensor(x_scale, dtype=torch.float32)
                )
                self.register_buffer(
                    "y_mean", torch.as_tensor(y_mean, dtype=torch.float32)
                )
                self.register_buffer(
                    "y_scale", torch.as_tensor(y_scale, dtype=torch.float32)
                )

            def forward(self, matrix):
                imputed = torch.where(torch.isnan(matrix), self.x_impute, matrix)
                standardized = self.core((imputed - self.x_mean) / self.x_scale)
                return standardized * self.y_scale + self.y_mean

        raw_model = _RawScaleModule().to(torch_device).eval()
        metadata = {
            "task": self.task,
            "architecture": list(self.hidden),
            "activation": "Softplus",
            "loss": loss_name,
            "prediction_margin": "raw_logit" if self.task == "binary" else "raw_response",
            "class_imbalance_policy": (
                "training_subset_inverse_prevalence_weight"
                if self.task == "binary"
                else None
            ),
            "negative_training_rows": n_negative,
            "positive_training_rows": n_positive,
            "positive_class_weight": positive_class_weight,
            "logit_prior_correction": (
                y_mean if self.task == "binary" else None
            ),
            "missing_value_policy": "training_subset_column_median",
            "training_missing_value_count": int(np.isnan(X_training).sum()),
            "all_missing_training_columns_filled_with_zero": all_missing_columns,
            "optimizer": "AdamW",
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "selected_epoch": best_epoch,
            "validation_loss": best_loss,
            "validation_loss_standardized": (
                best_loss if self.task == "regression" else None
            ),
            "validation_fraction": self.validation_fraction,
            "device": str(torch_device),
            "seed": int(seed),
            "silent_fallback": False,
        }
        return _SoftplusMLPTeacher(
            torch_model=raw_model,
            core_model=core,
            device=str(torch_device),
            batch_size=self.prediction_batch_size,
            fit_metadata=metadata,
        )


def make_locked_teacher_factory(
    name: TeacherName, *, task: TaskType, device: str
) -> TeacherFactory:
    """Build a frozen prediction-interface Teacher without dataset-specific rules."""

    if name == "xgboost":
        return XGBoostTeacherFactory(task, params={"device": device})
    if name == "softplus_mlp":
        return SoftplusMLPTeacherFactory(task=task, device=device)
    raise ValueError(f"unknown teacher: {name!r}")


@dataclass
class _FTTransformerTeacher:
    model: Any
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    device: str
    batch_size: int
    fit_metadata: dict[str, Any]

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        import torch

        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("FT-Transformer prediction input must be a finite 2-D array")
        normalized = ((features - self.x_mean) / self.x_scale).astype(np.float32)
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(normalized), self.batch_size):
                batch = torch.from_numpy(normalized[start : start + self.batch_size]).to(
                    self.device
                )
                values = self.model(batch, None).detach().cpu().numpy().reshape(-1)
                outputs.append(values)
        prediction = (
            np.concatenate(outputs).astype(np.float64, copy=False) * self.y_scale
            + self.y_mean
        )
        if not np.all(np.isfinite(prediction)):
            raise ValueError("FT-Transformer teacher emitted NaN or inf")
        return prediction


@dataclass
class FTTransformerTeacherFactory:
    """Official RTDL default FT-Transformer with a locked training rule."""

    task: TaskType = "regression"
    batch_size: int = 1024
    prediction_batch_size: int = 4096
    max_epochs: int = 200
    min_epochs: int = 30
    patience: int = 15
    validation_fraction: float = 0.15
    tolerance: float = 1e-5
    device: str = "cuda"
    name: str = "ft_transformer"

    def __post_init__(self) -> None:
        if self.task != "regression":
            raise ValueError("FTTransformerTeacherFactory currently supports regression")
        if self.batch_size < 1 or self.prediction_batch_size < 1:
            raise ValueError("FT-Transformer batch sizes must be positive")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0,0.5)")
        if not 1 <= self.min_epochs <= self.max_epochs or self.patience < 1:
            raise ValueError("invalid FT-Transformer epoch or patience settings")

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, seed: int
    ) -> _FTTransformerTeacher:
        import torch

        features, target = _validated_regression_arrays(X, y)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for FT-Transformer but is unavailable")
        try:
            from rtdl_revisiting_models import FTTransformer
        except ImportError as exc:
            raise RuntimeError(
                "FT-Transformer requires the official rtdl_revisiting_models package"
            ) from exc

        torch_device = torch.device(self.device)
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        permutation = np.random.default_rng(seed).permutation(len(features))
        n_validation = max(1, int(round(self.validation_fraction * len(features))))
        if len(features) - n_validation < 2:
            raise ValueError("FT-Transformer internal training split is too small")
        validation_indices = permutation[:n_validation]
        training_indices = permutation[n_validation:]
        X_training = features[training_indices]
        y_training = target[training_indices]
        x_mean = X_training.mean(axis=0)
        x_scale = X_training.std(axis=0)
        x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
        y_mean = float(y_training.mean())
        y_scale = float(y_training.std())
        if not np.isfinite(y_scale) or y_scale <= 1e-12:
            y_scale = 1.0
        X_normalized = ((features - x_mean) / x_scale).astype(np.float32)
        y_normalized = ((target - y_mean) / y_scale).astype(np.float32)

        default_kwargs = FTTransformer.get_default_kwargs()
        model = FTTransformer(
            n_cont_features=features.shape[1],
            cat_cardinalities=[],
            d_out=1,
            **default_kwargs,
        ).to(torch_device)
        optimizer = model.make_default_optimizer()
        loss_function = torch.nn.MSELoss()
        X_tensor = torch.from_numpy(X_normalized).to(torch_device)
        y_tensor = torch.from_numpy(y_normalized[:, None]).to(torch_device)
        train_index_tensor = torch.as_tensor(
            training_indices, dtype=torch.long, device=torch_device
        )
        validation_index_tensor = torch.as_tensor(
            validation_indices, dtype=torch.long, device=torch_device
        )
        order_generator = torch.Generator(device="cpu")
        order_generator.manual_seed(seed + 1)
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        bad_epochs = 0

        for epoch in range(self.max_epochs):
            model.train()
            local_order = torch.randperm(len(training_indices), generator=order_generator)
            for start in range(0, len(training_indices), self.batch_size):
                local = local_order[start : start + self.batch_size].to(torch_device)
                indices = train_index_tensor[local]
                prediction = model(X_tensor[indices], None)
                loss = loss_function(prediction, y_tensor[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(
                        model(X_tensor[validation_index_tensor], None),
                        y_tensor[validation_index_tensor],
                    )
                    .detach()
                    .cpu()
                )
            if validation_loss < best_loss - self.tolerance:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
            if epoch + 1 >= self.min_epochs and bad_epochs >= self.patience:
                break
        if best_state is None:
            raise RuntimeError(
                "FT-Transformer training did not produce a finite validation state"
            )
        model.load_state_dict(best_state)
        model.eval()
        metadata = {
            "official_package": "rtdl_revisiting_models",
            "package_version": importlib_metadata.version(
                "rtdl_revisiting_models"
            ),
            "architecture": {
                key: value for key, value in default_kwargs.items() if not key.startswith("_")
            },
            "optimizer": "official make_default_optimizer (AdamW)",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "selected_epoch": best_epoch,
            "validation_loss_standardized": best_loss,
            "validation_fraction": self.validation_fraction,
            "device": str(torch_device),
            "seed": int(seed),
            "silent_fallback": False,
        }
        return _FTTransformerTeacher(
            model=model,
            x_mean=np.asarray(x_mean, dtype=np.float64),
            x_scale=np.asarray(x_scale, dtype=np.float64),
            y_mean=y_mean,
            y_scale=y_scale,
            device=str(torch_device),
            batch_size=self.prediction_batch_size,
            fit_metadata=metadata,
        )


@dataclass
class _TabPFNTeacher:
    estimator: Any
    batch_size: int
    fit_metadata: dict[str, Any]

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("TabPFN prediction input must be a finite 2-D array")
        outputs: list[np.ndarray] = []
        for start in range(0, len(features), self.batch_size):
            values = self.estimator.predict(features[start : start + self.batch_size])
            outputs.append(np.asarray(values, dtype=np.float64).reshape(-1))
        prediction = np.concatenate(outputs)
        if prediction.shape != (len(features),) or not np.all(np.isfinite(prediction)):
            raise ValueError("TabPFN teacher emitted invalid predictions")
        return prediction


@dataclass
class TabPFNTeacherFactory:
    """Pinned local-checkpoint TabPFN regressor used as an in-context teacher."""

    model_path: str | Path
    task: TaskType = "regression"
    n_estimators: int = 8
    prediction_batch_size: int = 4096
    device: str = "cuda"
    name: str = "tabpfn"

    def __post_init__(self) -> None:
        if self.task != "regression":
            raise ValueError("TabPFNTeacherFactory currently supports regression")
        if self.n_estimators < 1 or self.prediction_batch_size < 1:
            raise ValueError("invalid TabPFN ensemble or prediction batch size")
        checkpoint = Path(self.model_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"TabPFN checkpoint does not exist: {checkpoint}")
        self.model_path = checkpoint.resolve()

    def fit(self, X: np.ndarray, y: np.ndarray, *, seed: int) -> _TabPFNTeacher:
        import torch

        features, target = _validated_regression_arrays(X, y)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for TabPFN but is unavailable")
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise RuntimeError("TabPFN requires the official tabpfn package") from exc

        checkpoint = Path(self.model_path)
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        estimator = TabPFNRegressor(
            n_estimators=self.n_estimators,
            categorical_features_indices=[],
            model_path=checkpoint,
            device=self.device,
            ignore_pretraining_limits=False,
            inference_precision=torch.float32,
            fit_mode="fit_with_cache",
            memory_saving_mode="auto",
            random_state=seed,
            n_preprocessing_jobs=1,
        )
        estimator.fit(features, target)
        metadata = {
            "official_package": "tabpfn",
            "package_version": importlib_metadata.version("tabpfn"),
            "model_path": str(checkpoint),
            "model_sha256": checkpoint_sha256,
            "n_estimators": self.n_estimators,
            "fit_mode": "fit_with_cache",
            "inference_precision": "torch.float32",
            "external_standardization": False,
            "device": self.device,
            "seed": int(seed),
            "silent_fallback": False,
        }
        return _TabPFNTeacher(
            estimator=estimator,
            batch_size=self.prediction_batch_size,
            fit_metadata=metadata,
        )


@dataclass
class _RBFNystroemTeacher:
    x_mean: np.ndarray
    x_scale: np.ndarray
    components: np.ndarray
    gamma: float
    kernel_weights: np.ndarray
    intercept: float
    y_mean: float
    y_scale: float
    device: str
    batch_size: int
    fit_metadata: dict[str, Any]
    reference_transform: Any
    reference_ridge: Any

    def _predict_cpu(self, X: np.ndarray) -> np.ndarray:
        from sklearn.metrics.pairwise import rbf_kernel

        normalized = (np.asarray(X, dtype=np.float64) - self.x_mean) / self.x_scale
        values = (
            rbf_kernel(normalized, self.components, gamma=self.gamma)
            @ self.kernel_weights
            + self.intercept
        )
        return np.asarray(values * self.y_scale + self.y_mean, dtype=np.float64)

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=np.float64)
        if features.ndim != 2 or not np.all(np.isfinite(features)):
            raise ValueError("RBF prediction input must be a finite 2-D array")
        if not self.device.startswith("cuda"):
            prediction = self._predict_cpu(features)
        else:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA RBF prediction requested but CUDA is unavailable")
            torch_device = torch.device(self.device)
            mean = torch.as_tensor(self.x_mean, dtype=torch.float32, device=torch_device)
            scale = torch.as_tensor(self.x_scale, dtype=torch.float32, device=torch_device)
            centers = torch.as_tensor(
                self.components, dtype=torch.float32, device=torch_device
            )
            weights = torch.as_tensor(
                self.kernel_weights, dtype=torch.float32, device=torch_device
            )
            center_norm = torch.sum(centers * centers, dim=1)[None, :]
            outputs: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(features), self.batch_size):
                    batch = torch.as_tensor(
                        features[start : start + self.batch_size],
                        dtype=torch.float32,
                        device=torch_device,
                    )
                    batch = (batch - mean) / scale
                    squared = (
                        torch.sum(batch * batch, dim=1)[:, None]
                        + center_norm
                        - 2.0 * batch @ centers.T
                    ).clamp_min_(0.0)
                    values = torch.exp(-self.gamma * squared) @ weights
                    values = (values + self.intercept) * self.y_scale + self.y_mean
                    outputs.append(values.detach().cpu().numpy())
            prediction = np.concatenate(outputs).astype(np.float64, copy=False)
        if not np.all(np.isfinite(prediction)):
            raise ValueError("RBF teacher emitted NaN or inf")
        return prediction

    def predict_margin_reference(self, X: np.ndarray) -> np.ndarray:
        """Float64 transform-equivalent prediction used by the audit tests."""
        normalized = (np.asarray(X, dtype=np.float64) - self.x_mean) / self.x_scale
        values = self.reference_ridge.predict(
            self.reference_transform.transform(normalized)
        )
        prediction = np.asarray(values * self.y_scale + self.y_mean, dtype=np.float64)
        if not np.all(np.isfinite(prediction)):
            raise ValueError("reference RBF teacher emitted NaN or inf")
        return prediction


@dataclass
class RBFNystroemTeacherFactory:
    """Label-independent median-bandwidth Nyström RBF kernel ridge teacher."""

    task: TaskType = "regression"
    n_components: int = 512
    alpha: float = 1e-3
    median_sample_size: int = 2048
    device: str = "cuda"
    prediction_batch_size: int = 65536
    name: str = "rbf_nystroem_krr"

    def __post_init__(self) -> None:
        if self.task != "regression":
            raise ValueError("RBFNystroemTeacherFactory currently supports regression")
        if self.n_components < 2 or self.median_sample_size < 2:
            raise ValueError("Nyström component and median sample sizes must be >=2")
        if self.alpha <= 0 or self.prediction_batch_size < 1:
            raise ValueError("invalid Nyström ridge settings")

    def fit(
        self, X: np.ndarray, y: np.ndarray, *, seed: int
    ) -> _RBFNystroemTeacher:
        from scipy.spatial.distance import pdist
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import Ridge

        features, target = _validated_regression_arrays(X, y)
        if self.device.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for RBF prediction but is unavailable")
        x_mean = features.mean(axis=0)
        x_scale = features.std(axis=0)
        x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
        normalized = (features - x_mean) / x_scale
        rng = np.random.default_rng(seed)
        bandwidth_indices = np.sort(
            rng.choice(
                len(normalized),
                min(self.median_sample_size, len(normalized)),
                replace=False,
            )
        )
        distances = pdist(normalized[bandwidth_indices], metric="sqeuclidean")
        positive = distances[np.isfinite(distances) & (distances > 0)]
        if not len(positive):
            raise ValueError("RBF median-distance heuristic has no positive distances")
        median_squared_distance = float(np.median(positive))
        gamma = 1.0 / median_squared_distance
        components = min(self.n_components, len(normalized))
        transform = Nystroem(
            kernel="rbf",
            gamma=gamma,
            n_components=components,
            random_state=seed,
            n_jobs=1,
        )
        features_nystroem = transform.fit_transform(normalized)
        y_mean = float(target.mean())
        y_scale = float(target.std())
        if not np.isfinite(y_scale) or y_scale <= 1e-12:
            y_scale = 1.0
        standardized_target = (target - y_mean) / y_scale
        ridge = Ridge(alpha=self.alpha, fit_intercept=True, solver="cholesky")
        ridge.fit(features_nystroem, standardized_target)
        coefficient = np.asarray(ridge.coef_, dtype=np.float64).reshape(-1)
        normalization = np.asarray(transform.normalization_, dtype=np.float64)
        kernel_weights = normalization @ coefficient
        metadata = {
            "kernel": "rbf",
            "gamma_rule": "inverse_median_nonzero_squared_distance",
            "gamma": gamma,
            "median_squared_distance": median_squared_distance,
            "median_sample_size": int(len(bandwidth_indices)),
            "requested_components": self.n_components,
            "fitted_components": int(components),
            "ridge_alpha": self.alpha,
            "device": self.device,
            "seed": int(seed),
            "silent_fallback": False,
        }
        return _RBFNystroemTeacher(
            x_mean=np.asarray(x_mean, dtype=np.float64),
            x_scale=np.asarray(x_scale, dtype=np.float64),
            components=np.asarray(transform.components_, dtype=np.float64),
            gamma=gamma,
            kernel_weights=kernel_weights,
            intercept=float(ridge.intercept_),
            y_mean=y_mean,
            y_scale=y_scale,
            device=self.device,
            batch_size=self.prediction_batch_size,
            fit_metadata=metadata,
            reference_transform=transform,
            reference_ridge=ridge,
        )
