"""Neural network models for poverty prediction.

This module implements two main architectures:
1. MixtureDensityNetwork (MDN): Predicts full consumption distribution
2. DirectRatePredictor: Directly predicts poverty rates
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path

from config import POVERTY_THRESHOLDS, MODELS_DIR, RANDOM_SEED


class PovertyDataset(Dataset):
    """PyTorch dataset for poverty prediction."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        weights: np.ndarray | None = None
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.weights = torch.FloatTensor(weights) if weights is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            if self.weights is not None:
                return self.X[idx], self.y[idx], self.weights[idx]
            return self.X[idx], self.y[idx]
        return self.X[idx]


class MixtureDensityNetwork(nn.Module):
    """Mixture Density Network for consumption distribution prediction.

    Predicts parameters of a mixture of Gaussians for each household,
    allowing estimation of the full consumption distribution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_components: int = 5,
        dropout: float = 0.2
    ):
        """Initialize MDN.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            n_components: Number of Gaussian mixture components
            dropout: Dropout probability
        """
        super().__init__()
        self.n_components = n_components

        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Output heads for mixture parameters
        self.pi_head = nn.Linear(hidden_dim // 2, n_components)  # Mixture weights
        self.mu_head = nn.Linear(hidden_dim // 2, n_components)  # Means
        self.sigma_head = nn.Linear(hidden_dim // 2, n_components)  # Log-std

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Tuple of (pi, mu, sigma):
                - pi: Mixture weights [batch_size, n_components]
                - mu: Means [batch_size, n_components]
                - sigma: Standard deviations [batch_size, n_components]
        """
        h = self.shared(x)

        # Mixture weights (softmax for proper probabilities)
        pi = F.softmax(self.pi_head(h), dim=-1)

        # Means (softplus to ensure positive values for consumption)
        mu = F.softplus(self.mu_head(h))

        # Standard deviations (softplus + small constant for stability)
        sigma = F.softplus(self.sigma_head(h)) + 1e-4

        return pi, mu, sigma

    def sample(self, x, n_samples: int = 1000):
        """Sample from the predicted distribution.

        Args:
            x: Input features [batch_size, input_dim]
            n_samples: Number of samples per household

        Returns:
            Samples [batch_size, n_samples]
        """
        pi, mu, sigma = self.forward(x)
        batch_size = x.shape[0]

        # Sample component indices
        component_idx = torch.multinomial(pi, n_samples, replacement=True)

        # Sample from each Gaussian
        samples = torch.zeros(batch_size, n_samples, device=x.device)
        for i in range(batch_size):
            for j in range(n_samples):
                k = component_idx[i, j]
                samples[i, j] = torch.normal(mu[i, k], sigma[i, k])

        return F.relu(samples)  # Consumption must be positive

    def predict_mean(self, x):
        """Predict mean consumption (expectation of mixture)."""
        pi, mu, sigma = self.forward(x)
        return (pi * mu).sum(dim=-1)

    def predict_median(self, x, n_samples: int = 1000):
        """Predict median consumption via sampling."""
        samples = self.sample(x, n_samples)
        return torch.median(samples, dim=-1).values


class MDNLoss(nn.Module):
    """Negative log-likelihood loss for Mixture Density Networks."""

    def __init__(self):
        super().__init__()

    def forward(self, pi, mu, sigma, y):
        """Compute NLL loss.

        Args:
            pi: Mixture weights [batch_size, n_components]
            mu: Means [batch_size, n_components]
            sigma: Standard deviations [batch_size, n_components]
            y: Target values [batch_size]

        Returns:
            Negative log-likelihood loss
        """
        y = y.unsqueeze(-1)  # [batch_size, 1]

        # Compute log probabilities for each component
        var = sigma ** 2
        log_probs = -0.5 * (
            torch.log(2 * np.pi * var) +
            (y - mu) ** 2 / var
        )

        # Log-sum-exp over components, weighted by mixture weights
        log_likelihood = torch.logsumexp(
            torch.log(pi + 1e-10) + log_probs, dim=-1
        )

        return -log_likelihood.mean()


class DirectRatePredictor(nn.Module):
    """Neural network that directly predicts poverty rates.

    Instead of predicting consumption, this model directly outputs
    the 19 poverty rates for each survey.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_rates: int = 19,
        dropout: float = 0.3
    ):
        """Initialize rate predictor.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            n_rates: Number of poverty rate thresholds (19)
            dropout: Dropout probability
        """
        super().__init__()
        self.n_rates = n_rates

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_rates),
            nn.Sigmoid(),  # Rates are between 0 and 1
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Predicted rates [batch_size, n_rates]
        """
        rates = self.net(x)

        # Ensure monotonicity: each rate should be >= previous
        # Apply cumulative max along rate dimension
        rates_mono = torch.cummax(rates, dim=-1).values

        return rates_mono


class SurveyRatePredictor(nn.Module):
    """Predict poverty rates at the survey level (aggregated).

    Takes mean features for a survey and predicts its poverty rates.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_rates: int = 19,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_rates = n_rates

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_rates),
            nn.Sigmoid(),
        )

    def forward(self, x):
        rates = self.net(x)
        return torch.cummax(rates, dim=-1).values


class ConsumptionPredictor(nn.Module):
    """Standard neural network for consumption prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """Predict log consumption."""
        return self.net(x).squeeze(-1)

    def predict(self, x):
        """Predict actual consumption (exp of output)."""
        log_pred = self.forward(x)
        return torch.exp(log_pred)


class WeightedMAPELoss(nn.Module):
    """Weighted Mean Absolute Percentage Error loss for poverty rates."""

    def __init__(self, thresholds: list[float] | None = None):
        super().__init__()
        self.thresholds = thresholds or POVERTY_THRESHOLDS

        # Precompute weights: w_t = 1 - |p_t - 0.4|
        # We'll use uniform weights for training since we don't have true rates
        n_rates = len(self.thresholds)
        self.register_buffer("weights", torch.ones(n_rates) / n_rates)

    def forward(self, pred_rates, true_rates):
        """Compute weighted MAPE.

        Args:
            pred_rates: Predicted rates [batch_size, n_rates] or [n_rates]
            true_rates: True rates [batch_size, n_rates] or [n_rates]

        Returns:
            Weighted MAPE loss
        """
        # Avoid division by zero
        true_rates = torch.clamp(true_rates, min=1e-6)

        # MAPE per rate
        mape = torch.abs(pred_rates - true_rates) / true_rates

        # Weighted average
        if mape.dim() == 1:
            return (mape * self.weights).sum()
        else:
            return (mape * self.weights).sum(dim=-1).mean()


class NeuralModelWrapper:
    """Wrapper for PyTorch models to provide sklearn-like interface."""

    def __init__(
        self,
        model_class: type,
        input_dim: int | None = None,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        n_epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        name: str = "neural_model"
    ):
        self.model_class = model_class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.name = name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None
        self.feature_cols = None
        self.feature_mean = None
        self.feature_std = None

    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        if fit:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0) + 1e-8
        return (X - self.feature_mean) / self.feature_std

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_cols: list[str],
        weights: np.ndarray | None = None,
        val_X: pd.DataFrame | None = None,
        val_y: pd.Series | None = None,
        verbose: bool = True
    ):
        """Train the neural network.

        Args:
            X: Training features
            y: Training target
            feature_cols: Feature columns to use
            weights: Sample weights
            val_X: Validation features
            val_y: Validation target
            verbose: Print training progress
        """
        self.feature_cols = feature_cols

        # Prepare data
        X_arr = X[feature_cols].values.astype(np.float32)
        y_arr = np.log1p(y.values.astype(np.float32))  # Log transform

        # Normalize features
        X_arr = self._normalize_features(X_arr, fit=True)

        # Create dataset
        if weights is not None:
            dataset = PovertyDataset(X_arr, y_arr, weights)
        else:
            dataset = PovertyDataset(X_arr, y_arr)

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        # Initialize model
        if self.input_dim is None:
            self.input_dim = X_arr.shape[1]

        self.model = self.model_class(self.input_dim, self.hidden_dim)
        self.model.to(self.device)

        # Loss and optimizer
        if isinstance(self.model, MixtureDensityNetwork):
            criterion = MDNLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        # Training loop
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                if len(batch) == 3:
                    X_batch, y_batch, w_batch = batch
                else:
                    X_batch, y_batch = batch
                    w_batch = None

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()

                if isinstance(self.model, MixtureDensityNetwork):
                    pi, mu, sigma = self.model(X_batch)
                    loss = criterion(pi, mu, sigma, y_batch)
                else:
                    pred = self.model(X_batch)
                    loss = criterion(pred, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Features DataFrame

        Returns:
            Consumption predictions
        """
        self.model.eval()

        X_arr = X[self.feature_cols].values.astype(np.float32)
        X_arr = self._normalize_features(X_arr, fit=False)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_arr).to(self.device)

            if isinstance(self.model, MixtureDensityNetwork):
                pred = self.model.predict_mean(X_tensor)
            else:
                pred = self.model(X_tensor)

            pred = pred.cpu().numpy()

        # Reverse log transform
        return np.expm1(pred)

    def save(self, path: str | None = None):
        """Save model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / f"{self.name}.pkl"

        save_dict = {
            "model_state": self.model.state_dict() if self.model else None,
            "model_class": self.model_class,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "feature_cols": self.feature_cols,
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "name": self.name,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        wrapper = cls(
            model_class=save_dict["model_class"],
            input_dim=save_dict["input_dim"],
            hidden_dim=save_dict["hidden_dim"],
            name=save_dict["name"]
        )

        wrapper.feature_cols = save_dict["feature_cols"]
        wrapper.feature_mean = save_dict["feature_mean"]
        wrapper.feature_std = save_dict["feature_std"]

        if save_dict["model_state"] is not None:
            wrapper.model = save_dict["model_class"](
                save_dict["input_dim"],
                save_dict["hidden_dim"]
            )
            wrapper.model.load_state_dict(save_dict["model_state"])
            wrapper.model.to(wrapper.device)

        return wrapper


def create_neural_models(
    input_dim: int,
    model_types: list[str] | None = None,
    n_seeds: int = 5
) -> list[NeuralModelWrapper]:
    """Create ensemble of neural network models.

    Args:
        input_dim: Number of input features
        model_types: Types of models to create
        n_seeds: Number of models per type

    Returns:
        List of model wrappers
    """
    if model_types is None:
        model_types = ["mdn", "consumption"]

    models = []

    for model_type in model_types:
        for i in range(n_seeds):
            torch.manual_seed(RANDOM_SEED + i)

            if model_type == "mdn":
                model = NeuralModelWrapper(
                    model_class=MixtureDensityNetwork,
                    input_dim=input_dim,
                    name=f"mdn_seed{i}"
                )
            elif model_type == "consumption":
                model = NeuralModelWrapper(
                    model_class=ConsumptionPredictor,
                    input_dim=input_dim,
                    name=f"consumption_nn_seed{i}"
                )
            elif model_type == "direct_rate":
                model = NeuralModelWrapper(
                    model_class=DirectRatePredictor,
                    input_dim=input_dim,
                    name=f"direct_rate_seed{i}"
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            models.append(model)

    return models
