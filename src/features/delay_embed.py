import numpy as np
import nolds
from scipy.signal import correlate
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, List, Tuple


class PhaseSpaceEmbedder(BaseEstimator, TransformerMixin):
    """
    Phase Space Reconstruction (PSR) Transformer using Takens' Embedding Theorem.
    
    This class performs Phase Space Reconstruction on time-series data, effectively 
    mapping a 1D or multi-dimensional dynamical system into a higher-dimensional 
    topological manifold that preserves the system's geometric invariants 
    (Lyapunov exponents, fractal dimension, etc.).
    
    According to Takens' Theorem, for a smooth dynamical system with an attractor 
    of dimension D, an embedding dimension m > 2D is sufficient to reconstruct 
    a manifold that is diffeomorphic to the original state space.

    Attributes:
        m (int): Embedding dimension. If None, estimated during fit.
        tau (int): Time delay (lag). If None, estimated during fit.
        fnn_threshold (float): Threshold for the False Nearest Neighbors algorithm.
    """

    def __init__(
        self, 
        m: Optional[int] = None, 
        tau: Optional[int] = None, 
        fnn_threshold: float = 0.1
    ):
        """
        Initializes the PhaseSpaceEmbedder.

        Args:
            m (int, optional): The embedding dimension. If None, it will be 
                automatically estimated using False Nearest Neighbors.
            tau (int, optional): The time delay. If None, it will be 
                automatically estimated using the first zero-crossing or 
                local minimum of the Autocorrelation Function (ACF).
            fnn_threshold (float): Convergence threshold for FNN estimation. 
                Defaults to 0.1.
        """
        self.m = m
        self.tau = tau
        self.fnn_threshold = fnn_threshold
        self._fitted_m: Optional[int] = None
        self._fitted_tau: Optional[int] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PhaseSpaceEmbedder":
        """
        Estimates optimal delay (tau) and embedding dimension (m) if not provided.

        Args:
            X (np.ndarray): Time-series data of shape (samples,) or (samples, channels).
            y (ignored): Included for scikit-learn compatibility.

        Returns:
            PhaseSpaceEmbedder: The fitted instance.
        """
        X = self._validate_input(X)
        
        # We use the first channel to estimate global parameters for multi-channel data
        # as the underlying dynamics are typically coupled.
        reference_signal = X[:, 0] if X.ndim == 2 else X

        # 1. Automated Time Delay (tau) Estimation
        # Physics intuition: tau should be large enough to make the coordinates 
        # statistically independent but small enough that they are still 
        # dynamically related. The first zero-crossing of ACF is a standard heuristic.
        if self.tau is None:
            self._fitted_tau = self._estimate_tau(reference_signal)
        else:
            self._fitted_tau = self.tau

        # 2. Automated Embedding Dimension (m) Estimation
        # Physics intuition: We increase m until the percentage of "false" neighbors 
        # (points that are close in dimension m but far in m+1) drops below a threshold.
        # This signifies that the attractor has been fully "unfolded".
        if self.m is None:
            self._fitted_m = self._estimate_m(reference_signal, self._fitted_tau)
        else:
            self._fitted_m = self.m

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Folds the input time-series into a multi-dimensional phase space trajectory.

        Args:
            X (np.ndarray): Time-series data of shape (samples,) or (samples, channels).

        Returns:
            np.ndarray: Reconstructed phase space of shape 
                (samples - (m-1)*tau, m * channels).
        """
        if self._fitted_m is None or self._fitted_tau is None:
            raise ValueError("Transformer must be fitted before calling transform.")

        X = self._validate_input(X)
        
        if X.ndim == 1:
            return self._embed_signal(X, self._fitted_m, self._fitted_tau)
        
        # For multi-channel data, embed each channel and concatenate
        embedded_channels = []
        for i in range(X.shape[1]):
            embedded = self._embed_signal(X[:, i], self._fitted_m, self._fitted_tau)
            embedded_channels.append(embedded)
        
        return np.hstack(embedded_channels)

    def _estimate_tau(self, signal: np.ndarray) -> int:
        """
        Estimates the optimal time delay tau using the Autocorrelation Function.
        
        Locates the first zero-crossing or the first local minimum of the ACF.
        """
        n = len(signal)
        # Normalize signal to zero mean for ACF
        s_norm = signal - np.mean(signal)
        
        # Compute ACF using scipy.signal.correlate (fast FFT-based)
        # We only need the second half of the correlation result
        acf = correlate(s_norm, s_norm, mode='full')[n-1:]
        acf /= acf[0]  # Normalize so acf[0] == 1.0

        # Find first zero crossing
        zero_crossings = np.where(np.diff(np.sign(acf)))[0]
        if len(zero_crossings) > 0:
            tau_z = zero_crossings[0]
        else:
            tau_z = n # Fallback

        # Find first local minimum
        local_minima = np.where(np.diff(np.sign(np.diff(acf))) > 0)[0] + 1
        if len(local_minima) > 0:
            tau_m = local_minima[0]
        else:
            tau_m = n # Fallback

        tau = min(tau_z, tau_m)
        
        # Basic structural validation: tau must be at least 1 and 
        # shouldn't consume too much of the signal.
        return max(1, int(min(tau, n // 10)))

    def _estimate_m(self, signal: np.ndarray, tau: int) -> int:
        """
        Estimates the embedding dimension m using False Nearest Neighbors (FNN).
        
        Utilizes nolds.afn (Algorithm for False Neighbors) to find the dimension 
        where the geometric structure of the attractor is preserved without 
        self-intersections.
        """
        # nolds.afn returns (E1, E2). E1 is the ratio of average distances.
        # It approaches 1 as m reaches the sufficient embedding dimension.
        # We look for where E1 stops increasing significantly.
        max_m = 10
        try:
            # We use afn which calculates the ratio of distances in m+1 vs m
            # E1(m) = mean(dist_{m+1}) / mean(dist_{m})
            # For a deterministic system, E1 saturates to 1 at the correct m.
            e1, e2 = nolds.afn(signal, tau=tau, dim=range(1, max_m + 1))
            
            # Find first index where e1 is close to 1
            for i, val in enumerate(e1):
                if val > (1.0 - self.fnn_threshold):
                    return i + 1
            return max_m
        except Exception:
            # Fallback to a sensible default if AFN fails
            return 3

    def _embed_signal(self, signal: np.ndarray, m: int, tau: int) -> np.ndarray:
        """
        Core logic for Delay Embedding on a single signal.
        
        Creates the trajectory matrix:
        [ x(t), x(t+tau), x(t+2*tau), ..., x(t+(m-1)*tau) ]
        """
        n = len(signal)
        if (m - 1) * tau >= n:
            raise ValueError(
                f"Signal length ({n}) is too short for m={m} and tau={tau}. "
                f"Required length: > (m-1)*tau = {(m - 1) * tau}"
            )

        # Number of rows in the embedded matrix
        n_rows = n - (m - 1) * tau
        
        # Efficiently create the delay matrix using fancy indexing or strides
        indices = np.arange(n_rows)[:, None] + np.arange(m) * tau
        return signal[indices]

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Ensures input is a NumPy array and handles dimensionality checks.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if X.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got {X.ndim}D.")
        
        if X.size == 0:
            raise ValueError("Input signal is empty.")
            
        return X

    @property
    def fitted_params_(self) -> Tuple[int, int]:
        """Returns the estimated (m, tau) after fitting."""
        return self._fitted_m, self._fitted_tau
