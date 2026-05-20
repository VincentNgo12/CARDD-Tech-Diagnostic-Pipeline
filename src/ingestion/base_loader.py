import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseDataLoader(ABC):
    """
    Abstract Base Class defining the contract for data ingestion modules.

    This class ensures that all data loaders, regardless of the hardware source
    or data format, provide a consistent interface for the machine learning
    pipeline. This decoupling allows the models (scikit-learn or PyTorch) to 
    remain agnostic of the underlying data source.

    Attributes:
        data_path (str): Path to the raw data file.
    """

    def __init__(self, data_path: str) -> None:
        """
        Initializes the data loader with the source path.

        Args:
            data_path: Path to the dataset on disk.
        """
        self.data_path = data_path

    @abstractmethod
    def load_and_clean(self) -> None:
        """
        Reads raw data from disk and performs initial cleaning.

        This method should handle file I/O, missing value treatment (e.g.,
        imputation or removal), and any domain-specific data filtering.
        The result should be stored as internal state (e.g., in a DataFrame).

        Raises:
            FileNotFoundError: If the data_path does not exist.
            ValueError: If the data format is invalid or corrupted.
        """
        pass

    @abstractmethod
    def get_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the dataset as NumPy arrays for scikit-learn compatibility.

        Returns:
            A tuple containing (X, y), where:
                X: Feature matrix of shape (n_samples, n_features).
                y: Label array of shape (n_samples,).
        """
        pass

    @abstractmethod
    def get_tensors(self, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the dataset as PyTorch Tensors for deep learning models.

        Args:
            device: The target device for the tensors ("cpu" or "cuda").

        Returns:
            A tuple containing (X_tensor, y_tensor) placed on the specified device.
        """
        pass
