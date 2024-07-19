from pydantic import BaseModel, Field, field_validator
from typing import Tuple
import numpy as np


# switch validator to field_validator
class ComputeInput(BaseModel):
    """
    Represents the input data for the computation.

    Attributes:
        arr_a (np.ndarray): Input array a.
        arr_b (np.ndarray): Input array b.
        statistical_threshold (float): Alpha value.
        feature_labels_a (np.ndarray): Labels for array a.
        feature_labels_b (np.ndarray): Labels for array b.
    """

    def __init__(self, **data):
        super().__init__(**data)

        n_none = sum([self.feature_labels_a is None, self.feature_labels_b is None])
        if n_none == 2:
            self.feature_labels_a = np.arange(self.arr_a.shape[0])
            self.feature_labels_b = np.arange(
                self.arr_a.shape[0], self.arr_a.shape[0] + self.arr_b.shape[0]
            )
        elif n_none == 1:
            raise ValueError("Both or none of the labels must be None")

        self.sort_by_labels()

    model_config = {"arbitrary_types_allowed": True}

    arr_a: np.ndarray = Field(..., description="Input array a")
    arr_b: np.ndarray = Field(..., description="Input array b")
    statistical_threshold: float = Field(
        ..., description="Alpha value for statistical testing"
    )
    feature_labels_a: np.ndarray = Field(None, description="Labels for array a")
    feature_labels_b: np.ndarray = Field(None, description="Labels for array b")

    @field_validator("arr_a", "arr_b")
    def validate_all(cls, to_check: np.ndarray) -> np.ndarray:
        if not np.isfinite(to_check).all():
            raise ValueError("Input arrays must contain only finite values")
        if to_check.shape[0] < 1:
            raise ValueError("Input arrays must contain at least 2 samples")
        return to_check

    # @validator("feature_labels_a", "feature_labels_b")
    # def validate_all(cls, to_check: np.ndarray) -> np.ndarray:
    #     if not np.isfinite(to_check).all():
    #         raise ValueError("Input arrays must contain only finite values")
    #     if to_check.shape[0] < 1:
    #         raise ValueError("Input arrays must contain at least 2 samples")
    #     return to_check

    @field_validator("feature_labels_a", "feature_labels_b")
    def validate_labels(cls, to_check: np.ndarray) -> np.ndarray:
        if np.unique(to_check).size <= 1:
            raise ValueError("Labels must contain more than one unique value")
        return to_check

    @field_validator("statistical_threshold")
    def validate_alpha(cls, to_check: float) -> float:
        if not 0 <= to_check <= 1:
            raise ValueError(
                "Statistical Threshold must be in range [0, 1], not NaN, not Inf"
            )
        return to_check

    def sort_by_labels(self):
        if not self.feature_labels_a.shape[0] == self.arr_a.shape[0]:
            raise ValueError(
                f"n_features in arr_a needs to match the number of labels in feature_labels_a"
            )
        if not self.feature_labels_b.shape[0] == self.arr_b.shape[0]:
            raise ValueError(
                f"n_features in arr_b needs to match the number of labels in feature_labels_b"
            )
        self.arr_a, self.feature_labels_a = self._sort_by_labels(
            self.arr_a, self.feature_labels_a
        )
        self.arr_b, self.feature_labels_b = self._sort_by_labels(
            self.arr_b, self.feature_labels_b
        )

    @staticmethod
    def _sort_by_labels(
        arr: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sorts the given array and labels based on the labels.

        Parameters:
            arr (numpy.ndarray): The array to be sorted.
            labels (numpy.ndarray): The labels used for sorting.

        Returns:
            tuple: A tuple containing the sorted array and labels.
        """
        sort_indices = np.argsort(labels)
        labels = labels[sort_indices]
        arr = arr[sort_indices, :]

        return arr, labels


class ComputeOutput(BaseModel):
    """
    Represents the output model for ACcorr calculations.

    Attributes:
        rel_a (float): The relative value of a.
        rel_b (float): The relative value of b.
        rel_ab (float): The relative value of ab.
        accorr (float): The ACcorr value.
        pearson (float): The Pearson correlation coefficient.
        sig_a (float): The significance value of a.
        sig_b (float): The significance value of b.
        pairwise_correlation_a (np.ndarray): Pairwise correlation of a.
        pairwise_correlation_b (np.ndarray): Pairwise correlation of b.
        pairwise_correlation_across (np.ndarray): Pairwise correlation across a and b.
    """

    model_config = {"arbitrary_types_allowed": True}

    reliability_a: float = Field(..., description="The reliability value of a")
    reliability_b: float = Field(..., description="The reliability value of b")
    reliability_across: float = Field(
        ..., description="The reliability between a and b"
    )
    corrected_correlation: float = Field(
        ..., description="Attenuation-corrected correlation coefficient"
    )
    pearson: float = Field(..., description="Pearson correlation coefficient")
    significance_level_a: float = Field(..., description="Significance value of a")
    significance_level_b: float = Field(..., description="Significance value of b")
    pairwise_correlation_a: np.ndarray = Field(
        ..., description="Pairwise correlation of a"
    )
    pairwise_correlation_b: np.ndarray = Field(
        ..., description="Pairwise correlation of b"
    )
    pairwise_correlation_across: np.ndarray = Field(
        ..., description="Pairwise correlation across a and b"
    )
