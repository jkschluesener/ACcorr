#!/usr/bin/env python

"""
@license: MIT
@author: Jan K. Schluesener
https://github.com/jkschluesener/accorr
accorr - attenuation-corrected correlations
"""

from typing import Union
import numpy as np
from scipy.stats import t

# add . for package installation
from .models import ComputeInput, ComputeOutput


class accorr:
    """Compute Attenuation-Corrected Correlation Coefficients"""

    # pylint: disable=too-few-public-methods
    # More public methods are currently not necessary

    def __init__(self) -> None:
        """Initialize to compute attenuation-corrected correlations."""
        pass

    def compute(
        self,
        arr_a: np.ndarray,
        arr_b: np.ndarray,
        statistical_threshold: float = 0.05,
        feature_labels_a: Union[None, np.ndarray] = None,
        feature_labels_b: Union[None, np.ndarray] = None,
    ) -> ComputeOutput:
        """
        Compute attenuation-corrected correlations between datasets,
        which isbased on the reliabilities within and across data.

        Args:
            arr_a (np.ndarray): Numpy array of shape (n_features, n_samples).
                n_samples needs to be identical with arr_b.
            arr_b (np.ndarray): Numpy array of shape (n_features, n_samples).
                n_samples needs to be identical with arr_a.
            statistical_threshold (float):
                Significance level (`alpha`) used for a ttest to ensure that
                reliabilities are significantly greater than 0.
            feature_labels_a (Union[None, np.ndarray], optional):
                Feature labels of arr_a as numpy array of shape (n_features,).
                Used for repeated measures. Defaults to None to set default unique labels.
            feature_labels_b (Union[None, np.ndarray], optional):
                Feature labels of arr_b as numpy array of shape (n_features,).
                Used for repeated measures. Defaults to None to set default unique labels.

        Returns:
            ComputeOutput: The output of the computation, which includes the
                attenuation-corrected correlations and other related data.
                The fields are:
                    - reliability_a (float): Description of reliability_a
                    - reliability_b (float): Description of reliability_b
                    - reliability_across (float): Description of reliability_across
                    - corrected_correlation (float): Description of corrected_correlation
                    - pearson (float): Description of pearson
                    - significance_level_a (float): Description of significance_level_a
                    - significance_level_b (float): Description of significance_level_b
                    - pairwise_correlation_a (np.ndarray): Description of pairwise_correlation_a
                    - pairwise_correlation_b (np.ndarray): Description of pairwise_correlation_b
                    - pairwise_correlation_across (np.ndarray): Description of pairwise_correlation_across
        """

        # pylint: disable=too-many-arguments

        # Data Handling
        # If labels are not none, hand them to ComputeInput, otherwise do not hand them over
        if feature_labels_a is not None and feature_labels_b is not None:
            input_data = ComputeInput(
                arr_a=arr_a,
                arr_b=arr_b,
                feature_labels_a=feature_labels_a,
                feature_labels_b=feature_labels_b,
                statistical_threshold=statistical_threshold,
            )
        else:
            input_data = ComputeInput(
                arr_a=arr_a,
                arr_b=arr_b,
                statistical_threshold=statistical_threshold,
            )

        results = self._compute(input_data)

        return results

    ####################
    # Computation
    ####################
    def _compute(self, data: ComputeInput) -> None:
        """
        Compute the reliability, statistical testing, averaging, and accorr values based on the input data.

        Args:
            data (ComputeInput): The input data containing arrays and labels.

        Returns:
            ComputeOutput: The output containing the computed values.
        """
        # Compute reliabilities
        within_a, within_b, across = self._reliability(data.arr_a, data.arr_b)

        # Satistical testing if reliabilities are significantly greater than 0
        within_a_pvalue = self.ttest(within_a, data.feature_labels_a)
        within_b_pvalue = self.ttest(within_b, data.feature_labels_b)

        # Averaging, accounting for the possibility of repeated measures
        mean_a = self._repeated_mean(
            within_a, data.feature_labels_a, data.feature_labels_a
        )
        mean_b = self._repeated_mean(
            within_b, data.feature_labels_b, data.feature_labels_b
        )
        mean_across = self._repeated_mean(
            across, data.feature_labels_a, data.feature_labels_b
        )

        # compute accorr only if both reliabilities are significantly greater than 0
        if (within_a_pvalue <= data.statistical_threshold) & (
            within_b_pvalue <= data.statistical_threshold
        ):
            accorr = self._rac(mean_across, mean_a, mean_b)
        else:
            accorr = np.nan

        pears = np.corrcoef(
            np.mean(data.arr_a, axis=0),
            np.mean(data.arr_b, axis=0),
            rowvar=False,
        )[0, 1]

        out = ComputeOutput(
            reliability_a=mean_a,
            reliability_b=mean_b,
            reliability_across=mean_across,
            corrected_correlation=accorr,
            pearson=pears,
            significance_level_a=within_a_pvalue,
            significance_level_b=within_b_pvalue,
            pairwise_correlation_a=np.tanh(within_a),
            pairwise_correlation_b=np.tanh(within_b),
            pairwise_correlation_across=np.tanh(across),
            feature_labels_a=data.feature_labels_a,
            feature_labels_b=data.feature_labels_b
        )

        return out

    @staticmethod
    def _reliability(
        arr_a: np.ndarray, arr_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the reliability measures between two arrays.

        Parameters:
        arr_a (np.ndarray): First array.
        arr_b (np.ndarray): Second array.

        Returns:
        tuple[np.ndarray]: A tuple containing three reliability measures:
            - within_a: Reliability measure within arr_a.
            - within_b: Reliability measure within arr_b.
            - across: Reliability measure across arr_a and arr_b.
        """

        n_a = arr_a.shape[0]
        n_b = arr_b.shape[0]

        data_stack = np.vstack((arr_a, arr_b))
        pearson = np.corrcoef(data_stack, rowvar=True)

        # If arr_a is arr_b, set the diagonal to NaN
        np.fill_diagonal(pearson, np.nan)

        # -1, 1 are not allowed for the Fisher Z transformation
        pearson[np.isclose(pearson, -1)] = -0.999999
        pearson[np.isclose(pearson, 1)] = 0.999999

        # Do the Fisher Z transformation - inverse hyperbolic tangent
        pearson = np.arctanh(pearson)

        within_a = pearson[:n_a, :n_a]  # Upper Left
        within_b = pearson[n_a:, n_a:]  # Lower Right
        across = pearson[:n_a, n_a:]  # Upper Right

        return within_a, within_b, across

    @staticmethod
    def _rac(rel_ab: float, rel_a: float, rel_b: float) -> float:
        """
        Compute the attenuation-corrected correlation from values.

        Args:
            rel_ab (float): Reliability across arrays.
            rel_a (float): Reliability within array a.
            rel_b (float): Reliability within array b.

        Returns:
            float: Attenuation-corrected correlation between array a and array b.
        """

        prod = rel_a * rel_b

        # This is to avoid potentially raising a warning in np.sqrt()
        if prod <= 0:
            return np.nan

        r_ac = rel_ab / np.sqrt(prod)

        return r_ac

    @staticmethod
    def _repeated_mean(
        arr: np.ndarray, labels0: np.ndarray, labels1: np.ndarray
    ) -> float:
        """
        Compute the repeated mean of an array based on two sets of labels.

        Args:
            arr (np.ndarray): The input array.
            labels0 (np.ndarray): The first set of labels, describing the rows of the array.
            labels1 (np.ndarray): The second set of labels, describing the columns of the array.

        Returns:
            float: The repeated mean of the array.
        """

        if (
            np.unique(labels0).shape[0]
            == labels0.shape[0] & np.unique(labels1).shape[0]
            == labels1.shape[0]
        ):

            out = np.nanmean(arr)
            out = np.tanh(out)

            return out

        # Flatten the arrays and labels
        arr_flat = arr.flatten()
        labels0_flat = np.repeat(labels0, arr.shape[1])
        labels1_flat = np.tile(labels1, arr.shape[0])

        arr_valid_mask = ~np.isnan(arr_flat)

        # Compute the sum and count for each label
        sum0 = np.bincount(
            labels0_flat[arr_valid_mask], weights=arr_flat[arr_valid_mask]
        )
        count0 = np.bincount(labels0_flat).astype(float)
        count0[count0 == 0] = np.nan  # Avoid division by zero
        mean0 = sum0 / count0

        sum1 = np.bincount(
            labels1_flat[arr_valid_mask], weights=arr_flat[arr_valid_mask]
        )
        count1 = np.bincount(labels1_flat).astype(float)
        count1[count1 == 0] = np.nan  # Avoid division by zero
        mean1 = sum1 / count1

        # Compute the overall mean
        out = np.nanmean(np.concatenate([mean0, mean1]))

        out = np.tanh(out)

        return out

    def ttest(self, array: np.ndarray, labels: np.ndarray) -> float:
        """
        Perform a t-test on the given array of values.
        This function is used to test if the reliabilities are significantly greater than 0.
        It selects only the upper triangular part of the array (excluding the diagonal) and removes NaNs.

        Args:
            array (np.ndarray): The input array of values.
            labels (np.ndarray): The labels corresponding to each row/column.

        Returns:
            float: The p-value resulting from the t-test.
        """

        # Flat triu versions without NaNs
        arr_flat = array[np.triu_indices_from(array)]
        arr_flat = arr_flat[~np.isnan(arr_flat)]

        dof = np.unique(labels).shape[0] - 1

        pvalue = self._ttest(arr_flat, dof)

        return pvalue

    @staticmethod
    def _ttest(x: np.ndarray, dof: int) -> float:
        """
        Perform a one-sample t-test.

        Args:
            x (np.ndarray): Input array of data.
            dof (int): Degrees of freedom.

        Returns:
            float: The p-value from the t-test.
        """

        # mean null hypothesis is 0, one-sample t-test
        Y: float = 0.0

        # get population info
        n: int = x.shape[0]
        mean: float = np.mean(x)  # sample mean
        std: float = np.std(
            x, ddof=1
        )  # set Bessel's correction to the actual number of degrees of freedom
        sem: float = std / np.sqrt(n)  # standard error of the mean

        # compute the t-score
        tscore = (mean - Y) / sem

        # get the p-value from the cumulative distribution
        pvalue = 1 - t.cdf(tscore, dof)  # right-tailed
        # pvalue = t.cdf(tscore, dof) # left-tailed
        # pvalue = 2 * t.cdf(-np.abs(tscore), dof) # two-tailed

        return pvalue
