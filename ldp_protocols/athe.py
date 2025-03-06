import numpy as np
from numba import jit
from scipy.special import loggamma
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
from scipy.special import comb

@jit(nopython=True)
def he_obfuscate(input_data: int, k: int, epsilon: float) -> np.ndarray:
    """
    Obfuscate the input data using the Histogram Encoding (HE) protocol.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be an integer >= 2.
    epsilon : float
        The privacy budget for the LDP mechanism. Must be a positive value.

    Returns
    -------
    np.ndarray
        A numpy array of size `k` representing the unary encoded input with added Laplace noise.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")
    
    # Unary encode the input
    input_ue_data = np.zeros(k)
    input_ue_data[input_data] = 1.0

    # Add Laplace noise
    return input_ue_data + np.random.laplace(loc=0.0, scale=2 / epsilon, size=k)

@jit(nopython=True)
def attack_the(ss_the, k):
    """
    Perform a privacy attack on an obfuscated vector generated using the Thresholding Histogram Encoding (THE) protocol.

    This attack attempts to infer the true input value by selecting indices where the obfuscated values
    exceed the threshold. If no values exceed the threshold, a random guess is made.

    Parameters
    ----------
    ss_the : np.ndarray
        An obfuscated vector generated using THE, which includes noisy Laplace values.
    k : int
        The size of the domain (number of possible values).

    Returns
    -------
    int
        The inferred true value. If no values exceed the threshold, a random value in the range `[0, k-1]` is returned.
    """

    if sum(ss_the) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(ss_the)

class AdaptiveThresholdingHistogramEncoding:
    def __init__(self, k: int, epsilon: float, w_asr: float = 0.5, w_mse: float = 0.5):
        """
        Initialize the Adaptive Thresholding Histogram Encoding (Adaptive THE) protocol.

        Parameters
        ----------
        k : int
            The size of the domain (number of possible values). Must be an integer >= 2.
        epsilon : float
            The privacy budget, which determines the level of privacy guarantee. Must be positive.
        w_asr : float, optional
            Weight given to the Adversarial Success Rate (ASR) in the objective function. Default is 0.5.
        w_mse : float, optional
            Weight given to the MSE in the objective function. Default is 0.5.

        Raises
        ------
        ValueError
            If `k` is not >= 2, `epsilon` is not positive, or the weights are invalid.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")
        if not (0 <= w_asr <= 1) or not (0 <= w_mse <= 1):
            raise ValueError("Weights must be between 0 and 1.")
        
        # Normalize the weights so that their sum is 1
        total_weight = w_asr + w_mse
        self.w_asr = w_asr / total_weight
        self.w_mse = w_mse / total_weight
        self.k = k
        self.epsilon = epsilon
        self.threshold = self.optimize_parameters()
        self.p = 1 - 0.5 * np.exp(self.epsilon*(self.threshold - 1)/2)
        self.q = 0.5 * np.exp(-self.epsilon*self.threshold/2)

    def get_parameter_range(self) -> np.ndarray:
        """
        Generate a range of threshold values for optimization.

        Returns
        -------
        np.ndarray
            A numpy array of threshold values in the range [0.5, 1].
        """
        
        return np.linspace(0.5, 1.0, 100)

    def optimize_parameters(self) -> float:
        """
        Optimize the threshold value to achieve a balance between MSE and ASR.

        This method performs a grid-search over a range of possible threshold values and selects the one
        that minimizes a weighted combination of the MSE and ASR.

        Returns
        -------
        float
            The optimized threshold value.
        """

        # Define range of threshold values to search over
        thresholds = self.get_parameter_range()

        # Perform grid search to find the best threshold
        best_threshold = 0.5
        best_obj_value = float('inf')

        for tresh in thresholds:
            asr = self.get_asr(tresh)
            mse = self.get_mse(tresh)
            obj_value = self.w_asr * asr + self.w_mse * mse
            if obj_value < best_obj_value:
                best_threshold = tresh
                best_obj_value = obj_value

        return best_threshold

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the Adaptive THE mechanism.

        Parameters
        ----------
        input_data : int
            The user's true input value. Must be in the range [0, k-1].

        Returns
        -------
        np.ndarray
            An array of indices where the obfuscated vector exceeds the threshold.
        """
        
        # Apply thresholding
        return np.where(he_obfuscate(input_data, self.k, self.epsilon) > self.threshold)[0]
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Thresholding Histogram Encoding (ATHE) mechanism.

        This method applies unbiased frequency estimation to recover approximate frequencies of values 
        in the domain `[0, k-1]`. The method uses thresholded noisy reports and corrects for the perturbation 
        introduced by the ATHE mechanism using `p` (true value probability) and `q` (false value probability).

        Parameters
        ----------
        noisy_reports : list of int
            A list of noisy reports collected from users. Each report corresponds to a value that exceeded 
            the adaptive threshold after Laplace noise was added in the ATHE mechanism.

        Returns
        -------
        np.ndarray
            An array of estimated frequencies for each value in the domain.
            The output array has size `k` and sums to 1.

        Raises
        ------
        ValueError
            If `noisy_reports` is empty.
        """
        n = len(noisy_reports)  # Number of reports
        if n == 0:
            raise ValueError("Noisy reports cannot be empty.")
        
        # Count the occurrences of each value in the noisy reports
        support_counts = np.zeros(self.k)
        for report in noisy_reports:
            support_counts[report] += 1

        # Unbiased frequency estimation
        freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))

        # Ensure non-negative estimates and normalize
        return np.maximum(freq_estimates, 0) / np.sum(np.maximum(freq_estimates, 0))
    
    def attack(self, obfuscated_vec: np.ndarray) -> int:
        """
        Perform a privacy attack on an obfuscated vector generated using the Adaptive THE mechanism.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated vector of size `k`, generated using the Adaptive THE mechanism.

        Returns
        -------
        int
            The inferred true value of the input. If no values exceed the threshold, a random value 
            in the range `[0, k-1]` is returned.
        """
        
        return attack_the(obfuscated_vec, self.k)

    def get_mse(self, threshold: float = None, n: int = 1) -> float:
        """
        Compute the MSE of the Adaptive THE mechanism.

        Parameters
        ----------
        threshold : float, optional
            The threshold value to be used for the MSE calculation.
            If None, the optimized threshold value will be used.

        Returns
        -------
        float
            The MSE of the Adaptive THE mechanism.
        """
        tresh = threshold if threshold is not None else self.threshold
        return (2 * np.exp(self.epsilon * tresh / 2) - 1) / (n * (1 + np.exp(self.epsilon * (tresh - 0.5)) - 2 * np.exp(self.epsilon * tresh / 2))**2)

    def get_asr(self, threshold: float = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the Adaptive THE mechanism.

        Parameters
        ----------
        threshold : float, optional
            The threshold value to be used for the ASR calculation.
            If None, the optimized threshold value will be used.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR) of the Adaptive THE mechanism.
        """
        tresh = threshold if threshold is not None else self.threshold

        # Dynamically calculate p and q for the current tresh
        p = 1 - 0.5 * np.exp(self.epsilon * (tresh - 1) / 2)
        q = 0.5 * np.exp(-self.epsilon * tresh / 2)

        term1 = (1 - p) * (1 - q) ** (self.k - 1) * (1 / self.k)
        term2 = 0
        for m in range(1, self.k + 1):
            try:
                # Calculate comb(self.k - 1, m - 1) in log space
                log_comb = loggamma(self.k) - loggamma(m) - loggamma(self.k - m + 1)
                comb_value = np.exp(log_comb)

                # If comb_value is too large, skip or approximate
                if np.isinf(comb_value) or comb_value > 1e308:
                    continue

                term2 += (1 / m) * comb_value * p * (q ** (m - 1)) * ((1 - q) ** (self.k - m))
            
            except OverflowError:
                # Skip this value if an OverflowError occurs
                continue

        # Final ASR calculation
        asr = term1 + term2
        if np.isinf(asr) or np.isnan(asr):
            asr = 0  # Handle overflow/numerical issues by setting ASR to a valid fallback 

        return asr

    def plot_objective_function(self) -> None:
        """
        Plot the objective function over a range of threshold values.

        This method visualizes the relationship between the threshold value and the objective function,
        which is a combination of ASR and MSE.
        """
        thresholds = self.get_parameter_range()
        objective_values = []

        for tresh in thresholds:
            asr = self.get_asr(tresh)
            mse = self.get_mse(tresh)
            objective_value = self.w_asr * asr + self.w_mse * mse
            objective_values.append(objective_value)

        plt.plot(thresholds, objective_values, marker='o')
        plt.axvline(x=self.threshold, color='r', linestyle='--', label=f'Optimal Threshold: {self.threshold}')
        plt.xlabel('Threshold')
        plt.ylabel('Objective Function Value')
        plt.title('Objective Function vs. Threshold')
        plt.grid(True)
        plt.legend()
        plt.show()
