import numpy as np
from numba import jit
from scipy.optimize import minimize_scalar
from scipy.special import loggamma
from scipy.stats import laplace
from scipy.integrate import quad
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")

@jit(nopython=True)
def get_opt_tresh(threshold: float, epsilon: float) -> float:
    """
    Compute the optimal threshold for the Thresholded Histogram Encoding (THE) mechanism.

    Parameters
    ----------
    threshold : float
        The threshold value used in the computation.
    epsilon : float
        Privacy budget for the LDP mechanism.

    Returns
    -------
    float
        The optimal threshold value for the THE mechanism.
    """
    return (2 * (np.exp(epsilon * threshold / 2)) - 1) / (1 + (np.exp(epsilon * (threshold - 1 / 2))) - 2 * (np.exp(epsilon * threshold / 2)))**2

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
    if len(ss_the) == 0:
        return np.random.randint(k)
    else:
        return np.random.choice(ss_the)

class HistogramEncoding:
    def __init__(self, k: int, epsilon: float, thresholding: bool = True):
        """
        Initialize the Histogram Encoding (HE) protocol with domain size k and privacy parameter epsilon.

        Parameters
        ----------
        k : int
            Attribute's domain size. Must be an integer greater than or equal to 2.
        epsilon : float
            Privacy guarantee. Must be a positive numerical value.
        thresholding : bool, optional
            Whether to use thresholding for the mechanism. Default is True.

        Raises
        ------
        ValueError
            If k is not an integer >= 2 or if epsilon is not greater than 0.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")
        self.k = k
        self.epsilon = epsilon
        self.thresholding = thresholding

        # Precompute the optimal threshold
        if self.thresholding:
            self.threshold = minimize_scalar(get_opt_tresh, bounds=[0.5, 1], method='bounded', args=(epsilon)).x
            self.p = 1 - 0.5 * np.exp(self.epsilon*(self.threshold - 1)/2)
            self.q = 0.5 * np.exp(-self.epsilon*self.threshold/2)

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the Histogram Encoding (HE) mechanism.

        Parameters
        ----------
        input_data : int
            The user's true value to be obfuscated. Must be in the range [0, k-1].

        Returns
        -------
        numpy.ndarray or numpy.int64
            If thresholding is enabled, returns the indices of the sanitized vector above the threshold.
            If thresholding is disabled, returns the sanitized vector with Laplace noise.
        """

        # Add Laplace noise
        obfuscated_vec = he_obfuscate(input_data, self.k, self.epsilon)

        if self.thresholding:
            # Apply thresholding
            return np.where(obfuscated_vec > self.threshold)[0]
        
        return obfuscated_vec
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Histogram Encoding (HE) mechanism.

        This method applies unbiased estimation to recover approximate frequencies of values 
        in the domain. It handles two configurations:
        - With thresholding: Uses `p` and `q` to adjust for perturbation in the thresholded reports.
        - Without thresholding: Uses the summed noisy vectors directly for estimation.

        Parameters
        ----------
        noisy_reports : list of int or list of np.ndarray
            A list of noisy reports collected from users.
            - If thresholding is enabled, each report is a binary vector.
            - If thresholding is disabled, each report is a vector of noisy values with Laplace noise added.

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
        
        if self.thresholding:
            # Count the occurrences of each value in the noisy reports
            support_counts = np.zeros(self.k)
            for report in noisy_reports:
                support_counts[report] += 1

            # Unbiased frequency estimation
            freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))

        else:
            # Sum the noisy vectors directly
            freq_estimates = sum(noisy_reports)

        # Ensure non-negative estimates and normalize
        return np.maximum(freq_estimates, 0) / np.sum(np.maximum(freq_estimates, 0))
    
    def attack(self, obfuscated_vec: np.ndarray) -> int:
        """
        Perform a privacy attack on an obfuscated vector generated using the Histogram Encoding (HE) mechanism.

        This method infers the true input value based on the obfuscated vector:
        - For Thresholded Histogram Encoding (THE), the attack identifies indices where the obfuscated 
        values exceed the threshold and randomly selects one. If no values exceed the threshold, 
        a random guess is made.
        - For Standard Histogram Encoding (SHE), the attack predicts the index with the highest 
        value in the obfuscated vector, assuming it corresponds to the true input.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated vector of size `k`, generated using the HE mechanism. 

        Returns
        -------
        int
            The inferred true value of the input. 
            - For THE: If no values exceed the threshold, a random guess in the range `[0, k-1]` is returned. 
            - For SHE: The index with the maximum value in the obfuscated vector is returned.
        """
        
        if self.thresholding:
            # Attack for THE: Use threshold-based attack
            return attack_the(obfuscated_vec, self.k)
        
        # Attack for SHE: Predict the index with the maximum value
        return np.argmax(obfuscated_vec)

    def get_mse(self, n: int = 1) -> float:
        """
        Compute the MSE of the Thresholded Histogram Encoding (THE) mechanism.

        Returns
        -------
        float
            The MSE of the THE mechanism.
        """
        if self.thresholding:
            return self.q * (1 - self.q) / (n * (self.p - self.q)**2)
        
        return 8 / (n * self.epsilon**2)
   
    def get_asr(self) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the Thresholded Histogram Encoding (THE) mechanism.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR) of the THE mechanism.
        """
        if self.thresholding:

            term1 = (1 - self.p) * (1 - self.q) ** (self.k - 1) * (1 / self.k)
            term2 = 0
            for m in range(1, self.k + 1):
                try:
                    # Calculate comb(self.k - 1, m - 1) in log space
                    log_comb = loggamma(self.k) - loggamma(m) - loggamma(self.k - m + 1)
                    comb_value = np.exp(log_comb)

                    # If comb_value is too large, skip or approximate
                    if np.isinf(comb_value) or comb_value > 1e308:
                        continue

                    term2 += (1 / m) * comb_value * self.p * (self.q ** (m - 1)) * ((1 - self.q) ** (self.k - m))
                
                except OverflowError:
                    # Skip this value if an OverflowError occurs
                    continue

            # Final ASR calculation
            asr = term1 + term2
            if np.isinf(asr) or np.isnan(asr):
                asr = 0  # Handle overflow/numerical issues by setting ASR to a valid fallback        

        else:
            # Numerically compute the expected ASR using integration
            scale = 2 / self.epsilon  # Scale parameter

            # Define the CDF and PDF of Z ~ Laplace(0, b)
            F_Z = lambda z: laplace.cdf(z, loc=0, scale=scale)
            f_Z = lambda z: laplace.pdf(z, loc=0, scale=scale)

            # Define the CDF and PDF of M
            def F_M(m):
                return F_Z(m) ** (self.k - 1)

            def f_M(m):
                return (self.k - 1) * (F_Z(m) ** (self.k - 2)) * f_Z(m)

            # Integrate to compute ASR
            integrand = lambda m: (1 - F_Z(m - 1)) * f_M(m)
            lower_limit = -np.inf
            upper_limit = np.inf

            asr, _ = quad(integrand, lower_limit, upper_limit)
                        
        return asr
