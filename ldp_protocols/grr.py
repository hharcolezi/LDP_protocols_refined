import numpy as np
from numba import jit

@jit(nopython=True)
def grr_obfuscate(input_data: int, k: int, p: float) -> int:
    """
    Obfuscate the input data using the Generalized Randomized Response (GRR) mechanism.

    The GRR mechanism sanitizes the user's true value by either reporting it truthfully 
    with probability `p` or randomly selecting a different value from the domain with 
    complementary probability.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be greater than or equal to 2.
    p : float
        The probability of reporting the true value. Derived from the privacy budget `epsilon`.

    Returns
    -------
    int
        The sanitized value, which is either the true input value or a randomly chosen value 
        from the domain (excluding the true value).

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")

    domain = np.arange(k)
    if np.random.binomial(1, p) == 1:
        return input_data
    else:
        return np.random.choice(domain[domain != input_data])

class GeneralizedRandomizedResponse:
    def __init__(self, k: int, epsilon: float):
        """
        Initialize the Generalized Randomized Response (GRR) mechanism.

        Parameters
        ----------
        k : int
            The size of the domain (number of possible values). Must be an integer >= 2.
        epsilon : float
            The privacy budget, which determines the strength of the privacy guarantee. 
            Must be a positive numerical value.

        Raises
        ------
        ValueError
            If `k` is not an integer >= 2 or if `epsilon` is not greater than 0.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer value >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")

        self.k = k
        self.epsilon = epsilon
        self.p = np.exp(epsilon) / (np.exp(epsilon) + k - 1)
        self.q = (1 - self.p) / (k - 1)

    def obfuscate(self, input_data: int) -> int:
        """
        Obfuscate the input data using the Generalized Randomized Response (GRR) mechanism.

        Parameters
        ----------
        input_data : int
            The user's true input value to be sanitized. Must be in the range [0, k-1].

        Returns
        -------
        int
            The sanitized value after applying the GRR mechanism.

        Raises
        ------
        ValueError
            If `input_data` is not in the range [0, k-1].
        """
        return grr_obfuscate(input_data, self.k, self.p)
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Generalized Randomized Response (GRR) mechanism.

        This method applies unbiased estimation to the collected noisy reports to recover the approximate 
        frequencies of values in the domain. It uses the GRR-specific parameters `p` and `q` 
        to correct for the randomized responses.

        Parameters
        ----------
        noisy_reports : list of int
            A list of noisy reports collected from users. Each report corresponds to a value 
            in the domain `[0, k-1]` that has been obfuscated by the GRR mechanism.

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
    
    def attack(self, obfuscated_value: int) -> int:
        """
        Perform a privacy attack on a reported value generated using the GRR mechanism.

        In the GRR mechanism, the reported value is predicted as the true value. 
        This is because the probability of reporting the true value (`p`) is always greater 
        than the probability of reporting any other value (`q`), making it the most likely inference.

        Parameters
        ----------
        obfuscated_value : int
            The obfuscated value reported by the GRR mechanism. Must be in the range `[0, k-1]`.

        Returns
        -------
        int
            The inferred true value of the input, which is directly taken as the obfuscated value.
        """
        
        return obfuscated_value

    def get_variance(self, n: int = 1) -> float:
        """
        Compute the variance of the Generalized Randomized Response (GRR) mechanism.

        Returns
        -------
        float
            The variance of the GRR mechanism.
        """
        return self.q * (1 - self.q) / (n * (self.p - self.q)**2)

    def get_asr(self) -> float:
        """
        Compute the Adversarial Success Rate (ASR) for the Generalized Randomized Response (GRR) mechanism.

        Returns
        -------
        float
            The probability that an attacker correctly infers the original input value after applying the GRR mechanism.
        """
        return self.p
