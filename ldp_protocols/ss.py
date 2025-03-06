import numpy as np
from numba import jit

@jit(nopython=True)
def ss_obfuscate(input_data: int, k: int, p: float, omega: int) -> np.ndarray:
    """
    Obfuscate the input data using the Subset Selection (SS) protocol.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the attribute's domain. Must be an integer >= 2.
    p : float
        The probability of including the true input value in the sanitized subset. 
        Derived from the privacy budget epsilon.
    omega : int
        The size of the output subset (number of values in the obfuscated result).

    Returns
    -------
    numpy.ndarray
        A sanitized subset (array) of size `omega` containing obfuscated values.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")

    # Mapping domain size k to the range [0, ..., k-1]
    domain = np.arange(k)

    # SS perturbation function
    sub_set = np.zeros(omega, dtype='int64')
    if np.random.random() <= p:
        sub_set[0] = int(input_data)
        sub_set[1:] = np.random.choice(domain[domain != input_data], size=omega - 1, replace=False)
        np.random.shuffle(sub_set)
        return sub_set
    else:
        return np.random.choice(domain[domain != input_data], size=omega, replace=False)

@jit(nopython=True)
def attack_ss(obfuscated_vec: np.ndarray) -> int:
    """
    Perform a privacy attack on an obfuscated subset generated using the Subset Selection (SS) protocol.

    This method attempts to infer the true value by randomly selecting a value from the obfuscated subset.
    Since the true value is included with higher probability in the subset, an adversary can exploit this 
    to make an educated guess.

    Parameters
    ----------
    obfuscated_vec : np.ndarray
        An obfuscated subset of values generated using the SS protocol. 
        The subset contains a fixed number of values selected from the domain.

    Returns
    -------
    int
        The inferred true value of the input. This is selected randomly from the values present 
        in the obfuscated subset.
    """
                
    return np.random.choice(obfuscated_vec)

@jit(nopython=True)
def ss_asr(epsilon: float, k: int, omega: int) -> float:
    """
    Calculate the Adversarial Success Rate (ASR) of the Subset Selection (SS) protocol.

    Parameters
    ----------
    epsilon : float
        The privacy budget for the SS mechanism. Must be a positive value.
    k : int
        The size of the attribute's domain. Must be an integer >= 2.
    omega : int
        The size of the output subset.

    Returns
    -------
    float
        The Adversarial Success Rate (ASR), which quantifies the adversary's ability 
        to correctly guess the true value from the sanitized subset.
    """
    # Calculate parameters
    return np.exp(epsilon) / (omega * np.exp(epsilon) + k - omega)

class SubsetSelection:
    def __init__(self, k: int, epsilon: float):
        """
        Initialize the Subset Selection (SS) protocol.

        Parameters
        ----------
        k : int
            The size of the attribute's domain. Must be an integer >= 2.
        epsilon : float
            The privacy budget for the SS mechanism. Must be a positive value.

        Raises
        ------
        ValueError
            If `k` is not an integer >= 2 or if `epsilon` is not a positive value.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")
        
        self.k = k
        self.epsilon = epsilon
        self.omega = int(max(1, np.rint(self.k / (np.exp(self.epsilon) + 1))))
        self.p = (self.omega * np.exp(self.epsilon)) / (self.omega * np.exp(self.epsilon) + self.k - self.omega)
        self.q = (self.omega * np.exp(self.epsilon) * (self.omega - 1) + (self.k - self.omega) * self.omega) / ((self.k - 1) * (self.omega * np.exp(self.epsilon) + self.k - self.omega))

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the Subset Selection (SS) mechanism.

        Parameters
        ----------
        input_data : int
            The user's true input value to be sanitized. Must be in the range [0, k-1].

        Returns
        -------
        numpy.ndarray
            A sanitized subset (array) of size `omega` containing obfuscated values.
        """
        return ss_obfuscate(input_data, self.k, self.p, self.omega)
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Subset Selection (SS) mechanism.

        This method applies unbiased estimation to the collected noisy reports to approximate 
        the true frequencies of values in the domain. It uses SS-specific parameters `p` (true value probability)
        and `q` (false value probability) to correct for the randomized responses.

        Parameters
        ----------
        noisy_reports : list of int
            A list of noisy reports collected from users. Each report corresponds to a single obfuscated value
            within the domain `[0, k-1]`, chosen as part of a subset generated by the SS mechanism.

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
        Perform a privacy attack on an obfuscated subset generated using the Subset Selection (SS) protocol.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated subset of values generated using the SS protocol. 
            The subset contains a fixed number of values selected from the domain.

        Returns
        -------
        int
            The inferred true value of the input. This is selected randomly from the values present 
        in the obfuscated subset.
        """
        
        return attack_ss(obfuscated_vec)

    def get_mse(self, n: int = 1) -> float:
        """
        Compute the MSE of the Subset Selection (SS) protocol.

        Returns
        -------
        float
            The MSE of the SS mechanism, quantifying the expected estimation error.
        """

        return self.q * (1 - self.q) / (n * (self.p - self.q)**2)
    
    def get_asr(self) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the Subset Selection (SS) protocol.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR), representing the adversary's ability to 
            correctly guess the true value.
        """
        return ss_asr(self.epsilon, self.k, self.omega)
