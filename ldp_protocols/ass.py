import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def ss_obfuscate(input_data: int, k: int, epsilon: float, omega: int) -> np.ndarray:
    """
    Obfuscate the input data using the Subset Selection (SS) protocol.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be an integer >= 2.
    epsilon : float
        The privacy budget for the LDP mechanism. Must be a positive value.
    omega : int
        The size of the subset used in the SS mechanism.

    Returns
    -------
    np.ndarray
        A sanitized subset of values of size `omega`.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")

    # Mapping domain size k to the range [0, ..., k-1]
    domain = np.arange(k)

    # SS parameters
    p = omega * np.exp(epsilon) / (omega * np.exp(epsilon) + k - omega)

    # SS perturbation function
    rnd = np.random.random()
    sub_set = np.zeros(omega, dtype='int64')
    if rnd <= p:
        sub_set[0] = int(input_data)
        sub_set[1:] = np.random.choice(domain[domain != input_data], size=omega - 1, replace=False)
        return sub_set
    else:
        return np.random.choice(domain[domain != input_data], size=omega, replace=False)
    
@jit(nopython=True)
def attack_ss(obfuscated_vec: np.ndarray) -> int:
    """
    Perform a privacy attack on an obfuscated subset generated using the Adaptive Subset Selection (ASS) protocol.

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

class AdaptiveSubsetSelection:
    def __init__(self, k: int, epsilon: float, w_asr: float = 0.5, w_variance: float = 0.5):
        """
        Initialize the Adaptive Subset Selection (ASS) protocol with domain size k and privacy parameter epsilon.

        Parameters
        ----------
        k : int
            Attribute's domain size. Must be an integer greater than or equal to 2.
        epsilon : float
            Privacy guarantee. Must be a positive numerical value.
        w_asr : float, optional
            Weight given to the Adversarial Success Rate (ASR) in the objective function. Default is 0.5.
        w_variance : float, optional
            Weight given to the variance in the objective function. Default is 0.5.

        Raises
        ------
        ValueError
            If `k` is not >= 2, `epsilon` is not positive, or the weights are invalid.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")
        if not (0 <= w_asr <= 1) or not (0 <= w_variance <= 1):
            raise ValueError("Weights must be between 0 and 1.")

        # Normalize the weights so that their sum is 1
        total_weight = w_asr + w_variance
        self.w_asr = w_asr / total_weight
        self.w_variance = w_variance / total_weight
        self.k = k
        self.epsilon = epsilon
        self.omega = self.optimize_parameters()
        self.p = (self.omega * np.exp(self.epsilon)) / (self.omega * np.exp(self.epsilon) + self.k - self.omega)
        self.q = (self.omega * np.exp(self.epsilon) * (self.omega - 1) + (self.k - self.omega) * self.omega) / ((self.k - 1) * (self.omega * np.exp(self.epsilon) + self.k - self.omega))

    def get_parameter_range(self) -> np.ndarray:
        """
        Get the range of omega values to search over during optimization.

        Returns
        -------
        numpy.ndarray
            The range of omega values to search over.
        """
        return np.arange(1, self.k - 1) # Omega must be between 1 and k-1
    
    def optimize_parameters(self) -> int:
        """
        Optimize the value of omega using grid search to balance variance and ASR.

        Returns
        -------
        int
            The optimized value of omega.
        """
        # Define range of omega values to search over
        omega_values = self.get_parameter_range()

        best_omega = 1
        best_obj_value = float('inf')

        for omega in omega_values:
            asr = self.get_asr(omega)
            variance = self.get_variance(omega)
            obj_value = self.w_asr * asr + self.w_variance * variance

            if obj_value < best_obj_value:
                best_omega = omega
                best_obj_value = obj_value

        return best_omega

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the Adaptive Subset Selection (ASS) mechanism.

        Parameters
        ----------
        input_data : int
            The user's true input value to be obfuscated. Must be in the range [0, k-1].

        Returns
        -------
        np.ndarray
            A sanitized subset of values of size omega.
        """
        return ss_obfuscate(input_data, self.k, self.epsilon, self.omega)
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Subset Selection (ASS) mechanism.

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
        Perform a privacy attack on an obfuscated subset generated using the Adaptive Subset Selection (ASS) protocol.

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

    def get_variance(self, omega: int = None) -> float:
        """
        Compute the variance of the Adaptive Subset Selection (ASS) mechanism.

        Parameters
        ----------
        omega : int, optional
            The subset size. If None, the optimized omega value is used.

        Returns
        -------
        float
            The variance of the ASS mechanism.
        """
        if omega is None:
            omega = self.omega

        # Dynamically calculate p and q for the current omega
        p = (omega * np.exp(self.epsilon)) / (omega * np.exp(self.epsilon) + self.k - omega)
        q = (omega * np.exp(self.epsilon) * (omega - 1) + (self.k - omega) * omega) / ((self.k - 1) * (omega * np.exp(self.epsilon) + self.k - omega))

        return q * (1 - q) / (p - q)**2

    def get_asr(self, omega: int = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) for the Adaptive Subset Selection (ASS) mechanism.

        Parameters
        ----------
        omega : int, optional
            The subset size. If None, the optimized omega value is used.

        Returns
        -------
        float
            The ASR of the ASS mechanism.
        """
        if omega is None:
            omega = self.omega
            
        return np.exp(self.epsilon) / (omega * np.exp(self.epsilon) + self.k - omega)

    def plot_objective_function(self) -> None:
        """
        Plot the objective function over a range of omega values, highlighting the optimal omega value.
        """
        omega_values = self.get_parameter_range()
        objective_values = []

        for omega in omega_values:
            asr = self.get_asr(omega)
            variance = self.get_variance(omega)
            obj_value = self.w_asr * asr + self.w_variance * variance
            objective_values.append(obj_value)

        plt.plot(omega_values, objective_values, marker='o', label='Objective Function')
        plt.xlabel('omega')
        plt.ylabel('Objective Function Value')
        plt.title(f'Objective Function vs. omega (epsilon={self.epsilon})')
        plt.grid(True)

        # Highlight the best omega value
        plt.axvline(self.omega, color='r', linestyle='--', label=f'Optimal omega={self.omega}')
        plt.legend()
        plt.yscale('log')
        plt.show()
