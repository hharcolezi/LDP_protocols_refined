import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.stats import binom
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")
warnings.filterwarnings("ignore", message="divide by zero encountered in _binom_pdf", module="scipy.stats._discrete_distns")

@jit(nopython=True)
def ue_obfuscate(input_data: int, k: int, p: float, q: float) -> np.ndarray:
    """
    Obfuscate the input data using the Unary Encoding (UE) protocol with custom p and q parameters.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be an integer >= 2.
    p : float
        The probability of retaining a '1' in the UE vector for the true value.
    q : float
        The probability of flipping a '0' to '1' in the UE vector.

    Returns
    -------
    np.ndarray
        An obfuscated unary vector of size `k`.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")

    # Unary encoding
    input_ue_data = np.zeros(k)
    if input_data is not None:
        input_ue_data[input_data] = 1

    # Initializing a zero-vector
    obfuscated_vec = np.zeros(k)

    # UE perturbation function
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q:
                obfuscated_vec[ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p:
                obfuscated_vec[ind] = 1
    return obfuscated_vec

@jit(nopython=True)
def attack_ue(obfuscated_vec: np.ndarray, k: int) -> int:
        """
        Perform a privacy attack on an obfuscated unary vector.

        This method attempts to infer the true value from the obfuscated vector. If the vector 
        contains no '1' values (all positions are 0), the method returns a random guess 
        within the domain `[0, k-1]`. Otherwise, it randomly selects one of the indices where 
        the vector has a '1'.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated unary vector of size `k`, generated using the UE mechanism.

        k : int
            Domain size.

        Returns
        -------
        int
            The inferred true value of the input. If no inference is possible (sum of the vector is 0),
            a random value in the range `[0, k-1]` is returned.
        """

        # If the vector contains no '1', make a random guess
        if np.sum(obfuscated_vec) == 0:
            return np.random.randint(k)
        else:
            # Randomly select one of the indices where the value is '1'
            return np.random.choice(np.where(obfuscated_vec == 1)[0])

class AdaptiveUnaryEncoding:
    def __init__(self, k: int, epsilon: float, w_asr: float = 0.5, w_variance: float = 0.5):
        """
        Initialize the Adaptive Unary Encoding (AUE) protocol.

        Parameters
        ----------
        k : int
            The size of the domain (number of possible values). Must be an integer >= 2.
        epsilon : float
            The privacy budget, which determines the level of privacy guarantee. Must be positive.
        w_asr : float, optional
            Weight for the Adversarial Success Rate (ASR) in the optimization objective. Default is 0.5.
        w_variance : float, optional
            Weight for the variance in the optimization objective. Default is 0.5.

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
        self.p, self.q = self.optimize_parameters()

    def get_parameter_range(self) -> np.ndarray:
        """
        Get a range of p values to optimize over.

        Returns
        -------
        numpy.ndarray
            A range of p values between 0.5 and 1.
        """
        return np.linspace(0.5, 0.999999, 100)

    def optimize_parameters(self) -> tuple[float, float]:
        """
        Optimize the parameters `p` and `q` using grid search to minimize the weighted sum of ASR and variance.

        Returns
        -------
        tuple[float, float]
            The optimized values of `p` and `q`.

        Raises
        ------
        ValueError
            If no feasible solution for the optimization is found.
        """

        p_values = self.get_parameter_range()  # Define a range for p between 0.5 and 1
        best_objective_value = float('inf')
        best_p, best_q = None, None

        for p in p_values:
            # Calculate q to satisfy epsilon-LDP
            q = p / (np.exp(self.epsilon) * (1 - p) + p)

            # Calculate the objective function value
            obj_value = self.w_asr * self.get_asr(p, q) + self.w_variance * self.get_variance(p, q)

            if obj_value < best_objective_value:
                best_objective_value = obj_value
                best_p, best_q = p, q

        if best_p is not None and best_q is not None:
            return best_p, best_q

        else:
            raise ValueError("Optimization failed. No feasible solution found.")

    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the optimized AUE mechanism.

        Parameters
        ----------
        input_data : int or None
            The user's true input value. Must be in the range [0, k-1], or None if no value is provided.

        Returns
        -------
        np.ndarray
            An obfuscated unary vector of size `k`.
        """

        return ue_obfuscate(input_data, self.k, self.p, self.q)
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Unary Encoding (AUE) mechanism.

        This method applies unbiased estimation to the noisy unary vectors (noisy reports) 
        to recover the approximate frequencies of values in the domain.

        Parameters
        ----------
        noisy_reports : list of np.ndarray
            A list of noisy unary vectors collected from users. Each unary vector 
            has size `k`, where `k` is the size of the domain.

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

        n = len(noisy_reports)
        if n == 0:
            raise ValueError("Noisy reports cannot be empty.")

        # Count the occurrences of each value in the noisy reports
        support_counts = sum(noisy_reports)

        # Unbiased frequency estimation
        freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))
        
        # Ensure non-negative estimates and normalize
        return np.maximum(freq_estimates, 0) / np.sum(np.maximum(freq_estimates, 0))
    
    def attack(self, obfuscated_vec: np.ndarray) -> int:
        """
        Perform a privacy attack on an obfuscated unary vector.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated unary vector of size `k`, generated using the UE mechanism.

        Returns
        -------
        int
            The inferred true value of the input. If no inference is possible (sum of the vector is 0),
            a random value in the range `[0, k-1]` is returned.
        """
        
        return attack_ue(obfuscated_vec, self.k)

    def get_variance(self, p: float = None, q: float = None) -> float:
        """
        Compute the variance of the AUE mechanism.

        Parameters
        ----------
        p : float, optional
            Probability of retaining a bit as 1. If None, use the optimized `p`.
        q : float, optional
            Probability of flipping a bit from 0 to 1. If None, use the optimized `q`.

        Returns
        -------
        float
            The variance of the AUE mechanism.
        """
        if p is None or q is None:
            p, q = self.p, self.q

        return q * (1 - q) / (p - q) ** 2

    def get_asr(self, p: float = None, q: float = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the AUE mechanism.

        Parameters
        ----------
        p : float, optional
            Probability of retaining a bit as 1. If None, use the optimized `p`.
        q : float, optional
            Probability of flipping a bit from 0 to 1. If None, use the optimized `q`.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR) of the AUE mechanism.
        """
        if p is None or q is None:
            p, q = self.p, self.q

        # ASR for Event E0: The original bit is flipped to 0, and all other bits remain 0
        asr_e0 = (1 - p) * (1 - q) ** (self.k - 1) * (1 / self.k)

        # Sum of ASR for all other events Ei (i >= 1): Original bit retained and i-1 other bits flipped
        asr_sum = 0
        for i in range(1, self.k + 1):
            binom_prob = binom.pmf(i - 1, self.k - 1, q)
            asr_sum += p * (1 / i) * binom_prob

        # Total expected ASR
        return asr_e0 + asr_sum
    
    def plot_objective_function(self) -> None:
        """
        Plot the objective function over the range of `p` values and highlight the optimized `p`.
        """
        p_values = self.get_parameter_range()  # Define a range for p between 0.5 and 1
        objective_values = []

        for p in p_values:
            # Calculate q to satisfy epsilon-LDP
            q = p / (np.exp(self.epsilon) * (1 - p) + p)

            # Calculate the objective function value
            asr = self.get_asr(p, q)
            variance = self.get_variance(p, q)
            objective_value = self.w_asr * asr + self.w_variance * variance
            objective_values.append(objective_value)

        plt.plot(p_values, objective_values, marker='o', label='Objective Function')
        plt.xlabel('p')
        plt.ylabel('Objective Function Value')
        plt.title(f'Objective Function vs. p (epsilon={self.epsilon})')
        plt.grid(True)

        # Highlight the best p value
        plt.axvline(self.p, color='r', linestyle='--', label=f'Optimal p={self.p:.4f}')

        plt.legend()
        plt.yscale('log')
        plt.show()
