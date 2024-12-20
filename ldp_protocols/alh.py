import numpy as np
from sys import maxsize
import xxhash
import matplotlib.pyplot as plt

class AdaptiveLocalHashing:
    def __init__(self, k: int, epsilon: float, w_asr: float = 0.5, w_variance: float = 0.5):
        """
        Initialize the Adaptive Local Hashing (ALH) protocol.

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
        self.g = self.optimize_parameters()

        # Calculate probability for GRR-based perturbation
        self.p = np.exp(self.epsilon) / (np.exp(self.epsilon) + self.g - 1)
        self.q = 1 / self.g

    def get_parameter_range(self) -> np.ndarray:
        """
        Get a range of g values to optimize over.

        Returns
        -------
        numpy.ndarray
            A range of g values between 2 and max(k, exp(epsilon) + 1).
        """
        return np.arange(2, max(self.k, int(np.round(np.exp(self.epsilon)) + 1) + 1))

    def optimize_parameters(self) -> int:
        """
        Grid-search optimization for the value of g to balance variance and ASR.

        Returns
        -------
        int
            The optimized value of g.
        """
        # Define range of g values to search over
        g_values = self.get_parameter_range()

        # Perform grid search to find the best g
        best_g = 2
        best_obj_value = float('inf')

        for g in g_values:
            asr = self.get_asr(g)
            variance = self.get_variance(g)
            obj_value = self.w_asr * asr + self.w_variance * variance
            if obj_value < best_obj_value:
                best_g = g
                best_obj_value = obj_value

        return best_g

    def obfuscate(self, input_data: int) -> tuple[int, int]:
        """
        Obfuscate the input data using the ALH mechanism.

        Parameters
        ----------
        input_data : int
            The true input value to be obfuscated. Must be in the range [0, k-1].

        Returns
        -------
        tuple[int, int]
            A tuple containing:
                - The sanitized (obfuscated) value (int) within the optimized hash domain size `g`.
                - The random seed (int) used for hashing.

        Raises
        ------
        ValueError
            If `input_data` is not in the range [0, k-1].
        """
        if input_data < 0 or input_data >= self.k:
            raise ValueError("input_data must be in the range [0, k-1].")

        # Generate random seed and hash the user's value
        rnd_seed = np.random.randint(0, maxsize, dtype=np.int64)
        hashed_input_data = (xxhash.xxh32(str(input_data), seed=rnd_seed).intdigest() % self.g)

        # GRR-based perturbation
        domain = np.arange(self.g)
        if np.random.binomial(1, self.p) == 1:
            sanitized_value = hashed_input_data
        else:
            sanitized_value = np.random.choice(domain[domain != hashed_input_data])

        return sanitized_value, rnd_seed
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Adaptive Local Hashing (ALH) mechanism.

        This method applies unbiased estimation to recover approximate frequencies of values 
        in the domain `[0, k-1]`. The LH mechanism maps input values to a hash domain of size `g`, 
        perturbs the mapped values, and reports the noisy results. The method uses `p` (true value probability) 
        and `q` (false value probability) to correct for this perturbation.

        Parameters
        ----------
        noisy_reports : list of tuple (int, int)
            A list of noisy reports collected from users. Each report is a tuple containing:
            - `value` : The obfuscated hash-mapped value.
            - `seed`  : The random seed used for hashing during the LH mechanism.

        Returns
        -------
        np.ndarray
            An array of estimated frequencies for each value in the domain `[0, k-1]`.
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
        
        # Hash-based support counting for LH protocols
        for value, seed in noisy_reports:
            for v in range(self.k):
                if value == (xxhash.xxh32(str(v), seed=seed).intdigest() % self.g):
                    support_counts[v] += 1

        # Unbiased frequency estimation
        freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))
        
        # Ensure non-negative estimates and normalize
        return np.maximum(freq_estimates, 0) / np.sum(np.maximum(freq_estimates, 0))
    
    def attack(self, val_seed):
        """
        Perform a privacy attack on an obfuscated value generated using the Adaptive Local Hashing (ALH) protocol.

        This method attempts to infer the true input value by leveraging the obfuscated hash-mapped value
        and the corresponding random seed used during hashing. The method reconstructs the possible 
        candidate values that could produce the same hash output and randomly selects one of them.

        Parameters
        ----------
        val_seed : tuple (int, int)
            A tuple containing:
            - `obfuscated value` : The hash-mapped value generated during obfuscation.
            - `seed` : The random seed used for hashing.

        Returns
        -------
        int
            The inferred true value of the input. If no valid candidate values are found, a random value 
            within the domain `[0, k-1]` is returned.
        """

        lh_val = val_seed[0]
        rnd_seed = val_seed[1]

        ss_lh = []
        for v in range(self.k):
            if lh_val == (xxhash.xxh32(str(v), seed=rnd_seed).intdigest() % self.g):
                ss_lh.append(v)

        if len(ss_lh) == 0:
            return np.random.randint(self.k)
        else:
            return np.random.choice(ss_lh)

    def get_variance(self, g: int = None) -> float:
        """
        Compute the variance of the LH mechanism for a given g.

        Parameters
        ----------
        g : int, optional
            Hash domain size. If None, use the optimized value of g.

        Returns
        -------
        float
            The variance of the LH mechanism.
        """
        if g is None:
            g = self.g
            
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + g - 1)
        q = 1 / g

        return q * (1 - q) / (p - q) ** 2

    def get_asr(self, g: int = None) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the LH mechanism for a given g.

        Parameters
        ----------
        g : int, optional
            Hash domain size. If None, use the optimized value of g.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR).
        """
        if g is None:
            g = self.g

        return np.exp(self.epsilon) / ((np.exp(self.epsilon) + g - 1) * max(self.k / g, 1))
    
    def plot_objective_function(self) -> None:
        """
        Plot the objective function over a range of g values, highlighting the optimal g value.
        """
        g_values = self.get_parameter_range()
        objective_values = []

        for g in g_values:
            asr = self.get_asr(g)
            variance = self.get_variance(g)
            objective_value = self.w_asr * asr + self.w_variance * variance
            objective_values.append(objective_value)

        plt.plot(g_values, objective_values, marker='o', label='Objective Function')
        plt.xlabel('g')
        plt.ylabel('Objective Function Value')
        plt.title(f'Objective Function vs. g (epsilon={self.epsilon})')
        plt.grid(True)

        # Highlight the best g value
        plt.axvline(self.g, color='r', linestyle='--', label=f'Optimal g={self.g}')
        plt.legend()
        plt.yscale('log')
        plt.show()
