import numpy as np
from sys import maxsize
import xxhash

class LocalHashing:
    def __init__(self, k: int, epsilon: float, optimal: bool = True):
        """
        Initialize the Local Hashing (LH) protocol.

        Parameters
        ----------
        k : int
            The domain size of the attribute (number of possible values). Must be >= 2.
        epsilon : float
            The privacy budget for the LDP mechanism. Must be a positive value.
        optimal : bool, optional
            If True, use the Optimized Local Hashing (OLH) protocol; otherwise, use Binary Local Hashing (BLH). Default is True.

        Raises
        ------
        ValueError
            If `k` is not >= 2 or if `epsilon` is not positive.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")
        self.k = k
        self.epsilon = epsilon
        self.optimal = optimal

        # Set the hash domain size
        self.g = 2  # BLH by default
        if self.optimal:
            self.g = int(round(np.exp(self.epsilon))) + 1

        # Calculate perturbation parameters
        self.p = np.exp(self.epsilon) / (np.exp(self.epsilon) + self.g - 1)
        self.q = 1/self.g

    def obfuscate(self, input_data: int) -> tuple[int, int]:
        """
        Obfuscate the input data using the Local Hashing (LH) mechanism.

        Parameters
        ----------
        input_data : int
            The true input value to be obfuscated. Must be in the range [0, k-1].

        Returns
        -------
        tuple[int, int]
            A tuple containing:
                - The sanitized (obfuscated) value (int) within the hash domain size `g`.
                - The random seed (int) used for hashing.

        Raises
        ------
        ValueError
            If `input_data` is not in the range [0, k-1].
        """
        if input_data is not None and (input_data < 0 or input_data >= self.k):
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
        Estimate frequencies from noisy reports collected using the Local Hashing (LH) mechanism.

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
        Perform a privacy attack on an obfuscated value generated using the Local Hashing (LH) protocol.

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

    def get_variance(self, n: int = 1) -> float:
        """
        Compute the variance of the LH mechanism.

        Returns:
        float: The variance of the LH mechanism.
        """
        
        return self.q * (1 - self.q) / (n * (self.p - self.q)**2)
    
    def get_asr(self) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the LH mechanism.

        Returns:
        float: The Adversarial Success Rate (ASR).
        """
        if self.optimal:
            return 1 / (2 * max(self.k / (np.exp(self.epsilon) + 1), 1))
        else:
            return 2 * np.exp(self.epsilon) / ((np.exp(self.epsilon) + 1) * self.k) 
        
    
