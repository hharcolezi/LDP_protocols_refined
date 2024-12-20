import numpy as np
from numba import jit
from scipy.stats import binom
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy.stats")
warnings.filterwarnings("ignore", message="divide by zero encountered in _binom_pdf", module="scipy.stats._discrete_distns")

@jit(nopython=True)
def ue_obfuscate(input_data: int, k: int, p: float, q: float) -> np.ndarray:
    """
    Obfuscate the input data using the Unary Encoding (UE) protocol, also known as Basic One-Time RAPPOR.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the domain (number of possible values). Must be an integer greater than or equal to 2.
    p : float
        The probability of retaining a '1' in the unary encoded vector for the true value.
    q : float
        The probability of flipping a '0' to '1' in the unary encoded vector for false values.

    Returns
    -------
    np.ndarray
        An obfuscated unary vector of size `k`.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data is not None and (input_data < 0 or input_data >= k):
        raise ValueError("input_data must be in the range [0, k-1].")

    # Unary encoding
    input_ue_data = np.zeros(k)
    if input_data is not None:
        input_ue_data[input_data] = 1

    # Initializing a zero-vector
    obfuscated_vec  = np.zeros(k)

    # UE perturbation function
    for ind in range(k):
        if input_ue_data[ind] != 1:
            rnd = np.random.random()
            if rnd <= q:
                obfuscated_vec [ind] = 1
        else:
            rnd = np.random.random()
            if rnd <= p:
                obfuscated_vec [ind] = 1
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

class UnaryEncoding:
    def __init__(self, k: int, epsilon: float, optimal: bool = True):
        """
        Initialize the Unary Encoding (UE) protocol.

        Parameters
        ----------
        k : int
            The size of the domain (number of possible values). Must be an integer greater than or equal to 2.
        epsilon : float
            The privacy budget, which determines the strength of the privacy guarantee. Must be positive.
        optimal : bool, optional
            If True, uses the Optimized Unary Encoding (OUE) protocol. Default is True.

        Raises
        ------
        ValueError
            If `k` is not an integer >= 2 or `epsilon` is not greater than 0.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")
        
        self.k = k
        self.epsilon = epsilon
        self.optimal = optimal

        # Optimized parameters
        if self.optimal:
            self.p = 1 / 2
            self.q = 1 / (np.exp(self.epsilon) + 1)
        # Symmetric parameters (p + q = 1)
        else: 
            self.p = np.exp(self.epsilon / 2) / (np.exp(self.epsilon / 2) + 1)
            self.q = 1 - self.p


    def obfuscate(self, input_data: int) -> np.ndarray:
        """
        Obfuscate the input data using the Unary Encoding (UE) mechanism.

        Parameters
        ----------
        input_data : int
            The user's true input value. Must be in the range [0, k-1], or None if no value is provided.

        Returns
        -------
        np.ndarray
            An obfuscated unary vector of size `k`.

        Raises
        ------
        ValueError
            If `input_data` is not in the range [0, k-1].
        """
        return ue_obfuscate(input_data, self.k, self.p, self.q)
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Unary Encoding (UE) mechanism.

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

    def get_variance(self) -> float:
        """
        Compute the variance of the Unary Encoding (UE) mechanism.

        Returns
        -------
        float
            The variance of the UE mechanism.
        """
        
        return self.q * (1 - self.q) / (self.p - self.q)**2
    
    def get_asr(self) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the Unary Encoding (UE) mechanism.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR) of the UE mechanism.
        """
        if self.optimal:
            return 1 / (2 * self.k) * (np.exp(self.epsilon) / (np.exp(self.epsilon) + 1))**(self.k - 1) + \
                   sum([(1 / (2 * i)) * binom.pmf(k=i - 1, n=self.k - 1, p=1 / (np.exp(self.epsilon) + 1)) for i in range(1, self.k + 1)])
        else:
            return 1 / (self.k * (np.exp(self.epsilon / 2) + 1)) * (np.exp(self.epsilon / 2) / (np.exp(self.epsilon / 2) + 1))**(self.k - 1) + \
                   sum([(np.exp(self.epsilon / 2) / ((np.exp(self.epsilon / 2) + 1) * i)) * binom.pmf(k=i - 1, n=self.k - 1, p=1 / (np.exp(self.epsilon / 2) + 1)) for i in range(1, self.k + 1)])
