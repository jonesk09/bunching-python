# Simulated Bunching Data Generator

import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BunchingData:
    """
    Class containing simulated data for bunching examples.
    
    Attributes:
        kink_vector (np.ndarray): Simulated earnings vector for kink examples
        notch_vector (np.ndarray): Simulated earnings vector for notch examples
    """
    kink_vector: np.ndarray
    notch_vector: np.ndarray

def generate_bunching_data(
    n_base: int = 25000,
    n_bunch: int = 2500,
    base_mean: float = 10000,
    base_std: float = 2000,
    seed: int = 42
) -> BunchingData:
    """
    Generate simulated data for bunching analysis at kinks and notches.
    
    Args:
        n_base: Number of base observations
        n_bunch: Number of bunching observations
        base_mean: Mean of the base distribution
        base_std: Standard deviation of the base distribution
        seed: Random seed for reproducibility
        
    Returns:
        BunchingData object containing kink and notch vectors
    """
    np.random.seed(seed)
    
    # Generate base distribution
    base_dist = np.random.normal(base_mean, base_std, n_base)
    
    # Generate kink bunching
    kink_bunchers = np.random.normal(base_mean, base_std/4, n_bunch)
    kink_vector = np.concatenate([base_dist, kink_bunchers])
    
    # Generate notch bunching (more concentrated than kink)
    notch_bunchers = np.random.normal(base_mean, base_std/8, n_bunch)
    notch_vector = np.concatenate([base_dist, notch_bunchers])
    
    return BunchingData(
        kink_vector=kink_vector,
        notch_vector=notch_vector
    )

# Example usage:
def get_example_bunching_data() -> BunchingData:
    """
    Get a standard example dataset with approximately 27,500 observations.
    
    Returns:
        BunchingData object with example kink and notch vectors
    """
    return generate_bunching_data(
        n_base=25000,
        n_bunch=2500,
        base_mean=10000,
        base_std=2000,
        seed=42
    )
