# config_experiments.py

import numpy as np

# List of markers
markers = ['o', 'd', 's', '*', 'x', 'D', '+', '^', 'v', '<', '>', 'p', 'h', 'H', '8']

# Define domain sizes
small_domain = [25, 50, 75, 100]
medium_domain = [250, 500, 750, 1000]
high_domain = [2500, 5000, 7500, 10000]

# Define privacy regimes
high_privacy = np.arange(0.5, 2.1, 0.25)
low_privacy = np.arange(2., 10.1, 0.25)

# Analyses
dic_analyses = {
                "high_priv_small_domain": {"k": small_domain, "lst_eps": high_privacy, "title": "High Privacy Regime & Small Domain Size"},
                "high_priv_medium_domain": {"k": medium_domain, "lst_eps": high_privacy, "title": "High Privacy Regime & Medium Domain Size"},
                "high_priv_large_domain": {"k": high_domain, "lst_eps": high_privacy, "title": "High Privacy Regime & Large Domain Size"}, 
                
                "low_priv_small_domain": {"k": small_domain, "lst_eps": low_privacy, "title": "Low Privacy Regime & Small Domain Size"},
                "low_priv_medium_domain": {"k": medium_domain, "lst_eps": low_privacy, "title": "Low Privacy Regime & Medium Domain Size"},
                "low_priv_large_domain": {"k": high_domain, "lst_eps": low_privacy, "title": "Low Privacy Regime & Large Domain Size"}
                }