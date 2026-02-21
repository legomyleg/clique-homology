import numpy as np

def calculate_p_vector(obs_betti, null_betti_matrix):

    null_mean = np.mean(null_betti_matrix, axis=0)
    centered_null = null_betti_matrix - null_mean
    l2_norm_null = np.linalg.norm(centered_null, axis=1)
    l2_norm_obs = np.linalg.norm(obs_betti - null_mean)

    nrow = null_betti_matrix.shape[0]

    pval = (np.sum(l2_norm_null >= l2_norm_obs) + 1) / (nrow + 1)

    return pval