#!/usr/bin/env python3
"""
Calculates the expectation step in the EM algorithm for a GMM.
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None
    if (X.shape[1] != m.shape[1] or S.shape[1] != S.shape[2] or
            m.shape[1] != S.shape[1] or pi.shape[0] != m.shape[0]):
        return None, None

    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    k = pi.shape[0]

    # Build array of PDF values w/ each cluster
    pdfs = np.array([pdf(X, m[i], S[i]) for i in range(k)])

    # Calculate the weighted PDFs
    weighted_pdfs = pi[:, np.newaxis] * pdfs

    # Normalize posterior probabilities by marginal probabilities
    marginal_prob = np.sum(weighted_pdfs, axis=0)
    post_probs = weighted_pdfs / marginal_prob

    # Calc. the log likelihood(sum of logs of all marginal probs)
    log_likelihood = np.sum(np.log(marginal_prob))

    return post_probs, log_likelihood
