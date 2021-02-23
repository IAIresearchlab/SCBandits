    # NOTATION
    # n: number of patients in trial.
    # prior.i: prior number of successes on treatment A.
    # prior.j: prior number of failures on treatment A.
    # prior.k: prior number of successes on treatment B.
    # prior.l: prior number of failures on treatment B.
    # V: value function representing the maximum expected total reward (i.e. number of successes) in the rest of the trial after t patients have been treated.
    # t: number of patients that have been treated.
    # p: the degree of randomisation.
    # Y: minimum number of observations required on each treatment arm (i.e. the degree of constraining).
    # i: observed number of successes on treatment A.
    # j: observed number of failures on treatment A.
    # k: observed number of successes on treatment B.
    # l: observed number of failures on treatment B.

import numpy as np

def CRDP(n, prior_arm0_successes, prior_arm0_failures, prior_arm1_successes, prior_arm1_failures, p, min_arm_pulls):

    V = np.full((n+2, n+2, n+2), 0)
    action = np.full((n+2, n+2, n+2, n+2), 0)
    t = n + 4

    for i in range(1, t-2):
        for j in range(1, t-i-1):
            for k in range(1, t-i-j):
                l = t - i - j - k
                # apply a large penalty for not pulling the arms enough times
                if i + j < min_arm_pulls:
                    V[i,j,k] = -n
                if k + l < min_arm_pulls:
                    V[i,j,k] = -n

    for t in range(n+3, 3, -1):
        for i in range(1, t-2):
            for j in range(1, t-i-1):
                for k in range(1, t-i-j):

                    l = t - i - j - k
                    # print(i, j, k, l)
                    expected_prob_0 = (i - 1 + prior_arm0_successes) / (i - 1 + prior_arm0_successes + j - 1 + prior_arm0_failures)
                    expected_prob_1 = (k - 1 + prior_arm1_successes) / (k - 1 + prior_arm1_successes + l - 1 + prior_arm1_failures)
                    # print(f"expected prob 0 {expected_prob_0}")
                    # print(f"expected prob 1 {expected_prob_1}")
                    V_a0 = expected_prob_0 * (1 + V[i+1, j, k]) + (1 - expected_prob_0) * (0 + V[i, j+1, k])
                    V_a1 = expected_prob_1 * (1 + V[i, j, k+1]) + (1 - expected_prob_1) * (0 + V[i, j, k])
                    # print(f"V a0: {V_a0}")
                    # print(f"V a1: {V_a1}")
                    if p * V_a0 + (1-p) * V_a1 > (1-p) * V_a0 + p * V_a1:
                        # action 1 is optimal
                        # print("selecting arm 0")
                        action[n-(t-4), i, j, k] = 0
                    elif p * V_a0 + (1-p) * V_a1 < (1-p) * V_a0 + p * V_a1:
                        # action 2 is optimal
                        # print("selecting arm 1")
                        action[n-(t-4), i, j, k] = 1
                    elif p * V_a0 + (1-p) * V_a1 == (1-p) * V_a1 + p * V_a0:
                        # either action 1 or 2 is optimal
                        action[n-(t-4), i ,j, k] = 2
                    V[i,j,k] <- max(p * V_a0 + (1-p) * V_a1, (1-p)* V_a0 + p*V_a1 )

    return action