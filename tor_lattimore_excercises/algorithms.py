import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy as dcopy

from bandits import *


def FollowTheLeader(bandit, n):

    num_arms = bandit.K()
    rewards = np.zeros(num_arms)
    curr_arm_index = 0
    for t in range(n):
        if t < num_arms:
            rewards[curr_arm_index] = bandit.pull(curr_arm_index)

            # updates after every arm pull for first k pulls
            # gives argmax with uniform tie breaking
            best_arm = np.random.choice(
                np.flatnonzero(rewards == rewards.max()))
        else:
            bandit.pull(best_arm)


def FTLRun():

    n = 1000
    regrets = np.zeros(1000)
    for j in range(1000):
        bernoulli_bandit = BernoulliBandit([0.5, 0.6])
        FollowTheLeader(bernoulli_bandit, n)
        regrets[j] = bernoulli_bandit.regret()

    plt.hist(regrets)
    plt.show()

    ns = [i * 100 for i in range(1, 11)]
    regrets = np.zeros((len(ns), 1000))
    for i, n in enumerate(ns):
        for j in range(1000):
            bernoulli_bandit = BernoulliBandit([0.5, 0.6])
            FollowTheLeader(bernoulli_bandit, n)
            regrets[i, j] = bernoulli_bandit.regret()

    plt.errorbar(x=ns, y=regrets.mean(axis=1),
                 yerr=regrets.std(axis=1), fmt='-o')
    plt.show()


def ExploreThenCommit2(n, delta, m=None):
    # ETC algo with u1 = 0, u2 = -delta
    gaussian_bandit = GaussianBandit([0, -delta])

    if m is None:
        m_val = (4 / (delta * delta)) * \
            np.log(n * delta * delta / 4)  # from eq 6.5
        m = max(1, int(np.ceil(m_val)))

    Rn_ub_val = delta + (4 / delta) * \
        (1 + max(0, np.log(n * delta * delta / 4)))  # from eq 6.6
    Rn_max = min(n * delta, Rn_ub_val)

    rewards = np.zeros((2, m))
    rewards[0, :] = gaussian_bandit.pull_repeatedly(0, m)
    rewards[1, :] = gaussian_bandit.pull_repeatedly(1, m)

    rewards = rewards.mean(axis=1)  # average over m pulls

    best_arm = np.random.choice(
        np.flatnonzero(rewards == rewards.max()))

    rewards = gaussian_bandit.pull_repeatedly(best_arm, n - 2 * m)

    return Rn_max, gaussian_bandit.regret()


def ETC2Run():
    delta_vals = [i * 0.04 for i in range(1, 25)]
    ub_regrets = []
    actual_regrets = []
    num_runs = 500

    for delta in delta_vals:
        ub_regret_sum, actual_regret_sum = 0, 0
        for run_no in range(num_runs):
            ub_regret, actual_regret = ExploreThenCommit2(n=1000, delta=delta)
            ub_regret_sum += ub_regret
            actual_regret_sum += actual_regret
        ub_regrets.append(ub_regret_sum / num_runs)
        actual_regrets.append(actual_regret_sum / num_runs)

    plt.plot(delta_vals, ub_regrets, label="Upper Bound")
    plt.plot(delta_vals, actual_regrets, label="Actual Regret")
    plt.legend()
    plt.show()

    m_vals = [i * 15 for i in range(1, 25)]
    actual_regrets = np.zeros((len(m_vals), num_runs))
    for i, m in enumerate(m_vals):
        actual_regret_sum = 0
        for run_no in range(num_runs):
            _, actual_regret = ExploreThenCommit2(
                n=2000, delta=0.1, m=m)
            actual_regrets[i, run_no] = actual_regret

    plt.plot(m_vals, actual_regrets.mean(axis=1))
    plt.show()

    plt.plot(m_vals, actual_regrets.std(axis=1))
    plt.show()

    hue = np.array(m_vals)
    hue = np.repeat(hue, num_runs)
    sns.displot(x=actual_regrets.flatten(),
                hue=hue, kind="kde")
    plt.show()


def UCB(gaussian_bandit, n, alpha):
    # alpha is confidence level
    mean_hats = np.zeros(gaussian_bandit.K())
    rewards = [[] for _ in range(gaussian_bandit.K())]
    ucb_vals = np.ones(gaussian_bandit.K()) * 10**10

    for i in range(n):
        best_arm = np.random.choice(
            np.flatnonzero(ucb_vals == ucb_vals.max()))
        reward = gaussian_bandit.pull(best_arm)
        rewards[best_arm].append(reward)
        n_arm = len(rewards[best_arm])
        mean_hats[best_arm] = np.mean(rewards[best_arm])
        for arm in range(gaussian_bandit.K()):
            if len(rewards[arm]) != 0:
                ucb_vals[arm] = mean_hats[arm] + alpha * np.sqrt(
                    np.log(i + 1) / len(rewards[arm]))
            # else ucb stays very large

    return gaussian_bandit.regret()


def UCBRun():
    n_vals = [500, 1000, 1500, 2000, 4000, 8000]
    # alpha = 10**5 is pure exploration
    # alpha = 0 is greedy
    alpha = 2
    delta = 0.2

    etc_means = []
    ucb_means = []

    for n in n_vals:
        n_reps = 10
        regret_etc_sum, regret_ucb_sum = 0, 0

        for _ in range(n_reps):
            gaussian_bandit = GaussianBandit([0, -0.1, -0.2, -0.5, -1, -2])
            _, regret_etc = ExploreThenCommit2(n, delta)
            regret_ucb = UCB(gaussian_bandit, n, alpha)

            regret_etc_sum += regret_etc
            regret_ucb_sum += regret_ucb

        etc_means.append(regret_etc_sum / n_reps)
        ucb_means.append(regret_ucb / n_reps)

    plt.plot(n_vals, etc_means, label="etc")
    plt.plot(n_vals, ucb_means, label="ucb")
    plt.legend()
    plt.show()


def LinUCB(features, theta, n, delta, lambda_reg):
    # constant set of actions
    # matrix inverse can be optimized using sherman-morrison formula
    lgb = LinearGaussianBandit(features, theta)

    d = len(features[0])
    summation_AtXt = 0
    Vt = lambda_reg * np.eye(d)
    theta_hat_t = np.random.normal(0, size=(d, 1))

    ucb = np.zeros(len(features))

    for t in range(n):

        root_beta_t = np.sqrt(lambda_reg) + np.sqrt(
            2 * np.log(1 / delta) + d * np.log(1 + t / (lambda_reg * d)))

        for i in range(len(ucb)):
            a = features[i]
            a = a.reshape(-1, 1)
            ucb[i] = np.dot(a.T, theta_hat_t) + root_beta_t * np.sqrt(a.T @ np.linalg.pinv(Vt) @ a)

        At = features[np.argmax(ucb)]
        At = At.reshape((-1, 1))  # At is dx1

        Xt = lgb.pull_repeatedly(np.argmax(ucb), 1)

        # update variables
        Vt += (At @ At.T)
        summation_AtXt += At * Xt
        theta_hat_t = np.linalg.pinv(Vt) @ summation_AtXt

    return lgb.regret(), theta_hat_t


def LinUCBRun():
    diff_vals = [0.1 * i for i in range(1, 10)]
    num_reps = 50
    linucb_regrets = []
    ucb_regrets = []

    for diff in diff_vals:
        linucb_reg_sum, ucb_reg_sum = 0, 0
        for rep in range(num_reps):
            features = [np.array([1, 0]), np.array([0, 1])]
            theta = np.array([0, -diff])
            n = 1000
            delta = 1 / n
            lambda_reg = 1e-3
            alpha = 2

            regret_lin_ucb, theta_pred = LinUCB(
                features, theta, n, delta, lambda_reg)
            gaussian_bandit = GaussianBandit([0, -diff])
            regret_ucb = UCB(gaussian_bandit, n, alpha)

            linucb_reg_sum += regret_lin_ucb
            ucb_reg_sum += regret_ucb

        linucb_regrets.append(linucb_reg_sum / num_reps)
        ucb_regrets.append(ucb_reg_sum / num_reps)

    plt.plot(diff_vals, linucb_regrets, label="linucb")
    plt.plot(diff_vals, ucb_regrets, label="ucb")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    algo = input("Algorithm name: ")

    if algo == "FTL":
        FTLRun()

    elif algo == "ETC2":
        ETC2Run()

    elif algo == "UCB":
        UCBRun()

    elif algo == "LinUCB":
        LinUCBRun()

    else:
        print("Invalid algo name")
