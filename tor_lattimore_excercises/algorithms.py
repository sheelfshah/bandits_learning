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
    mean_hats = np.ones(gaussian_bandit.K()) * 10**4
    rewards = [[] for _ in range(gaussian_bandit.K())]
    ucb_vals = dcopy(mean_hats)

    for i in range(n):
        best_arm = np.random.choice(
            np.flatnonzero(ucb_vals == ucb_vals.max()))
        reward = gaussian_bandit.pull(best_arm)
        rewards[best_arm].append(reward)
        n_arm = len(rewards[best_arm])
        mean_hats[best_arm] = np.mean(rewards[best_arm])
        ucb_vals[best_arm] = mean_hats[best_arm] + alpha * np.sqrt(
            np.log(i + 1) / n_arm)

    return gaussian_bandit.regret()


def UCBRun():
    n_vals = [100, 200, 500, 1000, 2000, 5000]
    alpha = 2
    delta = 0.5

    etc_means = []
    ucb_means = []

    for n in n_vals:
        n_reps = 25
        regret_etc_sum, regret_ucb_sum = 0, 0

        for _ in range(n_reps):
            gaussian_bandit = GaussianBandit([0, 0.1, 0.2, 0.3])
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


def LinUCB(lin_gaussian_bandit, n, delta):
    ...

if __name__ == '__main__':
    algo = input("Algorithm name: ")

    if algo == "FTL":
        FTLRun()

    elif algo == "ETC2":
        ETC2Run()

    elif algo == "UCB":
        UCBRun()

    else:
        print("Invalid algo name")
