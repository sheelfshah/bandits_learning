import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
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
