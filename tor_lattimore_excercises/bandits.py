import numpy as np


class BernoulliBandit:

    # accepts a list of K >= 2 floats , each lying in [0 ,1]
    def __init__(self, means):
        self.arm_means = means
        self.actual_regret = 0

    # Function should return the number of arms
    def K(self):
        return len(self.arm_means)

    # Accepts a parameter 0 <= a <= K -1 and returns the
    # realisation of random variable X with P ( X = 1) being
    # the mean of the (a +1) th arm .
    def pull(self, a):
        reward = np.random.binomial(size=1, n=1, p=self.arm_means[a])
        self.actual_regret += (max(self.arm_means) -
                               reward)  # regret
        return reward.item()

    # Returns the regret incurred so far .
    def regret(self):
        return self.actual_regret


class GaussianBandit:
    # gaussian arms, and actual regret

    # accepts a list of K >= 2 floats
    def __init__(self, means):
        self.arm_means = means
        self.actual_regret = 0

    # Function should return the number of arms
    def K(self):
        return len(self.arm_means)

    # Accepts a parameter 0 <= a <= K -1 and returns the
    # realisation of random variable X with gaussian(mean = ua, std=1)
    def pull(self, a):
        reward = np.random.normal(loc=self.arm_means[a], scale=1)
        self.actual_regret += (max(self.arm_means) - reward)  # regret
        return reward

    # pulls the same arm repeatedly, to speed things up
    def pull_repeatedly(self, a, num_reps):
        rewards = np.random.normal(
            loc=self.arm_means[a], scale=1, size=(num_reps, ))
        self.actual_regret += num_reps * \
            max(self.arm_means) - np.sum(rewards)  # regret
        return rewards

    # Returns the regret incurred so far .
    def regret(self):
        return self.actual_regret


if __name__ == '__main__':
    bernoulli_bandit = BernoulliBandit([0.1, 0.3, 0.5])
    print(bernoulli_bandit.K())
    print(bernoulli_bandit.pull(0))
    print(bernoulli_bandit.pull(1))
    print(bernoulli_bandit.pull(2))
    print(bernoulli_bandit.regret())
