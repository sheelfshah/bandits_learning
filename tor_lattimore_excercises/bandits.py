import numpy as np


class BernoulliBandit:

    # accepts a list of K >= 2 floats , each lying in [0 ,1]
    def __init__(self, means):
        self.arm_means = means
        self.pseudo_regret = 0

    # Function should return the number of arms
    def K(self):
        return len(self.arm_means)

    # Accepts a parameter 0 <= a <= K -1 and returns the
    # realisation of random variable X with P ( X = 1) being
    # the mean of the (a +1) th arm .
    def pull(self, a):
        reward = np.random.binomial(size=1, n=1, p=self.arm_means[a])
        self.pseudo_regret += (max(self.arm_means) -
                               self.arm_means[a])  # pseudo regret
        return reward.item()

    # Returns the regret incurred so far .
    def regret(self):
        return self.pseudo_regret


if __name__ == '__main__':
    bernoulli_bandit = BernoulliBandit([0.1, 0.3, 0.5])
    print(bernoulli_bandit.K())
    print(bernoulli_bandit.pull(0))
    print(bernoulli_bandit.pull(1))
    print(bernoulli_bandit.pull(2))
    print(bernoulli_bandit.regret())
