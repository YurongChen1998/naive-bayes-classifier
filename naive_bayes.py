from sklearn.datasets import make_blobs
import numpy as np
from scipy.stats import norm

def fit_distribution(data):
    mu = np.mean(data)
    sigma = np.std(data)
    dist = norm(mu, sigma)
    return dist

def probability(X, piror, dist1, dist2):
    return piror * dist1.pdf(X[0]) * dist1.pdf(X[0])

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
X0 = X[y == 0]
X1 = X[y == 1]

prior_0 = len(X0) / len(X)
prior_1 = len(X1) / len(X)

p_x0_given_y0,  p_x1_given_y0 = fit_distribution(X0[:, 0]), fit_distribution(X0[:, 1])
p_x0_given_y1,  p_x1_given_y1 = fit_distribution(X1[:, 0]), fit_distribution(X1[:, 1])

print("The prob. of class 0:", probability(X[2], prior_0, p_x0_given_y0, p_x1_given_y0) * 100)
print("The prob. of class 1:", probability(X[2], prior_1, p_x0_given_y1, p_x1_given_y1) * 100)
print("The predictive class is:", 0 if probability(X[2], prior_0, p_x0_given_y0, p_x1_given_y0) > probability(X[2], prior_1, p_x0_given_y1, p_x1_given_y1) else 1)
print("The ground turth is:", y[2])
