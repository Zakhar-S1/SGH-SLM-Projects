import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def sim_r2_k(n, k, reps):
    x = np.ones((n, k+1))
    y = np.zeros(n)
    result = list()
    
    for _ in range(0, reps):

        view = x[:, :k]

        view[:] = np.random.randn(view.shape[0], view.shape[1])
        np.sum(x, axis=1, out=y)
        
        for i in range(0, n):
            y[i] += np.random.randn()
        
        model = LinearRegression().fit(view, np.array(y).reshape(-1, 1))
        result.append(model.score(view, np.array(y).reshape(-1, 1)))
        
    return result

def run(k, reps=10000, maxn=200):
    sizes = range(10, maxn+1, 10)
    
    r2_q95 = list()
    r2_q5 = list()
    r2_mean = list()

    for s in sizes:
        result = sim_r2_k(s, k, reps)
        r2_mean.append(np.mean(result))
        r2_q5.append(np.quantile(result, 0.05))
        r2_q95.append(np.quantile(result, 0.95))
        
    plt.scatter(sizes, r2_mean)
    plt.xlabel("sample size")
    plt.ylabel("R²")
    plt.plot(sizes, r2_q5, color="black")
    plt.plot(sizes, r2_q95, color="black")

run(k=10)
plt.show()