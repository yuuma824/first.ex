import numpy as np
import matplotlib.pyplot as plt

def initialize_spins(N):
    """N x Nのグリッドにランダムなスピン(+1 or -1)を配置"""
    return np.random.choice([-1, 1], size=(N, N))

def mcmc_step(spins, beta, J=1.0):
    """メトロポリス法による1ステップの更新"""
    N = spins.shape[0]
    for _ in range(N * N):
        
        i, j = np.random.randint(0, N, 2)
        s = spins[i, j]
        
        neighbors = spins[(i+1)%N, j] + spins[(i-1)%N, j] + \
                    spins[i, (j+1)%N] + spins[i, (j-1)%N]
        
        delta_E = 2.0 * J * s * neighbors
        
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            spins[i, j] = -s
    return spins

N = 50           
T = 2.0          
beta = 1.0 / T   
steps = 100      

spins = initialize_spins(N)

for step in range(steps):
    spins = mcmc_step(spins, beta)

plt.figure(figsize=(6, 6))
plt.imshow(spins, cmap='coolwarm', interpolation='nearest')
plt.title(f'2D Ising Model (T = {T}) after {steps} steps')
plt.axis('off')
plt.show()