import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 50                
T_start = 4.0          
T_end = 1.0            
frames = 200          
sweeps_per_frame = 2   

T_array = np.linspace(T_start, T_end, frames)

spins = np.random.choice([-1, 1], size=(N, N))

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(spins, cmap='coolwarm', animated=True, interpolation='nearest')
title = ax.set_title('')
ax.axis('off')

def update(frame):
    global spins
    T = T_array[frame]
    beta = 1.0 / T
    
    for _ in range(sweeps_per_frame):
        for _ in range(N * N):
            i, j = np.random.randint(0, N, 2)
            s = spins[i, j]
            
            neighbors = spins[(i+1)%N, j] + spins[(i-1)%N, j] + \
                        spins[i, (j+1)%N] + spins[i, (j-1)%N]
            
            dE = 2.0 * s * neighbors
            if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                spins[i, j] = -s
                
    im.set_array(spins)
    title.set_text(f'Temperature T = {T:.2f}')
    return im, title

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

plt.tight_layout()

print("保存中...（少し時間がかかります）")
ani.save('ising_model.gif', writer='pillow', fps=30) 
print("保存完了！")

plt.show()