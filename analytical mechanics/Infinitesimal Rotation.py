import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

r = np.array([1.0, 0.5, 0.2])          
d_theta = np.array([0.0, 0.0, 0.4])    
dr = np.cross(d_theta, r)              
r_new = r + dr                         

origin = np.array([0, 0, 0])

def draw_vector(ax, start, vec, color, label):
    ax.quiver(start[0], start[1], start[2], vec[0], vec[1], vec[2], 
              color=color, arrow_length_ratio=0.1, label=label, linewidth=2)

draw_vector(ax, origin, r, 'blue', r'Position Vector $\mathbf{r}$')
draw_vector(ax, origin, d_theta, 'green', r'Rotation Axis $\delta\boldsymbol{\theta}$')
draw_vector(ax, r, dr, 'red', r'Infinitesimal Displacement $\delta\mathbf{r} = \delta\boldsymbol{\theta} \times \mathbf{r}$')
draw_vector(ax, origin, r_new, 'lightblue', r'New Position $\mathbf{r} + \delta\mathbf{r}$')

ax.set_xlim([-0.2, 1.2])
ax.set_ylim([-0.2, 1.2])
ax.set_zlim([0, 0.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Visualization of Infinitesimal Rotation')
ax.legend()
plt.show()