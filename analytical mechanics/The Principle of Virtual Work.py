import numpy as np
import matplotlib.pyplot as plt

m = 1.0      
g = 9.8       
F = 5.0       
L = 1.0       
d_theta = 0.05 

thetas = np.linspace(-np.pi/2, np.pi/2, 100)

delta_W = (F * L * np.cos(thetas) - m * g * L * np.sin(thetas)) * d_theta

theta_eq = np.arctan(F / (m * g))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(np.degrees(thetas), delta_W, label=r'Virtual Work $\delta W$')
ax1.axhline(0, color='black', linestyle='--')
ax1.axvline(np.degrees(theta_eq), color='red', linestyle=':', 
            label=f'Equilibrium $\\theta_{{eq}} \\approx {np.degrees(theta_eq):.1f}^\\circ$')
ax1.set_xlabel(r'Angle $\theta$ (degrees)')
ax1.set_ylabel(r'Virtual Work $\delta W$')
ax1.set_title('Virtual Work vs Angle')
ax1.legend()
ax1.grid(True)

x_eq = L * np.sin(theta_eq)
y_eq = -L * np.cos(theta_eq)

ax2.plot([0, x_eq], [0, y_eq], 'k-', lw=2)  
ax2.plot(0, 0, 'ko', markersize=8)         
ax2.plot(x_eq, y_eq, 'bo', markersize=10)  

scale_factor = 40
ax2.quiver(x_eq, y_eq, 0, -m*g, color='green', scale=scale_factor, width=0.01, label='Gravity $mg$')
ax2.quiver(x_eq, y_eq, F, 0, color='orange', scale=scale_factor, width=0.01, label='Horizontal Force $F$')

ax2.quiver(x_eq, y_eq, F, -m*g, color='red', scale=scale_factor, width=0.012, label='Total Active Force')

dr_dir = np.array([L * np.cos(theta_eq), L * np.sin(theta_eq)])
dr_dir = dr_dir / np.linalg.norm(dr_dir)
ax2.quiver(x_eq, y_eq, dr_dir[0], dr_dir[1], color='purple', scale=5, width=0.01, label=r'Virtual Displacement $\delta \mathbf{r}$')

circle = plt.Circle((0, 0), L, color='gray', fill=False, linestyle='--')
ax2.add_patch(circle)

ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.5, 0.5)
ax2.set_aspect('equal')
ax2.set_title('Forces and Virtual Displacement at Equilibrium')
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True)

plt.tight_layout()
plt.savefig('virtual_work.png')