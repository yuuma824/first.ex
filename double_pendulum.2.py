import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def double_pendulum(y, t, L1, L2, m1, m2, g):
    theta1, z1, theta2, z2 = y  
    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    den1 = m1 + m2 * s**2
    den2 = (L2/L1) * den1

    d_theta1 = z1
    d_z1 = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) - (m1+m2)*g*np.sin(theta1)) / (L1*den1)
    
    d_theta2 = z2
    d_z2 = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + m2*L2*z2**2*s*c) / (L2*den1)

    return [d_theta1, d_z1, d_theta2, d_z2]

L1, L2 = 1.0, 1.0  
m1, m2 = 1.0, 1.0  
g = 9.81
t = np.linspace(0, 20, 1000) 

y0 = [np.pi - 0.1, 0, np.pi - 0.1, 0]

sol = odeint(double_pendulum, y0, t, args=(L1, L2, m1, m2, g))

x1 = L1 * np.sin(sol[:, 0])
y1 = -L1 * np.cos(sol[:, 0])
x2 = x1 + L2 * np.sin(sol[:, 2])
y2 = y1 - L2 * np.cos(sol[:, 2])

plt.figure(figsize=(6, 6))
plt.plot(x2, y2, lw=0.5, color='blue', label='Trajectory')
plt.title('Double Pendulum Chaos')
plt.xlim(-2.2, 2.2)
plt.ylim(-2.2, 2.2)
plt.grid()
plt.show()