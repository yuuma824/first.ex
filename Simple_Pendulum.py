import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def pendulum_equations(y, t, g, L):
    theta, omega = y
    dydt = [omega, -(g/L) * np.sin(theta)]
    return dydt

L = 1.0  
g = 9.8  
y0 = [np.pi/4, 0.0] 
t = np.linspace(0, 10, 101)  

sol = odeint(pendulum_equations, y0, t, args=(g, L))

plt.plot(t, sol[:, 0])
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.grid()
plt.show()