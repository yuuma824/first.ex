import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def spring_mass_system(y, t, k, m, c):
    x, v = y 
    dydt = [v, -(k/m)*x - (c/m)*v]
    return dydt

k = 10.0   
m = 1.0   
c = 0.5

y0 = [1.0, 0.0] 
t = np.linspace(0, 20, 200)  
sol = odeint(spring_mass_system, y0, t, args=(k, m, c))

plt.plot(t, sol[:, 0], label='Position (x)')
plt.title('Damped Harmonic Motion (Spring-Mass)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.axhline(0, color='black', linewidth=0.5) 
plt.grid()
plt.legend()
plt.show()