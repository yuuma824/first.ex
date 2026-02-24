import numpy as np
import matplotlib.pyplot as plt

V = np.linspace(1, 10, 100)  
P_initial = 10.0             
V_initial = 1.0              
gamma = 1.4                  

const_iso = P_initial * V_initial          
const_adi = P_initial * (V_initial**gamma) 

P_isothermal = const_iso / V
P_adiabatic = const_adi / (V**gamma)

plt.figure(figsize=(8, 5))
plt.plot(V, P_isothermal, label='Isothermal (等温変化: $PV = const$)', color='blue')
plt.plot(V, P_adiabatic, label='Adiabatic (断熱変化: $PV^\gamma = const$)', color='red')

plt.title('Thermodynamics: P-V Diagram of an Ideal Gas')
plt.xlabel('Volume $V$')
plt.ylabel('Pressure $P$')
plt.legend()
plt.grid(True)
plt.show()