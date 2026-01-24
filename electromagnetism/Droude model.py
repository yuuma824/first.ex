import numpy as np
import matplotlib.pyplot as plt

c = 3.0e8  
tau = 3.0e-14  
omega_p = 1.0e15  

wavelength_nm = np.linspace(10, 3000, 1000)
wavelength_m = wavelength_nm * 1.0e-9

omega = 2 * np.pi * c / wavelength_m

epsilon = 1 - (omega_p**2) / (omega * (omega + 1j / tau))

N = np.sqrt(epsilon)

R = np.abs((1 - N) / (1 + N))**2

plt.figure(figsize=(10, 6))
plt.plot(wavelength_nm, R, label='Reflectivity of Copper (Drude Model)', color='#b87333') 

lambda_p_nm = (2 * np.pi * c / omega_p) * 1e9
plt.axvline(x=lambda_p_nm, color='k', linestyle='--', alpha=0.5, label=f'Plasma Wavelength ~{lambda_p_nm:.0f} nm')

plt.xscale('log') 
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Reflectivity R', fontsize=12)
plt.title('Reflectivity of Copper vs Wavelength', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.ylim(0, 1.05)

plt.show()