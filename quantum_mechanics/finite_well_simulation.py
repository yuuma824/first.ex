import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

hbar = 1.0545718e-34  
m_e = 9.10938356e-31  
eV_to_J = 1.60218e-19 

V0_eV = 20.0          
width_nm = 0.5        

V0 = V0_eV * eV_to_J
a = (width_nm / 2) * 1e-9  
R = (np.sqrt(2 * m_e * V0) * a) / hbar 

print(f"井戸の深さ V0: {V0_eV} eV")
print(f"パラメータ R: {R:.4f} (R < π/2なら解は1つ)")

def find_energies():
    energies = []
    
    def func_even(xi):
        if xi == 0: return -R 
        return xi * np.tan(xi) - np.sqrt(R**2 - xi**2)

    def func_odd(xi):
        if xi == 0 or np.sin(xi) == 0: return np.inf 
        return -xi * (1/np.tan(xi)) - np.sqrt(R**2 - xi**2)

    n_points = 1000
    xis = np.linspace(1e-4, R - 1e-4, n_points)
    
    for i in range(len(xis)-1):
        if func_even(xis[i]) * func_even(xis[i+1]) < 0:
            root = brentq(func_even, xis[i], xis[i+1])
            E = (root * hbar / a)**2 / (2 * m_e)
            energies.append(("Even", E))

    for i in range(len(xis)-1):
        
        if func_odd(xis[i]) * func_odd(xis[i+1]) < 0 and np.abs(func_odd(xis[i])) < 100:
            root = brentq(func_odd, xis[i], xis[i+1])
            E = (root * hbar / a)**2 / (2 * m_e)
            energies.append(("Odd", E))
            
    energies.sort(key=lambda x: x[1])
    return energies

found_energies = find_energies()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
xi_plot = np.linspace(0, R*1.1, 500)
eta_circle = np.sqrt(np.maximum(0, R**2 - xi_plot**2)) 

eta_even = xi_plot * np.tan(xi_plot)
eta_odd = -xi_plot / np.tan(xi_plot)

eta_even[eta_even > R*1.5] = np.nan
eta_even[eta_even < 0] = np.nan
eta_odd[eta_odd > R*1.5] = np.nan
eta_odd[eta_odd < 0] = np.nan

ax1.plot(xi_plot, eta_circle, 'k-', lw=2, label=r'Circle ($R^2 = \xi^2+\eta^2$)')
ax1.plot(xi_plot, eta_even, 'b--', label=r'Even ($\eta = \xi \tan \xi$)')
ax1.plot(xi_plot, eta_odd, 'r--', label=r'Odd ($\eta = -\xi \cot \xi$)')


for parity, E in found_energies:
    xi_sol = np.sqrt(2 * m_e * E) * a / hbar
    eta_sol = np.sqrt(R**2 - xi_sol**2)
    ax1.plot(xi_sol, eta_sol, 'go', ms=8)

ax1.set_ylim(0, R*1.2)
ax1.set_xlim(0, R*1.2)
ax1.set_xlabel(r'$\xi = \alpha a$', fontsize=12)
ax1.set_ylabel(r'$\eta = \beta a$', fontsize=12)
ax1.set_title('Graphical Solution for Eigenvalues', fontsize=14)
ax1.legend()
ax1.grid(True)

ax2 = axes[1]
x = np.linspace(-a*2.5, a*2.5, 500)

V_plot = np.where(np.abs(x) < a, 0, V0_eV)
ax2.plot(x * 1e9, V_plot, 'k-', lw=2, alpha=0.5, label='Potential $V(x)$')

for i, (parity, E) in enumerate(found_energies):
    E_eV = E / eV_to_J
    k = np.sqrt(2 * m_e * E) / hbar      
    kappa = np.sqrt(2 * m_e * (V0 - E)) / hbar 
    
    psi = np.zeros_like(x)
    
    if parity == "Even":
        
        amp_out = np.cos(k*a) * np.exp(kappa*a)        
        mask_in = np.abs(x) <= a
        psi[mask_in] = np.cos(k * x[mask_in])
        psi[~mask_in] = amp_out * np.exp(-kappa * np.abs(x[~mask_in]))
        
    else:
       
        amp_out = np.sin(k*a) * np.exp(kappa*a)
        
        mask_in = np.abs(x) <= a
        psi[mask_in] = np.sin(k * x[mask_in])
        
        psi[x > a] = amp_out * np.exp(-kappa * x[x > a])
        
        psi[x < -a] = -amp_out * np.exp(kappa * x[x < -a])

    scale_factor = V0_eV * 0.15 
    ax2.plot(x * 1e9, psi * scale_factor + E_eV, label=f'n={i+1} ({E_eV:.2f} eV)')
    ax2.hlines(E_eV, -a*2.5*1e9, a*2.5*1e9, colors='gray', linestyles=':', alpha=0.5)

ax2.set_xlabel('Position x (nm)', fontsize=12)
ax2.set_ylabel('Energy (eV)', fontsize=12)
ax2.set_title(f'Wavefunctions (V0={V0_eV}eV)', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()

print("\n=== 計算されたエネルギー固有値 ===")
for i, (parity, E) in enumerate(found_energies):
    print(f"n={i+1} [{parity}]: {E/eV_to_J:.4f} eV")