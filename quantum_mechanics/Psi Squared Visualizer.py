import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import sys

# エラー回避用設定
try:
    import matplotlib
    matplotlib.use('TkAgg') 
except:
    pass

# ==========================================
# 物理定数とパラメータ設定
# ==========================================
hbar = 1.0545718e-34
m_e = 9.10938356e-31
eV_to_J = 1.60218e-19

# パラメータ（n=20程度出る設定）
# パラメータ（n=6程度に戻す）
V0_eV = 20.0
width_nm = 0.5

V0 = V0_eV * eV_to_J
a = (width_nm / 2) * 1e-9
R = (np.sqrt(2 * m_e * V0) * a) / hbar

print(f"井戸の深さ V0: {V0_eV} eV")
print(f"井戸の幅 2a: {width_nm} nm")
print(f"パラメータ R: {R:.4f}")

def find_energies():
    energies = []
    
    def func_even(xi):
        if xi == 0: return -R 
        return xi * np.tan(xi) - np.sqrt(R**2 - xi**2)

    def func_odd(xi):
        if xi == 0 or np.sin(xi) == 0: return np.inf 
        return -xi * (1/np.tan(xi)) - np.sqrt(R**2 - xi**2)

    n_points = 2000
    xis = np.linspace(1e-4, R - 1e-4, n_points)
    
    for i in range(len(xis)-1):
        if func_even(xis[i]) * func_even(xis[i+1]) < 0:
            try:
                root = brentq(func_even, xis[i], xis[i+1])
                E = (root * hbar / a)**2 / (2 * m_e)
                energies.append(("Even", E))
            except: pass

    for i in range(len(xis)-1):
        if func_odd(xis[i]) * func_odd(xis[i+1]) < 0 and np.abs(func_odd(xis[i])) < 100:
            try:
                root = brentq(func_odd, xis[i], xis[i+1])
                E = (root * hbar / a)**2 / (2 * m_e)
                energies.append(("Odd", E))
            except: pass
            
    energies.sort(key=lambda x: x[1])
    return energies

found_energies = find_energies()

# ==========================================
# グラフ描画
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 10))

# --- 左図：図式解法 ---
ax1 = axes[0]
xi_plot = np.linspace(0, R*1.1, 1000)

with np.errstate(divide='ignore', invalid='ignore'):
    eta_circle = np.sqrt(np.maximum(0, R**2 - xi_plot**2))
    eta_even = xi_plot * np.tan(xi_plot)
    eta_odd = -xi_plot / np.tan(xi_plot)

eta_even[eta_even > R*1.5] = np.nan
eta_even[eta_even < 0] = np.nan
eta_odd[eta_odd > R*1.5] = np.nan
eta_odd[eta_odd < 0] = np.nan

ax1.plot(xi_plot, eta_circle, 'k-', lw=2, label='Circle')
ax1.plot(xi_plot, eta_even, 'b--', alpha=0.5, label='Even')
ax1.plot(xi_plot, eta_odd, 'r--', alpha=0.5, label='Odd')

for parity, E in found_energies:
    xi_sol = np.sqrt(2 * m_e * E) * a / hbar
    eta_sol = np.sqrt(R**2 - xi_sol**2)
    ax1.plot(xi_sol, eta_sol, 'go', ms=5)

ax1.set_ylim(0, R*1.1)
ax1.set_xlim(0, R*1.1)
ax1.set_xlabel(r'$\xi = \alpha a$', fontsize=12)
ax1.set_ylabel(r'$\eta = \beta a$', fontsize=12)
ax1.set_title(f'Graphical Solution (N={len(found_energies)})', fontsize=14)
ax1.legend(loc='upper right')
ax1.grid(True)

# --- 右図：確率密度 ---
ax2 = axes[1]
x = np.linspace(-a*1.5, a*1.5, 1000)

V_plot = np.where(np.abs(x) < a, 0, V0_eV)
ax2.plot(x * 1e9, V_plot, 'k-', lw=3, alpha=0.3, label='Potential')

colors = plt.cm.viridis(np.linspace(0, 1, len(found_energies)))

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
        
    else: # Odd
        amp_out = np.sin(k*a) * np.exp(kappa*a)
        mask_in = np.abs(x) <= a
        psi[mask_in] = np.sin(k * x[mask_in])
        psi[x > a] = amp_out * np.exp(-kappa * x[x > a])
        psi[x < -a] = -amp_out * np.exp(kappa * x[x < -a])

    # 【修正箇所】NumPy 2.0対応: np.trapz -> np.trapezoid
    try:
        integral = np.trapezoid(np.abs(psi)**2, x)
    except AttributeError:
        # 古いNumPyを使っている場合のフォールバック
        integral = np.trapz(np.abs(psi)**2, x)
        
    psi_norm = psi / np.sqrt(integral)
    
    prob_density = np.abs(psi_norm)**2
    
    scale_factor_prob = V0_eV * 2.0e-10 
    
    ax2.plot(x * 1e9, prob_density * scale_factor_prob + E_eV, color=colors[i], lw=1.5)
    ax2.hlines(E_eV, -a*1.5*1e9, a*1.5*1e9, colors='gray', linestyles=':', alpha=0.3)

ax2.set_xlabel('Position x (nm)', fontsize=12)
ax2.set_ylabel('Energy (eV)', fontsize=12)
# 【修正箇所】警告回避のため fr文字列 に変更
ax2.set_title(fr'Probability Densities $|\psi|^2$ (n=1 to {len(found_energies)})', fontsize=14)
ax2.grid(True)

plt.tight_layout()
plt.show()

print(f"\n=== 計算されたエネルギー固有値 (全{len(found_energies)}個) ===")
for i in range(min(5, len(found_energies))):
    p, E = found_energies[i]
    print(f"n={i+1:02} [{p}]: {E/eV_to_J:.4f} eV")