import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
import math

def harmonic_oscillator(n, x):
    """
    量子調和振動子の波動関数を計算する関数
    n: 量子数 (0, 1, 2, ...)
    x: 位置
    """
    # 正規化定数
    N = 1.0 / math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
    # エルミート多項式 (scipyを使用)
    Hn = hermite(n)
    
    # 波動関数 psi = N * Hn(x) * exp(-x^2 / 2)
    psi = N * Hn(x) * np.exp(-x**2 / 2)
    return psi

# 設定
x = np.linspace(-4, 4, 1000)  # -4から4までの範囲
n_states = 3  # 表示する状態数 (n=0, 1, 2)

# プロットの準備
plt.figure(figsize=(8, 6))
plt.title('Quantum Harmonic Oscillator')
plt.xlabel('Position x')
plt.ylabel('Energy / Amplitude')

# ポテンシャルエネルギー V(x) = 0.5 * x^2 の描画（背景の放物線）
V = 0.5 * x**2
plt.plot(x, V, 'k--', linewidth=1, label='Potential $V(x)$')

# 各状態のプロット
for n in range(n_states):
    psi = harmonic_oscillator(n, x)
    probability = psi**2
    
    # エネルギー準位 E_n = n + 0.5 に合わせてグラフをずらす
    energy_level = n + 0.5
    
    # 波動関数をスケーリングしてエネルギー準位の上に描画
    scale = 0.8 # 見やすくするための倍率
    
    # 波動関数の描画 (実線) - raw文字列(r)を使って警告回避！
    plt.plot(x, psi * scale + energy_level, label=rf'$n={n}, E={energy_level}$')
    
    # 確率密度の塗りつぶし
    plt.fill_between(x, energy_level, probability * scale + energy_level, alpha=0.3)

plt.legend(loc='upper right')
plt.ylim(0, 4.5) # Y軸の表示範囲
plt.grid(True, alpha=0.3)

# グラフ表示
plt.show()