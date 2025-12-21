import numpy as np
import matplotlib.pyplot as plt

def solve_quantum_well(L, n_max):
    """
    1次元無限井戸型ポテンシャルの波動関数と確率密度を計算
    L: 井戸の幅
    n_max: 表示する量子数の最大値
    """
    # x軸の定義（井戸の中）
    x = np.linspace(0, L, 1000)
    
    plt.figure(figsize=(10, 6))
    
    # 各量子数 n についてプロット
    for n in range(1, n_max + 1):
        # 波動関数 psi_n(x) = sqrt(2/L) * sin(n * pi * x / L)
        psi = np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)
        
        # 確率密度 |psi|^2
        prob_density = psi**2
        
        # グラフ描画（少し上にずらして表示）
        offset = 2.5 * (n - 1) # 見やすくするためにy軸方向にずらす
        
        # 波動関数のプロット（点線）
        plt.plot(x, psi + offset, '--', alpha=0.5, label=f'Wave function $\psi_{n}$')
        
        # 確率密度のプロット（塗りつぶし）
        plt.plot(x, prob_density + offset, '-', linewidth=2, label=f'Probability $|\psi_{n}|^2$')
        plt.fill_between(x, offset, prob_density + offset, alpha=0.3)
        
        # 基準線
        plt.hlines(offset, 0, L, colors='black', linestyles='-', linewidth=0.5)
        plt.text(-0.1 * L, offset + 0.5, f'n={n}', fontsize=12, fontweight='bold')

    plt.title("Quantum Particle in a 1D Infinite Box")
    plt.xlabel("Position x")
    plt.ylabel("Wave Function / Probability Density (shifted)")
    plt.xlim(0, L)
    # y軸の目盛りは意味がないので消す
    plt.yticks([])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 井戸の幅 L=1.0, 量子数 n=1から3まで表示
    solve_quantum_well(L=1.0, n_max=3)