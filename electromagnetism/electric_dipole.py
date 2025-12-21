import numpy as np
import matplotlib.pyplot as plt

def plot_electric_dipole():
    """
    電気双極子（+qと-q）が作る電場と等電位線の描画
    """
    # 空間の定義
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # 電荷の位置と大きさ [[x, y, q], ...]
    charges = [
        [-0.5, 0,  1.0], # プラス電荷 (+q)
        [ 0.5, 0, -1.0]  # マイナス電荷 (-q)
    ]
    
    # 電位 V と 電場 E の初期化
    V = np.zeros_like(X)
    Ex = np.zeros_like(X)
    Ey = np.zeros_like(X)
    
    # クーロンの法則定数 k (シミュレーション用なので1とする)
    k = 1.0
    
    for cx, cy, q in charges:
        # 距離 r を計算
        dx = X - cx
        dy = Y - cy
        r = np.sqrt(dx**2 + dy**2)
        
        # ゼロ除算を防ぐための微小量
        r[r < 0.1] = 0.1
        
        # 電位 V = k * q / r
        V += k * q / r
        
        # 電場 E = k * q / r^2 * (単位ベクトル)
        # E = k * q * r_vec / r^3
        Ex += k * q * dx / r**3
        Ey += k * q * dy / r**3

    plt.figure(figsize=(8, 8))
    
    # 電場の流線プロット（Streamplot: 電力線のような表示）
    # 色は電場の強さ(log scale)で変化させる
    E_strength = np.sqrt(Ex**2 + Ey**2)
    strm = plt.streamplot(X, Y, Ex, Ey, color=np.log(E_strength), linewidth=1, cmap='autumn', density=1.5)
    
    # 等電位線のプロット（Contour）
    contours = plt.contour(X, Y, V, levels=20, colors='black', alpha=0.5, linestyles='solid')
    plt.clabel(contours, inline=True, fontsize=8)
    
    # 電荷の位置を点で表示
    for cx, cy, q in charges:
        color = 'red' if q > 0 else 'blue'
        label = '+' if q > 0 else '-'
        plt.scatter(cx, cy, s=200, c=color, edgecolors='black', zorder=10)
        plt.text(cx, cy, label, color='white', ha='center', va='center', fontsize=15, fontweight='bold')

    plt.title("Electric Field & Potential (Dipole)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_electric_dipole()