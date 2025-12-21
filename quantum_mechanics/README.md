# Physics Simulations (物理シミュレーション)

大学で学んだ物理現象をPythonを用いて数値計算・可視化するためのリポジトリです。
力学、量子力学、電磁気学などの分野ごとにコードを整理しています。

## 📂 ディレクトリ構成

### 1. Classical Mechanics (力学)
`classical_mechanics/`
- **simple_pendulum.py**: 単振り子の運動方程式を解き、角度の時間変化をプロットします。
- **double_pendulum.py**: カオス挙動を示す二重振り子のシミュレーションです。
- **damped_oscillation.py**: 減衰振動の様子を可視化します。

### 2. Quantum Mechanics (量子力学)
`quantum_mechanics/`
- **infinite_well.py**: 1次元無限井戸型ポテンシャル中の粒子の波動関数を描画します。
- **harmonic.py**: 量子調和振動子のエネルギー準位と確率密度を計算します。

### 3. Electromagnetism (電磁気学)
`electromagnetism/`
- **biot_savart.py**: ビオ・サバールの法則を用いた磁場の計算など。

---

## 📊 実行結果の例

### 量子調和振動子 (Quantum Harmonic Oscillator)
`quantum_mechanics/harmonic.py` の実行結果です。

![量子調和振動子のグラフ](quantum_mechanics/harmonic_result.png)
---

## 🛠️ 要件 (Requirements)
このシミュレーションを実行するには、以下のライブラリが必要です。

```bash
pip install numpy matplotlib scipy