# Verifiable-ITHSS

### 🚀 Introduction
This repository contains the C++ implementation and benchmarking suite for **Verifiable Information-Theoretic Homomorphic Secret Sharing (vITHSS)** protocol. It enable a client to outsource multivariate polynomial evaluations to $k$ untrusted servers while ensuring:
1. **Privacy**: Input $\mathbf{x}$ remains hidden from servers via Secret Sharing.
2. **Verifiability**: Correctness is guaranteed using Algebraic Derivatives and Hermite Interpolation.

### 📂 File Descriptions
* **`vit_hss_flint.cpp`**: Information-Theoretic (IT) implementation. Based on the FLINT library, achieving high-speed algebraic operations with polynomial communication cost $O(m^\ell)$.


### 🛠️ Dependencies
* **GMP**: GNU Multi-Precision library.
* **FLINT**: Fast Library for Number Theory (matrix and field operations).

#### Installation (Ubuntu/WSL):
```bash
# Install GMP and FLINT
sudo apt update
sudo apt install -y libflint-dev libgmp-dev
```

### 🔨 Compilation & Usage
#### Run IT-based Protocol
```bash
g++ -O3 vit_hss_flint.cpp -o vit_hss -lflint -lgmp
./vit_hss
```


---

### 📊 Benchmark Dimensions (4D Sweep)
The code performs a comprehensive scan across the following parameters:
* **Sparsity**: Comparison between 10% (realistic sparse) and 100% (dense worst-case).
* **Degree ($d$)**: Polynomial degrees from 2 to 4.
* **Derivative Order ($\ell$)**: Including Function value ($\ell=0$), Gradient ($\ell=1$), and Hessian ($\ell=2$).
* **Variables ($m$)**: Dimension scale from 10 to 100.

All parameters satisfy the theoretical constraint: $$(\ell+1)k \ge dt+1$$.
