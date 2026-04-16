# Verifiable-HSS (Local Simulation & Benchmark)

### 🚀 Introduction
This repository contains the C++ implementation and benchmarking suite for **Verifiable Homomorphic Secret Sharing (vHSS)** protocols. These protocols enable a client to outsource multivariate polynomial evaluations to $k$ untrusted servers while ensuring:
1. **Privacy**: Input $\mathbf{x}$ remains hidden from servers via Secret Sharing.
2. **Verifiability**: Correctness is guaranteed using Algebraic Derivatives and Hermite Interpolation.

### 📂 File Descriptions
* **`vit_hss_flint.cpp`**: Information-Theoretic (IT) implementation. Based on the FLINT library, achieving high-speed algebraic operations with polynomial communication cost $O(m^\ell)$.
* **`vhss_seal.cpp`**: Homomorphic Encryption (HE) optimized implementation. Integrated with **Microsoft SEAL (BFV scheme)**, it reduces the client's download cost to a constant $O(\ell)$, effectively decoupling it from the variable dimension $m$.

### 🛠️ Dependencies
* **GMP**: GNU Multi-Precision library.
* **FLINT**: Fast Library for Number Theory (matrix and field operations).
* **Microsoft SEAL (v4.1)**: Standard library for homomorphic encryption.

#### Installation (Ubuntu/WSL):
```bash
# Install GMP and FLINT
sudo apt update
sudo apt install -y libflint-dev libgmp-dev

# Install Microsoft SEAL (Building from source)
git clone -b v4.1.1 https://github.com/microsoft/SEAL.git
cd SEAL
cmake -S . -B build -DSEAL_USE_ZLIB=OFF
cmake --build build -j
sudo cmake --install build
```

### 🔨 Compilation & Usage
#### 1. Run IT-based Protocol
```bash
g++ -O3 vit_hss_flint.cpp -o vit_hss -lflint -lgmp
./vit_hss
```
#### 2. Run HE-optimized Protocol
```bash
g++ -O3 vhss_seal.cpp -o vhss_seal -lflint -lgmp -lseal-4.1 -I/usr/local/include/SEAL-4.1 -std=c++17
./vhss_seal
```
