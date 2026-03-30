# Verifiable-ITHSS (Local Simulation & Benchmark)

### 🚀 Introduction
This repository contains the C++ implementation and benchmarking suite for the **Verifiable Information-Theoretic Homomorphic Secret Sharing (vITHSS)** protocol. 
> **Note**: This is a **single-machine simulation framework**. It implements the full cryptographic logic (Share, Eval, Dec) to benchmark computational complexity and verify correctness, simulating the client-server architecture locally without network overhead.

The protocol enables a client to outsource multivariate polynomial evaluations to $k$ untrusted servers while ensuring:
1. **Privacy**: Input $\mathbf{x}$ remains hidden from servers via Secret Sharing (simulated via memory partitioning).
2. **Verifiability**: Correctness is guaranteed using Algebraic Derivatives and Hermite Interpolation.

### 📂 File Descriptions
* **`vit_hss_flint.cpp`**: Information-Theoretic (IT) implementation. Based on the FLINT library, achieving high-speed algebraic operations. 
  * *Note*: All phases (Share, Eval, Dec) are executed sequentially in this binary to measure pure algebraic cost.

### ⚠️ Implementation Mode: Local Simulation
To ensure accurate measurement of computational complexity $O(m^\ell)$ without network latency noise:
* **Deployment**: All algorithms run on a **single machine** within one process.
* **Share Phase**: Secret shares are generated and stored in local memory vectors (simulating transmission to $k$ servers).
* **Eval Phase**: Server-side computations are simulated sequentially.
* **Dec Phase**: Reconstruction is performed locally using the simulated shares.
* **Communication Cost**: Reported communication costs are **theoretical calculations** based on share sizes, not actual socket traffic.

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
Run Local Simulation:
```bash
g++ -O3 vit_hss_flint.cpp -o vit_hss -lflint -lgmp
./vit_hss
```
