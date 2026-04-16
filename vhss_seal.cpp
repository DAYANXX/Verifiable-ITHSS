#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <random>
#include <string>
#include <cmath>
#include <flint/flint.h>
#include <flint/nmod_mat.h>
#include <flint/nmod_vec.h>
#include <seal/seal.h>

using namespace std;
using namespace seal;


#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

// 全局参数
int m_vars = 5;       
int d_deg = 3;        
int t_thr = 1;        
int current_ell = 2;  
int k_srv = 4;        

std::mt19937_64 fast_rng(12345);

const mp_limb_t PRIME_LOCAL = 18446744073709551557ULL; 
nmod_t ctx_local; 

inline mp_limb_t fast_rand_mod_nonzero_local() {
    mp_limb_t r = fast_rng() % ctx_local.n;
    return r == 0 ? 1 : r;
}

inline mp_limb_t add_local(mp_limb_t a, mp_limb_t b) { return nmod_add(a, b, ctx_local); }
inline mp_limb_t mul_local(mp_limb_t a, mp_limb_t b) { return nmod_mul(a, b, ctx_local); }

struct PolyTerm_Local {
    mp_limb_t coeff;
    int vars[6]; 
};
vector<PolyTerm_Local> global_poly_local;

struct EvalResult_Local {
    mp_limb_t f_val;
    vector<mp_limb_t> grad;
    vector<vector<mp_limb_t>> hessian;
};


NOINLINE void eval_all_local(const vector<mp_limb_t>& x, int ell, EvalResult_Local& out) {
    out.f_val = 0;
    if (ell >= 1) out.grad.assign(m_vars, 0);
    if (ell >= 2) {
        out.hessian.resize(m_vars);
        for(int i = 0; i < m_vars; ++i) out.hessian[i].assign(m_vars, 0);
    }

    for (const auto& term : global_poly_local) {
        mp_limb_t c = term.coeff;
        int v[6];
        mp_limb_t xv[6];
        for (int i = 0; i < d_deg; ++i) {
            v[i] = term.vars[i];
            xv[i] = x[v[i]];
        }

        if (d_deg == 2) {
            out.f_val = add_local(out.f_val, mul_local(c, mul_local(xv[0], xv[1])));
            if (ell >= 1) {
                out.grad[v[0]] = add_local(out.grad[v[0]], mul_local(c, xv[1]));
                out.grad[v[1]] = add_local(out.grad[v[1]], mul_local(c, xv[0]));
            }
            if (ell >= 2) {
                out.hessian[v[0]][v[1]] = add_local(out.hessian[v[0]][v[1]], c);
                out.hessian[v[1]][v[0]] = add_local(out.hessian[v[1]][v[0]], c);
            }
        } else if (d_deg == 3) {
            mp_limb_t p01 = mul_local(xv[0], xv[1]);
            mp_limb_t p02 = mul_local(xv[0], xv[2]);
            mp_limb_t p12 = mul_local(xv[1], xv[2]);
            out.f_val = add_local(out.f_val, mul_local(c, mul_local(p01, xv[2])));
            if (ell >= 1) {
                out.grad[v[0]] = add_local(out.grad[v[0]], mul_local(c, p12));
                out.grad[v[1]] = add_local(out.grad[v[1]], mul_local(c, p02));
                out.grad[v[2]] = add_local(out.grad[v[2]], mul_local(c, p01));
            }
            if (ell >= 2) {
                mp_limb_t cx0 = mul_local(c, xv[0]);
                mp_limb_t cx1 = mul_local(c, xv[1]);
                mp_limb_t cx2 = mul_local(c, xv[2]);
                out.hessian[v[0]][v[1]] = add_local(out.hessian[v[0]][v[1]], cx2);
                out.hessian[v[1]][v[0]] = add_local(out.hessian[v[1]][v[0]], cx2);
                out.hessian[v[0]][v[2]] = add_local(out.hessian[v[0]][v[2]], cx1);
                out.hessian[v[2]][v[0]] = add_local(out.hessian[v[2]][v[0]], cx1);
                out.hessian[v[1]][v[2]] = add_local(out.hessian[v[1]][v[2]], cx0);
                out.hessian[v[2]][v[1]] = add_local(out.hessian[v[2]][v[1]], cx0);
            }
        } else {
            mp_limb_t term_val = c;
            for(int i=0; i<d_deg; ++i) term_val = mul_local(term_val, xv[i]);
            out.f_val = add_local(out.f_val, term_val);
        }
    }

    if (ell >= 2 && d_deg >= 2) {
        mp_limb_t inv_2 = nmod_inv(2, ctx_local);
        for (int i = 0; i < m_vars; ++i) {
            for (int j = i + 1; j < m_vars; ++j) {
                mp_limb_t avg = mul_local(add_local(out.hessian[i][j], out.hessian[j][i]), inv_2);
                out.hessian[i][j] = out.hessian[j][i] = avg;
            }
        }
    }
}

mp_limb_t PRIME_HE; 

inline mp_limb_t add_fast(mp_limb_t a, mp_limb_t b) { 
    unsigned __int128 r = (unsigned __int128)a + b; 
    return (mp_limb_t)(r >= PRIME_HE ? r - PRIME_HE : r); 
}
inline mp_limb_t mul_fast(mp_limb_t a, mp_limb_t b) { 
    return (mp_limb_t)(((unsigned __int128)a * b) % PRIME_HE); 
}
mp_limb_t power_fast(mp_limb_t base, mp_limb_t exp) {
    mp_limb_t res = 1;
    base = base % PRIME_HE;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul_fast(res, base);
        base = mul_fast(base, base);
        exp /= 2;
    }
    return res;
}

shared_ptr<SEALContext> seal_context;
unique_ptr<Encryptor> encryptor;
unique_ptr<Decryptor> decryptor;
unique_ptr<BatchEncoder> batch_encoder;

struct EvalResult_HE {
    mp_limb_t f_val;                
    vector<Ciphertext> g_enc;       
};

long long comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    long long res = 1;
    for (int i = 1; i <= k; ++i) res = res * (n - i + 1) / i;
    return res;
}

void generate_random_polynomial(int m, int d, long long target_Mf) {
    global_poly_local.clear();
    long long max_terms = comb(m + d - 1, d);
    if (target_Mf > max_terms) target_Mf = max_terms; 
    if (target_Mf <= 0) target_Mf = 1; 

    map<vector<int>, mp_limb_t> unique_terms;
    while(unique_terms.size() < (size_t)target_Mf) {
        vector<int> vars(d);
        for(int j = 0; j < d; ++j) vars[j] = fast_rng() % m;
        sort(vars.begin(), vars.end()); 
        if (unique_terms.find(vars) == unique_terms.end()) {
            unique_terms[vars] = fast_rand_mod_nonzero_local();
        }
    }

    for (auto const& [vars_key, coeff_val] : unique_terms) {
        PolyTerm_Local pt; pt.coeff = coeff_val;
        for (int i = 0; i < d; ++i) pt.vars[i] = vars_key[i];
        for (int i = d; i < 6; ++i) pt.vars[i] = 0;
        global_poly_local.push_back(pt);
    }
}

long long get_Mf(int m, int d, int sparsity) {
    long long max_terms = comb(m + d - 1, d);
    long long mf = (max_terms * sparsity) / 100;
    return mf == 0 ? 1 : mf;
}

int find_best_m(int d, long long target_Mf, int sparsity) {
    int best_m = 1;
    long long min_diff = -1;
   
    for (int m = 1; m <= 3000; ++m) { 
        long long mf = get_Mf(m, d, sparsity);
        long long diff = std::abs(mf - target_Mf);
        if (min_diff == -1 || diff < min_diff) {
            min_diff = diff;
            best_m = m;
        }
        if (mf > target_Mf) break; 
    }
    return best_m;
}

vector<mp_limb_t> precompute_dec_weights(int D_deg_local, int k, int ell) {
    nmod_mat_t M, invM;
    nmod_mat_init(M, D_deg_local + 1, D_deg_local + 1, PRIME_HE);
    nmod_mat_init(invM, D_deg_local + 1, D_deg_local + 1, PRIME_HE);

    int eq_count = 0;
    for (int j = 1; j <= k && eq_count < D_deg_local + 1; ++j) {
        mp_limb_t u = j;
        for (int h = 0; h <= ell && eq_count < D_deg_local + 1; ++h) {
            for (int col = h; col <= D_deg_local; ++col) {
                mp_limb_t coeff = 1;
                for (int c = 0; c < h; ++c) coeff = mul_fast(coeff, col - c);
                nmod_mat_entry(M, eq_count, col) = mul_fast(coeff, power_fast(u, col - h));
            }
            eq_count++;
        }
    }
    if (nmod_mat_inv(invM, M) == 0) exit(EXIT_FAILURE);
    vector<mp_limb_t> W(D_deg_local + 1);
    for (int i = 0; i < D_deg_local + 1; ++i) W[i] = nmod_mat_entry(invM, 0, i); 
    nmod_mat_clear(M); nmod_mat_clear(invM);
    return W;
}

Ciphertext generate_dummy_ciphertext() {
    vector<uint64_t> pod(batch_encoder->slot_count(), 1ULL);
    Plaintext pt;
    batch_encoder->encode(pod, pt);
    Ciphertext ct;
    encryptor->encrypt(pt, ct); 
    return ct;
}

void Simulate_Eval_Results(vector<vector<EvalResult_HE>>& outs, const Ciphertext& dummy_ct) {
    for (int j = 0; j < k_srv; ++j) {
        for (int r = 0; r <= d_deg; ++r) {
            outs[j][r].f_val = fast_rng() % PRIME_HE;
            if (current_ell >= 1) outs[j][r].g_enc[0] = dummy_ct;
            if (current_ell >= 2) outs[j][r].g_enc[1] = dummy_ct;
        }
    }
}

NOINLINE void Dec_HE_TimeTest(const vector<mp_limb_t>& vk_alpha, const vector<vector<EvalResult_HE>>& outs, const vector<mp_limb_t>& W) {
    vector<mp_limb_t> y_values(d_deg + 1);
    int D_deg_local = d_deg * t_thr; 
    Plaintext pt_res;
    vector<uint64_t> pod;

    for (int r = 0; r <= d_deg; ++r) {
        mp_limb_t y_r = 0;
        int eq_count = 0;
        for (int j = 1; j <= k_srv && eq_count < D_deg_local + 1; ++j) {
            const auto& out_r = outs[j-1][r];
            y_r = add_fast(y_r, mul_fast(W[eq_count++], out_r.f_val));
            for (int h = 1; h <= current_ell && eq_count < D_deg_local + 1; ++h) {
                decryptor->decrypt(out_r.g_enc[h-1], pt_res);
                batch_encoder->decode(pt_res, pod);
                y_r = add_fast(y_r, mul_fast(W[eq_count++], pod[0] % PRIME_HE));
            }
        }
        y_values[r] = y_r;
    }
    mp_limb_t z = 0;
    for (int r = 0; r <= d_deg; ++r) z = add_fast(z, mul_fast(vk_alpha[r], y_values[r]));
    volatile mp_limb_t dummy = z; 
}

void execute_benchmark(ofstream& csv_file, int target_m, int iterations, long long target_Mf, const vector<mp_limb_t>& W_weights, string run_type) {
    m_vars = target_m; 
    generate_random_polynomial(m_vars, d_deg, target_Mf);
    
    vector<mp_limb_t> x_secret(m_vars);
    for(int i = 0; i < m_vars; ++i) x_secret[i] = fast_rng() % ctx_local.n;

    vector<mp_limb_t> vk(d_deg + 1, 1ULL);
    vector<vector<EvalResult_HE>> outs(k_srv, vector<EvalResult_HE>(d_deg + 1));
    for (int j = 0; j < k_srv; ++j) {
        for(int r = 0; r <= d_deg; ++r) outs[j][r].g_enc.resize(current_ell);
    }

    Ciphertext dummy_ct = generate_dummy_ciphertext();
    double t_local = 0, t_dec = 0;

    for (int iter = 0; iter < iterations; ++iter) {
        auto tL0 = chrono::high_resolution_clock::now();
        EvalResult_Local dummy_res;
        eval_all_local(x_secret, 0, dummy_res);
        volatile mp_limb_t dummy = dummy_res.f_val;
        auto tL1 = chrono::high_resolution_clock::now();
        t_local += chrono::duration_cast<chrono::microseconds>(tL1 - tL0).count();

        Simulate_Eval_Results(outs, dummy_ct);

        auto tD0 = chrono::high_resolution_clock::now();
        Dec_HE_TimeTest(vk, outs, W_weights); 
        auto tD1 = chrono::high_resolution_clock::now();
        t_dec += chrono::duration_cast<chrono::microseconds>(tD1 - tD0).count();
    }

    double avg_local = t_local / iterations;
    double avg_dec = t_dec / iterations;

    cout << left << setw(9) << run_type << "| " 
         << setw(2) << d_deg << "| " 
         << setw(2) << t_thr << "| " 
         << setw(2) << current_ell << "| " 
         << setw(4) << m_vars << "| " 
         << setw(8) << target_Mf << "| "
         << setw(13) << fixed << setprecision(2) << avg_local << "| " 
         << setw(10) << fixed << setprecision(2) << avg_dec << endl;
         
    csv_file << target_Mf << "," << d_deg << "," << t_thr << "," << k_srv << "," << current_ell << "," << m_vars << "," << avg_local << "," << avg_dec << "\n";
}

int main() {
    nmod_init(&ctx_local, PRIME_LOCAL);

    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 64; 
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 50, 49 })); 
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20)); 
    
    seal_context = make_shared<SEALContext>(parms, true, sec_level_type::none);
    if (!seal_context->parameters_set()) {
        poly_modulus_degree = 1024;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
        parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
        seal_context = make_shared<SEALContext>(parms);
    }
    
    KeyGenerator keygen(*seal_context);
    PublicKey pk; keygen.create_public_key(pk); 
    
    encryptor = make_unique<Encryptor>(*seal_context, pk);
    decryptor = make_unique<Decryptor>(*seal_context, keygen.secret_key());
    batch_encoder = make_unique<BatchEncoder>(*seal_context);

    PRIME_HE = parms.plain_modulus().value();
    
    ofstream csv_uni("benchmark_Local_vs_Dec_UniformMf.csv");
    string header = "Mf,d_deg,t_thr,k_srv,ell_order,m_vars,Local_Direct_us,Dec_Time_us\n";
    csv_uni << header;

    cout << "=== HIGH DEGREE Local vs Dec Benchmark (Logarithmic Mf Focus) ===" << endl;
    cout << left << setw(9) << "Mode" << "| d | t | l | m   | Mf       | Local_Dir(us) | Dec(us)" << endl;
    cout << "----------------------------------------------------------------------------" << endl;

    for (int test_d : {2, 3}) {
        d_deg = test_d;
        for (int test_t : {1, 2, 3}) {
            t_thr = test_t; k_srv = d_deg * t_thr + 1;
            for (int test_ell : {1, 2}) {
                current_ell = test_ell;
                vector<mp_limb_t> W = precompute_dec_weights(d_deg * t_thr, k_srv, test_ell);
                
                int iterations = 5; 
                
              
                long long mMf = 1000;
                long long XMf = 200000;
                
               
                int num_points = 15; 
                
            
                for (int i = 0; i < num_points; ++i) {
                    double exponent = (double)i / (num_points - 1);
                    double power_term = std::pow((double)XMf / mMf, exponent);
                    long long target_Mf = static_cast<long long>(mMf * power_term);
                    
                    if (target_Mf <= 0) target_Mf = 1;
                    
                    int best_m = find_best_m(d_deg, target_Mf, 10);
                    execute_benchmark(csv_uni, best_m, iterations, target_Mf, W, "LogScaleMf");
                }
            }
        }
    }
    
    csv_uni.close();
    cout << "----------------------------------------------------------------------------" << endl;
    return 0;
}
