#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <random>
#include <flint/flint.h>
#include <flint/nmod_mat.h>
#include <flint/nmod_vec.h>

using namespace std;

const mp_limb_t PRIME = 18446744073709551557ULL; 
nmod_t ctx; 

int m_vars = 5;       
int d_deg = 3;        
int t_thr = 1;        
int current_ell = 2;  
int k_srv = 4;        

std::mt19937_64 fast_rng(12345);

inline mp_limb_t fast_rand_mod() {
    return fast_rng() % ctx.n;
}

inline mp_limb_t fast_rand_mod_nonzero() {
    mp_limb_t r = fast_rng() % ctx.n;
    return r == 0 ? 1 : r;
}

mp_limb_t add(mp_limb_t a, mp_limb_t b) { return nmod_add(a, b, ctx); }
mp_limb_t mul(mp_limb_t a, mp_limb_t b) { return nmod_mul(a, b, ctx); }
mp_limb_t sub(mp_limb_t a, mp_limb_t b) { return nmod_sub(a, b, ctx); }
mp_limb_t inv(mp_limb_t a) { return nmod_inv(a, ctx); }
mp_limb_t power(mp_limb_t base, mp_limb_t exp) {
    mp_limb_t res = 1;
    base = base % ctx.n;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul(res, base);
        base = mul(base, base);
        exp /= 2;
    }
    return res;
}

long long comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n / 2) k = n - k;
    long long res = 1;
    for (int i = 1; i <= k; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

struct PolyTerm {
    mp_limb_t coeff;
    int vars[4];
};

vector<PolyTerm> global_poly;

void generate_random_polynomial(int m, int d, int sparsity_percent) {
    global_poly.clear();
    
    long long max_terms = comb(m + d - 1, d);
    long long target_terms = (max_terms * sparsity_percent) / 100;
    if (target_terms == 0) target_terms = 1; 

    map<vector<int>, mp_limb_t> unique_terms;
    
    while(unique_terms.size() < (size_t)target_terms) {
        vector<int> vars(d);
        for(int j = 0; j < d; ++j) vars[j] = fast_rng() % m;
        sort(vars.begin(), vars.end()); 
        if (unique_terms.find(vars) == unique_terms.end()) {
            unique_terms[vars] = fast_rand_mod_nonzero();
        }
    }

    for (auto const& [vars_key, coeff_val] : unique_terms) {
        PolyTerm pt;
        pt.coeff = coeff_val;
        for (int i = 0; i < d; ++i) pt.vars[i] = vars_key[i];
        for (int i = d; i < 4; ++i) pt.vars[i] = 0;
        global_poly.push_back(pt);
    }
}

struct ServerShare {
    vector<vector<mp_limb_t>> in_r;      
    vector<vector<vector<mp_limb_t>>> rec_r; 
};

struct EvalResult {
    mp_limb_t f_val;
    vector<mp_limb_t> grad;
    vector<vector<mp_limb_t>> hessian;
};

void eval_all(const vector<mp_limb_t>& x, int ell, EvalResult& out) {
    out.f_val = 0;
    if (ell >= 1) out.grad.assign(m_vars, 0);
    if (ell >= 2) {
        out.hessian.resize(m_vars);
        for(int i = 0; i < m_vars; ++i) out.hessian[i].assign(m_vars, 0);
    }

    for (const auto& term : global_poly) {
        mp_limb_t c = term.coeff;
        int v[4];
        mp_limb_t xv[4];
        for (int i = 0; i < d_deg; ++i) {
            v[i] = term.vars[i];
            xv[i] = x[v[i]];
        }

        if (d_deg == 2) {
            out.f_val = add(out.f_val, mul(c, mul(xv[0], xv[1])));
            if (ell >= 1) {
                out.grad[v[0]] = add(out.grad[v[0]], mul(c, xv[1]));
                out.grad[v[1]] = add(out.grad[v[1]], mul(c, xv[0]));
            }
            if (ell >= 2) {
                out.hessian[v[0]][v[1]] = add(out.hessian[v[0]][v[1]], c);
                out.hessian[v[1]][v[0]] = add(out.hessian[v[1]][v[0]], c);
            }
        } else if (d_deg == 3) {
            mp_limb_t p01 = mul(xv[0], xv[1]);
            mp_limb_t p02 = mul(xv[0], xv[2]);
            mp_limb_t p12 = mul(xv[1], xv[2]);
            out.f_val = add(out.f_val, mul(c, mul(p01, xv[2])));
            if (ell >= 1) {
                out.grad[v[0]] = add(out.grad[v[0]], mul(c, p12));
                out.grad[v[1]] = add(out.grad[v[1]], mul(c, p02));
                out.grad[v[2]] = add(out.grad[v[2]], mul(c, p01));
            }
            if (ell >= 2) {
                mp_limb_t cx0 = mul(c, xv[0]);
                mp_limb_t cx1 = mul(c, xv[1]);
                mp_limb_t cx2 = mul(c, xv[2]);
                out.hessian[v[0]][v[1]] = add(out.hessian[v[0]][v[1]], cx2);
                out.hessian[v[1]][v[0]] = add(out.hessian[v[1]][v[0]], cx2);
                out.hessian[v[0]][v[2]] = add(out.hessian[v[0]][v[2]], cx1);
                out.hessian[v[2]][v[0]] = add(out.hessian[v[2]][v[0]], cx1);
                out.hessian[v[1]][v[2]] = add(out.hessian[v[1]][v[2]], cx0);
                out.hessian[v[2]][v[1]] = add(out.hessian[v[2]][v[1]], cx0);
            }
        } else if (d_deg == 4) {
            mp_limb_t p01 = mul(xv[0], xv[1]);
            mp_limb_t p02 = mul(xv[0], xv[2]);
            mp_limb_t p03 = mul(xv[0], xv[3]);
            mp_limb_t p12 = mul(xv[1], xv[2]);
            mp_limb_t p13 = mul(xv[1], xv[3]);
            mp_limb_t p23 = mul(xv[2], xv[3]);
            out.f_val = add(out.f_val, mul(c, mul(p01, p23)));
            if (ell >= 1) {
                out.grad[v[0]] = add(out.grad[v[0]], mul(c, mul(p12, xv[3])));
                out.grad[v[1]] = add(out.grad[v[1]], mul(c, mul(p02, xv[3])));
                out.grad[v[2]] = add(out.grad[v[2]], mul(c, mul(p01, xv[3])));
                out.grad[v[3]] = add(out.grad[v[3]], mul(c, mul(p01, xv[2])));
            }
            if (ell >= 2) {
                mp_limb_t cp01 = mul(c, p01);
                mp_limb_t cp02 = mul(c, p02);
                mp_limb_t cp03 = mul(c, p03);
                mp_limb_t cp12 = mul(c, p12);
                mp_limb_t cp13 = mul(c, p13);
                mp_limb_t cp23 = mul(c, p23);
                
                out.hessian[v[0]][v[1]] = add(out.hessian[v[0]][v[1]], cp23);
                out.hessian[v[1]][v[0]] = add(out.hessian[v[1]][v[0]], cp23);
                out.hessian[v[0]][v[2]] = add(out.hessian[v[0]][v[2]], cp13);
                out.hessian[v[2]][v[0]] = add(out.hessian[v[2]][v[0]], cp13);
                out.hessian[v[0]][v[3]] = add(out.hessian[v[0]][v[3]], cp12);
                out.hessian[v[3]][v[0]] = add(out.hessian[v[3]][v[0]], cp12);
                out.hessian[v[1]][v[2]] = add(out.hessian[v[1]][v[2]], cp03);
                out.hessian[v[2]][v[1]] = add(out.hessian[v[2]][v[1]], cp03);
                out.hessian[v[1]][v[3]] = add(out.hessian[v[1]][v[3]], cp02);
                out.hessian[v[3]][v[1]] = add(out.hessian[v[3]][v[1]], cp02);
                out.hessian[v[2]][v[3]] = add(out.hessian[v[2]][v[3]], cp01);
                out.hessian[v[3]][v[2]] = add(out.hessian[v[3]][v[2]], cp01);
            }
        }
    }

    if (ell >= 2 && d_deg >= 2) {
        mp_limb_t inv_2 = inv(2);
        for (int i = 0; i < m_vars; ++i) {
            for (int j = i + 1; j < m_vars; ++j) {
                mp_limb_t avg = mul(add(out.hessian[i][j], out.hessian[j][i]), inv_2);
                out.hessian[i][j] = out.hessian[j][i] = avg;
            }
        }
    }
}

vector<mp_limb_t> precompute_dec_weights(int D_deg_local, int k, int ell) {
    nmod_mat_t M, invM;
    nmod_mat_init(M, D_deg_local + 1, D_deg_local + 1, ctx.n);
    nmod_mat_init(invM, D_deg_local + 1, D_deg_local + 1, ctx.n);

    int eq_count = 0;
    for (int j = 1; j <= k && eq_count < D_deg_local + 1; ++j) {
        mp_limb_t u = j;
        for (int h = 0; h <= ell && eq_count < D_deg_local + 1; ++h) {
            for (int col = h; col <= D_deg_local; ++col) {
                mp_limb_t coeff = 1;
                for (int c = 0; c < h; ++c) coeff = mul(coeff, col - c);
                nmod_mat_entry(M, eq_count, col) = mul(coeff, power(u, col - h));
            }
            eq_count++;
        }
    }

    int can_inv = nmod_mat_inv(invM, M);
    if (can_inv == 0) {
        exit(EXIT_FAILURE);
    }

    vector<mp_limb_t> W(D_deg_local + 1);
    for (int i = 0; i < D_deg_local + 1; ++i) {
        W[i] = nmod_mat_entry(invM, 0, i); 
    }

    nmod_mat_clear(M);
    nmod_mat_clear(invM);
    return W;
}

void Share(const vector<mp_limb_t>& x_input, vector<mp_limb_t>& vk_alpha, vector<ServerShare>& shares) {
    vector<mp_limb_t> v(d_deg + 1);
    v[0] = 1;
    for (int r = 1; r <= d_deg; ++r) {
        bool is_unique;
        do {
            is_unique = true;
            v[r] = fast_rand_mod_nonzero();
            if (v[r] <= 1) is_unique = false;
            for (int s = 1; s < r; ++s) if (v[r] == v[s]) is_unique = false;
        } while (!is_unique);
    }

    vk_alpha.resize(d_deg + 1);
    for (int r = 0; r <= d_deg; ++r) {
        mp_limb_t prod = 1;
        for (int s = 0; s <= d_deg; ++s) {
            if (s != r) {
                mp_limb_t num = sub(0, v[s]);
                mp_limb_t den = sub(v[r], v[s]);
                prod = mul(prod, mul(num, inv(den)));
            }
        }
        vk_alpha[r] = prod;
    }

    shares.resize(k_srv);
    for (int j = 0; j < k_srv; ++j) {
        shares[j].in_r.assign(d_deg + 1, vector<mp_limb_t>(m_vars, 0));
        shares[j].rec_r.assign(d_deg + 1, vector<vector<mp_limb_t>>(current_ell, vector<mp_limb_t>(m_vars, 0)));
    }

    for (int r = 0; r <= d_deg; ++r) {
        vector<vector<mp_limb_t>> a_coeffs(t_thr, vector<mp_limb_t>(m_vars));
        for (int h = 0; h < t_thr; ++h)
            for (int i = 0; i < m_vars; ++i)
                a_coeffs[h][i] = fast_rand_mod();

        for (int j = 1; j <= k_srv; ++j) {
            mp_limb_t u = j;
            vector<mp_limb_t>& phi_val = shares[j-1].in_r[r];
            for (int i = 0; i < m_vars; ++i) phi_val[i] = mul(v[r], x_input[i]);

            mp_limb_t u_h = u; 
            for (int h = 1; h <= t_thr; ++h) {
                for (int i = 0; i < m_vars; ++i)
                    phi_val[i] = add(phi_val[i], mul(a_coeffs[h-1][i], u_h));
                u_h = mul(u_h, u); 
            }

            if (current_ell >= 1) {
                vector<mp_limb_t>& phi_d1 = shares[j-1].rec_r[r][0]; 
                mp_limb_t u_h_minus_1 = 1; 
                for (int h = 1; h <= t_thr; ++h) {
                    mp_limb_t term = mul(h, u_h_minus_1);
                    for (int i = 0; i < m_vars; ++i) phi_d1[i] = add(phi_d1[i], mul(a_coeffs[h-1][i], term));
                    u_h_minus_1 = mul(u_h_minus_1, u); 
                }
            }
            if (current_ell >= 2) {
                vector<mp_limb_t>& phi_d2 = shares[j-1].rec_r[r][1];
                mp_limb_t u_h_minus_2 = 1;
                for (int h = 2; h <= t_thr; ++h) {
                    mp_limb_t term = mul(mul(h, h - 1), u_h_minus_2);
                    for (int i = 0; i < m_vars; ++i) phi_d2[i] = add(phi_d2[i], mul(a_coeffs[h-1][i], term));
                    u_h_minus_2 = mul(u_h_minus_2, u);
                }
            }
        }
    }
}

vector<EvalResult> Eval(const ServerShare& share) {
    vector<EvalResult> out(d_deg + 1);
    for (int r = 0; r <= d_deg; ++r) {
        eval_all(share.in_r[r], current_ell, out[r]);
    }
    return out;
}

bool Dec(const vector<mp_limb_t>& vk_alpha, const vector<ServerShare>& recs, 
         const vector<vector<EvalResult>>& outs, const vector<mp_limb_t>& W, 
         mp_limb_t& final_result) {
    vector<mp_limb_t> y_values(d_deg + 1);
    int D_deg_local = d_deg * t_thr; 

    for (int r = 0; r <= d_deg; ++r) {
        mp_limb_t y_r = 0;
        int eq_count = 0;

        for (int j = 1; j <= k_srv && eq_count < D_deg_local + 1; ++j) {
            const auto& out_r = outs[j-1][r];
            const auto& rec_r = recs[j-1].rec_r[r];

            y_r = add(y_r, mul(W[eq_count++], out_r.f_val));

            if (current_ell >= 1 && eq_count < D_deg_local + 1) { 
                mp_limb_t g1 = 0;
                for (int i = 0; i < m_vars; ++i) g1 = add(g1, mul(out_r.grad[i], rec_r[0][i]));
                y_r = add(y_r, mul(W[eq_count++], g1));
            }

            if (current_ell >= 2 && eq_count < D_deg_local + 1) { 
                mp_limb_t g2 = 0;
                for (int i = 0; i < m_vars; ++i) {
                    mp_limb_t row_sum = 0;
                    for (int l = 0; l < m_vars; ++l) row_sum = add(row_sum, mul(out_r.hessian[i][l], rec_r[0][l]));
                    g2 = add(g2, mul(rec_r[0][i], row_sum));
                }
                for (int i = 0; i < m_vars; ++i) g2 = add(g2, mul(out_r.grad[i], rec_r[1][i]));
                y_r = add(y_r, mul(W[eq_count++], g2));
            }
        }
        y_values[r] = y_r;
    }

    mp_limb_t z = 0;
    for (int r = 0; r <= d_deg; ++r) z = add(z, mul(vk_alpha[r], y_values[r]));

    if (z == 0) {
        final_result = y_values[0];
        return true;
    }
    return false;
}

int calc_N(int m, int ell) {
    if (ell == 0) return 1;
    if (ell == 1) return 1 + m;
    if (ell == 2) return 1 + m + (m * (m + 1)) / 2; 
    return 1;
}

int main() {
    nmod_init(&ctx, PRIME);

    ofstream csv_file("benchmark_4D_sweep_optimized.csv");
    csv_file << "sparsity_pct,d_deg,k_srv,ell_order,m_vars,Local_Direct_us,Share_Time_us,Eval_Time_Per_Server_us,Dec_Time_us,"
             << "Total_Comm_KB,Math_Correct\n";
    
    cout << "=== Fast 4D Academic Grade Benchmark ===" << endl;
    cout << left << setw(8) << "Sparsity" << "| " 
         << setw(3) << "d" << "| " 
         << setw(3) << "ell" << "| " 
         << setw(3) << "m" << "| " 
         << setw(13) << "Local_Dir(us)" << "| "
         << setw(15) << "Share(us)" << "| " 
         << setw(15) << "Eval/Srv(us)" << "| " 
         << setw(10) << "Dec(us)" << "| " 
         << "Correct?" << endl;
    cout << "--------------------------------------------------------------------------------------------------" << endl;

    int sparsity_levels[] = {10, 100};

    for (int test_sparsity : sparsity_levels) {
        for (int test_d = 2; test_d <= 4; ++test_d) {
            d_deg = test_d;
            k_srv = d_deg * t_thr + 1; 
            
            if (t_thr > k_srv - 1) {
                exit(EXIT_FAILURE);
            }

            for (int test_ell = 0; test_ell <= 2; ++test_ell) {
                current_ell = test_ell;
                
                int D_deg_local = d_deg * t_thr;
                vector<mp_limb_t> W_weights = precompute_dec_weights(D_deg_local, k_srv, current_ell);
                
                for (int test_m = 10; test_m <= 100; test_m += 10) {
                    m_vars = test_m; 
                    
                    int iterations = (d_deg == 4) ? 5 : (d_deg == 3 ? 20 : 50); 
                    generate_random_polynomial(m_vars, d_deg, test_sparsity);

                    double bytes_per_element = 8.0; 
                    long long upload_elements = k_srv * (d_deg + 1) * (m_vars + current_ell * m_vars);
                    long long N_size = calc_N(m_vars, current_ell);
                    long long download_elements = k_srv * (d_deg + 1) * N_size;
                    double total_comm_kb = ((upload_elements + download_elements) * bytes_per_element) / 1024.0;

                    vector<mp_limb_t> x_secret(m_vars);
                    for(int i = 0; i < m_vars; ++i) x_secret[i] = fast_rand_mod();
                    
                    EvalResult temp_expected;
                    eval_all(x_secret, 0, temp_expected);
                    mp_limb_t expected_result = temp_expected.f_val;

                    double total_local = 0, total_share = 0, total_eval = 0, total_dec = 0;
                    bool all_correct = true;

                    for (int iter = 0; iter < iterations; ++iter) {
                        auto tL0 = chrono::high_resolution_clock::now();
                        EvalResult dummy_res;
                        eval_all(x_secret, 0, dummy_res);
                        volatile mp_limb_t dummy = dummy_res.f_val;
                        auto tL1 = chrono::high_resolution_clock::now();
                        total_local += chrono::duration_cast<chrono::microseconds>(tL1 - tL0).count();

                        vector<mp_limb_t> vk;
                        vector<ServerShare> shares;
                        vector<vector<EvalResult>> outs(k_srv);
                        mp_limb_t result;

                        auto t0 = chrono::high_resolution_clock::now();
                        Share(x_secret, vk, shares);
                        auto t1 = chrono::high_resolution_clock::now();
                        total_share += chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

                        auto t2 = chrono::high_resolution_clock::now();
                        for (int j = 0; j < k_srv; ++j) outs[j] = Eval(shares[j]); 
                        auto t3 = chrono::high_resolution_clock::now();
                        total_eval += chrono::duration_cast<chrono::microseconds>(t3 - t2).count();

                        auto t4 = chrono::high_resolution_clock::now();
                        bool is_valid = Dec(vk, shares, outs, W_weights, result); 
                        auto t5 = chrono::high_resolution_clock::now();
                        total_dec += chrono::duration_cast<chrono::microseconds>(t5 - t4).count();

                        if (!is_valid || result != expected_result) all_correct = false;
                    }

                    double avg_local = total_local / iterations;
                    double avg_share = total_share / iterations;
                    double avg_eval = (total_eval / iterations) / k_srv;
                    double avg_dec = total_dec / iterations;

                    cout << left << setw(7) << test_sparsity << "% | " 
                         << setw(3) << d_deg << "| " 
                         << setw(3) << current_ell << "| " 
                         << setw(3) << m_vars << "| " 
                         << setw(13) << fixed << setprecision(2) << avg_local << "| " 
                         << setw(15) << fixed << setprecision(2) << avg_share << "| " 
                         << setw(15) << fixed << setprecision(2) << avg_eval << "| " 
                         << setw(10) << fixed << setprecision(2) << avg_dec << "| " 
                         << (all_correct ? "True" : "False") << endl;
                         
                    csv_file << test_sparsity << "," << d_deg << "," << k_srv << "," << current_ell << "," << m_vars << "," 
                             << avg_local << "," << avg_share << "," << avg_eval << "," << avg_dec << "," 
                             << total_comm_kb << "," << (all_correct ? "True" : "False") << "\n";
                }
            }
        }
    }

    csv_file.close();
    cout << "--------------------------------------------------------------------------------------------------" << endl;
    return 0;
}
