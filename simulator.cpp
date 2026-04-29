#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <fstream>
#include <map>
#include <thread>
#include <atomic>
#include <sstream>
#include <cstdlib>
#include <random>
#include <limits>
#include <aff3ct.hpp>

using namespace aff3ct;

struct Chain {
    std::string name;
    int K;
    int N_enc;
    int N;

    bool is_concat = false;
    int depth = 1;

    std::vector<int> pattern;
    std::vector<int> puncture_indices;
    std::vector<int> depuncture_indices;
    std::vector<std::unique_ptr<uint8_t[]>> dummy_mem;
    std::unique_ptr<spu::module::Source<int>> src;
    std::shared_ptr<module::Encoder<int>> enc_ptr;
    module::Encoder<int>* enc;
    std::unique_ptr<module::Modem<int, float, float>> mdm;
    std::unique_ptr<module::Channel<float>> chn;
    std::shared_ptr<module::Decoder_SIHO<int, float>> dec_ptr;
    module::Decoder_SIHO<int, float>* dec;
    std::unique_ptr<module::Monitor_BFER<int>> mnt;
    std::shared_ptr<void> cdc;
    std::shared_ptr<void> cdc_inner;
    module::Encoder<int>* enc_inner = nullptr;
    module::Decoder_SIHO<int, float>* dec_inner = nullptr;
};

class SystemBuilder {
public:
    virtual ~SystemBuilder() = default;
    virtual std::unique_ptr<Chain> build(int seed = 0) = 0;
};

class Uncoded_System : public SystemBuilder {
public:
    std::unique_ptr<Chain> build(int seed = 0) override {
        auto c = std::make_unique<Chain>();
        c->name = "Uncoded";
        c->K = 10500;
        c->N_enc = 10500;
        c->N = 10500;

        factory::Source p_src;
        p_src.K = c->K;
        p_src.seed = seed;
        c->src.reset(p_src.build<int>());

        c->enc_ptr = std::make_shared<module::Encoder_NO<int>>(c->K);
        c->enc = c->enc_ptr.get();
        c->mdm.reset(new module::Modem_BPSK<int, float, float>(c->N));
        c->chn.reset(new module::Channel_AWGN_LLR<float>(c->N, tools::Gaussian_noise_generator_implem::STD, seed));
        c->dec_ptr = std::make_shared<module::Decoder_NO<int, float>>(c->K);
        c->dec = c->dec_ptr.get();
        c->mnt.reset(new module::Monitor_BFER<int>(c->K, 1000));

        return c;
    }
};

class Conv_System : public SystemBuilder {
    std::string rate_name;
public:
    Conv_System(std::string rate) : rate_name(rate) {}

    std::unique_ptr<Chain> build(int seed = 0) override {
        auto c = std::make_unique<Chain>();
        c->name = "Conv Rate " + rate_name;
        c->K = 10494;

        factory::Codec_RSC p_cdc;
        p_cdc.K = c->K;
        p_cdc.N_cw = c->K * 2 + 12;
        p_cdc.N = p_cdc.N_cw;
        if (p_cdc.enc.get() != nullptr) {
            p_cdc.enc->K = p_cdc.K;
            p_cdc.enc->N_cw = p_cdc.N_cw;
        }
        if (p_cdc.dec.get() != nullptr) {
            p_cdc.dec->K = p_cdc.K;
            p_cdc.dec->N_cw = p_cdc.N_cw;
        }
        if (p_cdc.pct.get() != nullptr) {
            p_cdc.pct->K = p_cdc.K;
            p_cdc.pct->N_cw = p_cdc.N_cw;
            p_cdc.pct->N = p_cdc.N;
        }

        auto p_enc = dynamic_cast<factory::Encoder_RSC*>(p_cdc.enc.get());
        auto p_dec = dynamic_cast<factory::Decoder_RSC*>(p_cdc.dec.get());
        if (p_enc) {
            p_enc->poly = {0133, 0171};
            p_enc->buffered = false;
        }
        if (p_dec) {
            p_dec->poly = {0133, 0171};
            p_dec->type = "VITERBI";
            p_dec->implem = "STD";
            p_dec->buffered = false;
        }

        auto cdc_ptr = p_cdc.build();
        if (!cdc_ptr) {
            throw std::runtime_error("AFF3CT failed to build Conv Codec! Check implementation parameters.");
        }
        c->cdc.reset(cdc_ptr);
        c->enc = &cdc_ptr->get_encoder();
        c->dec = &cdc_ptr->get_decoder_siho();
        c->N_enc = c->enc->get_N();

        if (rate_name == "1/2")      c->pattern.clear();
        else if (rate_name == "2/3") c->pattern = {1, 1, 0, 1};
        else if (rate_name == "3/4") c->pattern = {1, 1, 0, 1, 1, 0};
        else if (rate_name == "5/6") c->pattern = {1, 1, 0, 1, 1, 0, 0, 1, 1, 0};
        else if (rate_name == "7/8") c->pattern = {1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0};

        int sum = 0;
        if (!c->pattern.empty()) {
            for (int b : c->pattern) if (b) sum++;
            c->N = (c->N_enc / c->pattern.size()) * sum;
            int rem = c->N_enc % c->pattern.size();
            for (int i = 0; i < rem; i++) { if (c->pattern[i]) c->N++; }

            c->puncture_indices.reserve(c->N);
            c->depuncture_indices.reserve(c->N);
            int p_idx = 0;
            int p_size = c->pattern.size();
            for (int i = 0; i < c->N_enc; i++) {
                if (c->pattern[p_idx]) {
                    c->puncture_indices.push_back(i);
                    c->depuncture_indices.push_back(i);
                }
                if (++p_idx == p_size) p_idx = 0;
            }
        } else {
            c->N = c->N_enc;
        }

        factory::Source p_src;
        p_src.K = c->K;
        p_src.seed = seed;
        c->src.reset(p_src.build<int>());

        c->mdm.reset(new module::Modem_BPSK<int, float, float>(c->N));
        c->chn.reset(new module::Channel_AWGN_LLR<float>(c->N, tools::Gaussian_noise_generator_implem::STD, seed));
        c->mnt.reset(new module::Monitor_BFER<int>(c->K, 1000));

        return c;
    }
};

class RS_System : public SystemBuilder {
    int depth;
public:
    RS_System(int d) : depth(d) {}
    
    std::unique_ptr<Chain> build(int seed = 0) override {
        auto c = std::make_unique<Chain>();
        c->name = "RS Depth " + std::to_string(depth);

        int K_rs = 223 * 8; // 1784 bits
        int N_rs = 255 * 8; // 2040 bits

        c->K = K_rs * depth;
        c->N_enc = N_rs * depth;
        c->N = c->N_enc;

        factory::Codec_RS p_cdc;
        p_cdc.K = K_rs;
        p_cdc.N_cw = N_rs;
        p_cdc.N = N_rs;
        if (p_cdc.enc.get() != nullptr) {
            p_cdc.enc->K = 223;
            p_cdc.enc->N_cw = 255;
        }
        if (p_cdc.dec.get() != nullptr) {
            p_cdc.dec->K = 223;
            p_cdc.dec->N_cw = 255;
        }

        auto p_dec_rs = dynamic_cast<factory::Decoder_RS*>(p_cdc.dec.get());
        if (p_dec_rs) {
            p_dec_rs->m = 8;
            p_dec_rs->t = 16;
        }

        if (p_cdc.pct.get() != nullptr) {
            p_cdc.pct->K = p_cdc.K;
            p_cdc.pct->N_cw = p_cdc.N_cw;
            p_cdc.pct->N = p_cdc.N;
        }
        auto cdc_ptr = p_cdc.build();
        if (!cdc_ptr) {
            throw std::runtime_error("AFF3CT failed to build RS Codec! Check implementation parameters.");
        }
        c->cdc.reset(cdc_ptr);
        c->enc = &cdc_ptr->get_encoder();
        c->dec = &cdc_ptr->get_decoder_siho();
        
        factory::Source p_src;
        p_src.K = c->K;
        p_src.seed = seed;
        c->src.reset(p_src.build<int>());

        c->mdm.reset(new module::Modem_BPSK<int, float, float>(c->N));
        c->chn.reset(new module::Channel_AWGN_LLR<float>(c->N, tools::Gaussian_noise_generator_implem::STD, seed));
        c->mnt.reset(new module::Monitor_BFER<int>(c->K, 1000));

        return c;
    }
};

class Concat_System : public SystemBuilder {
    std::string rate_name;
    int depth;
public:
    Concat_System(std::string rate, int d) : rate_name(rate), depth(d) {}

    std::unique_ptr<Chain> build(int seed = 0) override {
        auto c = std::make_unique<Chain>();
        c->name = "Concat " + rate_name + " D" + std::to_string(depth);
        c->is_concat = true;
        c->depth = depth;

        int K_rs = 223 * 8; // 1784 bits
        int N_rs = 255 * 8; // 2040 bits
        c->K = K_rs * depth;

        factory::Codec_RS p_cdc_rs;
        p_cdc_rs.K = K_rs;
        p_cdc_rs.N_cw = N_rs;
        p_cdc_rs.N = N_rs;
        if (p_cdc_rs.enc.get() != nullptr) { p_cdc_rs.enc->K = 223; p_cdc_rs.enc->N_cw = 255; }
        if (p_cdc_rs.dec.get() != nullptr) { p_cdc_rs.dec->K = 223; p_cdc_rs.dec->N_cw = 255; }

        auto p_dec_rs = dynamic_cast<factory::Decoder_RS*>(p_cdc_rs.dec.get());
        if (p_dec_rs) { p_dec_rs->m = 8; p_dec_rs->t = 16; }

        auto cdc_rs_ptr = p_cdc_rs.build();
        if (!cdc_rs_ptr) throw std::runtime_error("Failed to build Outer RS Codec!");
        c->cdc.reset(cdc_rs_ptr);
        c->enc = &cdc_rs_ptr->get_encoder();
        c->dec = &cdc_rs_ptr->get_decoder_siho();

        int K_conv = N_rs * depth;

        factory::Codec_RSC p_cdc_conv;
        p_cdc_conv.K = K_conv;
        p_cdc_conv.N_cw = K_conv * 2 + 12;
        p_cdc_conv.N = p_cdc_conv.N_cw;
        if (p_cdc_conv.enc.get() != nullptr) { p_cdc_conv.enc->K = p_cdc_conv.K; p_cdc_conv.enc->N_cw = p_cdc_conv.N_cw; }
        if (p_cdc_conv.dec.get() != nullptr) { p_cdc_conv.dec->K = p_cdc_conv.K; p_cdc_conv.dec->N_cw = p_cdc_conv.N_cw; }

        auto p_enc_conv = dynamic_cast<factory::Encoder_RSC*>(p_cdc_conv.enc.get());
        if (p_enc_conv) { p_enc_conv->poly = {0133, 0171}; p_enc_conv->buffered = false; }
        auto p_dec_conv = dynamic_cast<factory::Decoder_RSC*>(p_cdc_conv.dec.get());
        if (p_dec_conv) {
            p_dec_conv->poly = {0133, 0171};
            p_dec_conv->type = "VITERBI";
            p_dec_conv->implem = "STD";
            p_dec_conv->buffered = false;
        }

        auto cdc_conv_ptr = p_cdc_conv.build();
        if (!cdc_conv_ptr) throw std::runtime_error("Failed to build Inner Conv Codec!");
        c->cdc_inner.reset(cdc_conv_ptr);
        c->enc_inner = &cdc_conv_ptr->get_encoder();
        c->dec_inner = &cdc_conv_ptr->get_decoder_siho();

        c->N_enc = c->enc_inner->get_N();

        if (rate_name == "1/2")      c->pattern.clear();
        else if (rate_name == "2/3") c->pattern = {1, 1, 0, 1};
        else if (rate_name == "3/4") c->pattern = {1, 1, 0, 1, 1, 0};
        else if (rate_name == "5/6") c->pattern = {1, 1, 0, 1, 1, 0, 0, 1, 1, 0};
        else if (rate_name == "7/8") c->pattern = {1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0};

        int sum = 0;
        if (!c->pattern.empty()) {
            for (int b : c->pattern) if (b) sum++;
            c->N = (c->N_enc / c->pattern.size()) * sum;
            int rem = c->N_enc % c->pattern.size(); // Handle tail bits
            for (int i = 0; i < rem; i++) { if (c->pattern[i]) c->N++; }

            c->puncture_indices.reserve(c->N);
            c->depuncture_indices.reserve(c->N);
            int p_idx = 0;
            int p_size = c->pattern.size();
            for (int i = 0; i < c->N_enc; i++) {
                if (c->pattern[p_idx]) {
                    c->puncture_indices.push_back(i);
                    c->depuncture_indices.push_back(i);
                }
                if (++p_idx == p_size) p_idx = 0;
            }
        } else {
            c->N = c->N_enc;
        }

        factory::Source p_src;
        p_src.K = c->K;
        p_src.seed = seed;
        c->src.reset(p_src.build<int>());

        c->mdm.reset(new module::Modem_BPSK<int, float, float>(c->N));
        c->chn.reset(new module::Channel_AWGN_LLR<float>(c->N, tools::Gaussian_noise_generator_implem::STD, seed));
        c->mnt.reset(new module::Monitor_BFER<int>(c->K, 1000));

        return c;
    }
};

double get_bpsk_capacity_esno(double esno_linear) {
    const double PI = std::acos(-1.0);
    const double E = std::exp(1.0);
    double sigma = std::sqrt(1.0 / (2.0 * esno_linear));
    double y_max = std::max(15.0, 10.0 * sigma);
    int steps = 10000;
    double dy = (2.0 * y_max) / steps;
    double h_y = 0.0;

    for (int i = 0; i < steps; ++i) {
        double y = -y_max + i * dy + dy / 2.0;
        double p1 = std::exp(-std::pow(y - 1.0, 2) / (2.0 * sigma * sigma)) / (sigma * std::sqrt(2.0 * PI));
        double p0 = std::exp(-std::pow(y + 1.0, 2) / (2.0 * sigma * sigma)) / (sigma * std::sqrt(2.0 * PI));
        double py = 0.5 * p1 + 0.5 * p0;
        if (py > 1e-30) {
            h_y -= py * std::log2(py) * dy;
        }
    }
    double h_yx = 0.5 * std::log2(2.0 * PI * E * sigma * sigma);
    return h_y - h_yx;
}

double calculate_exact_bpsk_shannon_limit(double rate) {
    if (rate >= 1.0 || rate <= 0.0) return std::numeric_limits<double>::infinity();

    double low_ebno = 1e-4;
    double high_ebno = 100.0;
    double mid_ebno = 0.0;
    for (int i = 0; i < 50; ++i) {
        mid_ebno = (low_ebno + high_ebno) / 2.0;
        double esno_linear = rate * mid_ebno;
        if (get_bpsk_capacity_esno(esno_linear) < rate) low_ebno = mid_ebno;
        else high_ebno = mid_ebno;
    }
    return 10.0 * std::log10(mid_ebno);
}

struct CPUStats {
    unsigned long long active_time = 0;
    unsigned long long total_time = 0;
};

CPUStats get_cpu_stats() {
    std::ifstream file("/proc/stat");
    std::string line;
    CPUStats stats;
    if (std::getline(file, line)) {
        if (line.compare(0, 4, "cpu ") == 0) {
            std::istringstream ss(line.substr(4));
            unsigned long long user = 0, nice = 0, system = 0, idle = 0, iowait = 0, irq = 0, softirq = 0, steal = 0, guest = 0, guest_nice = 0;
            ss >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;
            stats.total_time = user + nice + system + idle + iowait + irq + softirq + steal + guest + guest_nice;
            stats.active_time = stats.total_time - (idle + iowait);
        }
    }
    return stats;
}

void load_progress(std::vector<std::string>& system_names, std::map<std::string, std::vector<float>>& all_ber_data) {
    std::ifstream f("data/ber_simulation_results.csv");
    if (!f.is_open()) return;

    std::string line;
    if (!std::getline(f, line)) return;

    std::stringstream ss_header(line);
    std::string cell;
    std::getline(ss_header, cell, ',');

    while (std::getline(ss_header, cell, ',')) {
        cell.erase(std::remove(cell.begin(), cell.end(), '\r'), cell.end());
        cell.erase(std::remove(cell.begin(), cell.end(), '\n'), cell.end());
        if (!cell.empty()) system_names.push_back(cell);
    }

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::getline(ss, cell, ',');

        int sys_idx = 0;
        while (std::getline(ss, cell, ',')) {
            cell.erase(std::remove(cell.begin(), cell.end(), '\r'), cell.end());
            cell.erase(std::remove(cell.begin(), cell.end(), '\n'), cell.end());
            if (sys_idx < system_names.size() && !cell.empty()) {
                try {
                    all_ber_data[system_names[sys_idx]].push_back(std::stof(cell));
                } catch (...) {}
            }
            sys_idx++;
        }
    }
    std::cout << "Resuming existing simulation..." << std::endl;
}

void save_progress(const std::vector<float>& ebno_range, const std::vector<std::string>& system_names, const std::map<std::string, std::vector<float>>& all_ber_data, bool is_slave = false) {
    if (is_slave) return;

    std::ofstream f("data/ber_simulation_results.csv");
    f << "Eb/No (dB)";
    for (const auto& name : system_names) f << "," << name;
    f << "\n";
    for (size_t j = 0; j < ebno_range.size(); ++j) {
        f << std::fixed << std::setprecision(1) << ebno_range[j];
        for (const auto& name : system_names) {
            auto it = all_ber_data.find(name);
            if (it != all_ber_data.end() && j < it->second.size()) {
                f << "," << std::scientific << std::setprecision(6) << it->second[j];
            } else {
                f << ",";
            }
        }
        f << "\n";
    }
}

double get_ebno_for_ber(const std::vector<float>& ebno_range, const std::vector<float>& ber_list, double target_ber) {
    for (size_t i = 0; i < ber_list.size(); ++i) {
        if (ber_list[i] <= target_ber) {
            if (i == 0) return ebno_range[i];
            double x0 = ebno_range[i - 1]; double x1 = ebno_range[i];
            double y0 = ber_list[i - 1];   double y1 = ber_list[i];
            if (y0 <= 0.0 || y1 <= 0.0 || std::abs(y0 - y1) < 1e-15) return x1;
            double f = (std::log10(target_ber) - std::log10(y0)) / (std::log10(y1) - std::log10(y0));
            return x0 + f * (x1 - x0);
        }
    }
    return -100.0; // Target not found (curve hasn't dropped this deep yet)
}

void run_simulation(SystemBuilder& builder, const std::vector<float>& ebno_range, std::vector<std::string>& system_names, std::map<std::string, std::vector<float>>& all_ber_data, std::map<std::string, float>& system_rates, bool is_slave, int num_threads_override) {
    int num_threads = num_threads_override;
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }
    if (is_slave && num_threads_override > 0) {
        std::cout << "Slave overriding thread count to " << num_threads << std::endl;
    }

    // Default max frames is 10,000,000
    unsigned long long SIM_MAX_FRAMES = 10000000ULL;
    const char* env_max = std::getenv("SIM_MAX_FRAMES");
    if (env_max) {
        try {
            SIM_MAX_FRAMES = std::stoull(std::string(env_max));
        } catch (...) {
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist;

    std::vector<std::unique_ptr<Chain>> chains;
    for (int t = 0; t < num_threads; t++) {
        chains.push_back(builder.build(dist(gen)));
    }

    Chain& master = *chains[0];
    if (std::find(system_names.begin(), system_names.end(), master.name) == system_names.end()) {
        system_names.push_back(master.name);
    }

    float R = (float)master.K / (float)master.N;
    system_rates[master.name] = R;

    size_t start_idx = all_ber_data[master.name].size();
    if (start_idx >= ebno_range.size()) {
        std::cout << "\nSkipping " << master.name << " (already fully simulated)." << std::endl;
        return;
    }

    std::cout << "\nSimulating " << master.name << " (Rate " << R << ") using " << num_threads << " threads..." << std::endl;
    std::cout << "Eb/No (dB) | BER        | FER        | Frames simulated" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    auto allocate_sockets = [](Chain* c) {
        auto bind_m = [c](auto* m) {
            if (m) {
                for (auto& t : m->tasks) {
                    for (auto& s : t->sockets) {
                        c->dummy_mem.push_back(std::make_unique<uint8_t[]>(s->get_databytes()));
                        s->bind((void*)c->dummy_mem.back().get());
                    }
                }
            }
        };
        bind_m(c->src.get());
        bind_m(c->enc);
        bind_m(c->mdm.get());
        bind_m(c->chn.get());
        bind_m(c->dec);
        bind_m(c->mnt.get());
        if (c->is_concat) {
            bind_m(c->enc_inner);
            bind_m(c->dec_inner);
        }
    };
    for (int t = 0; t < num_threads; t++) {
        allocate_sockets(chains[t].get());
    }

    struct ThreadBuffers {
        std::vector<int>   U_K;
        std::vector<int>   X_N_enc;
        std::vector<int>   X_N;
        std::vector<float> X_N_mod;
        std::vector<float> Y_N;
        std::vector<float> LLR_N;
        std::vector<float> LLR_N_depct;
        std::vector<int>   V_K;

        std::vector<int>   X_N_rs;
        std::vector<int>   X_N_itl;
        std::vector<int>   V_K_itl;
        std::vector<int>   V_K_rs;
        std::vector<float> LLR_N_rs;

        std::vector<float> CP;
    };

    std::vector<ThreadBuffers> t_buffers(num_threads);
    for (int t = 0; t < num_threads; t++) {
        Chain& c = *chains[t];
        ThreadBuffers& b = t_buffers[t];

        b.U_K.resize(c.K);
        b.X_N_enc.resize(c.N_enc);
        b.X_N.resize(c.N);
        b.X_N_mod.resize(c.N);
        b.Y_N.resize(c.N);
        b.LLR_N.resize(c.N);
        b.LLR_N_depct.resize(c.N_enc);
        b.V_K.resize(c.K);

        if (c.is_concat) {
            int N_rs = c.enc->get_N();
            b.X_N_rs.resize(N_rs * c.depth); b.X_N_itl.resize(N_rs * c.depth);
            b.V_K_itl.resize(N_rs * c.depth); b.V_K_rs.resize(N_rs * c.depth); b.LLR_N_rs.resize(N_rs * c.depth);
        }
        b.CP.resize(c.chn->get_n_frames());
    }

    for (size_t ebno_idx = start_idx; ebno_idx < ebno_range.size(); ++ebno_idx) {
        auto point_start_time = std::chrono::steady_clock::now();

        float ebno = ebno_range[ebno_idx];
        float esno = ebno + 10.0f * std::log10(R);
        float sigma = std::sqrt(1.0f / (2.0f * std::pow(10.0f, esno / 10.0f)));

        CPUStats prev_cpu_stats = get_cpu_stats();

        std::atomic<unsigned long long> global_be(0);
        std::atomic<int> global_fe(0);
        std::atomic<int> global_fra(0);
        auto worker = [&](int t) {
            Chain& c = *chains[t];
            ThreadBuffers& b = t_buffers[t];
            int last_printed_fra = 0;

            unsigned long long local_be = 0;
            int local_fe = 0;
            int local_fra = 0;

            std::fill(b.CP.begin(), b.CP.end(), sigma);

                 // Loop until 1000 errors or SIM_MAX_FRAMES
                 while (global_fe.load(std::memory_order_relaxed) < 1000 &&
                     global_fra.load(std::memory_order_relaxed) < SIM_MAX_FRAMES) {
                c.src->generate(b.U_K.data());

                if (c.is_concat) {
                    int K_rs = c.enc->get_K();
                    int N_rs = c.enc->get_N();
                    for (int i = 0; i < c.depth; ++i) {
                        c.enc->encode(b.U_K.data() + i * K_rs, b.X_N_rs.data() + i * N_rs);
                    }
                    // Block interleave
                    const int* __restrict__ rs_out = b.X_N_rs.data();
                    int* __restrict__ itl_in = b.X_N_itl.data();
                    for (int d = 0; d < c.depth; d++) {
                        for (int b_idx = 0; b_idx < 255; b_idx++) {
                            std::copy_n(rs_out + (d * 255 + b_idx) * 8, 8, itl_in + (b_idx * c.depth + d) * 8);
                        }
                    }
                    // Encode inner
                    c.enc_inner->encode(b.X_N_itl.data(), b.X_N_enc.data());
                } else if (c.name == "Uncoded") {
                    std::copy(b.U_K.begin(), b.U_K.end(), b.X_N_enc.begin());
                } else {
                    int K_enc = c.enc->get_K();
                    int N_enc = c.enc->get_N();
                    int frames = c.K / K_enc;
                    for (int i = 0; i < frames; ++i) {
                        c.enc->encode(b.U_K.data() + i * K_enc, b.X_N_enc.data() + i * N_enc);
                    }
                }

                if (!c.puncture_indices.empty()) {
                    for (size_t i = 0; i < c.puncture_indices.size(); ++i) {
                        b.X_N[i] = b.X_N_enc[c.puncture_indices[i]];
                    }
                } else {
                    std::copy(b.X_N_enc.begin(), b.X_N_enc.end(), b.X_N.begin());
                }

                c.mdm->modulate(b.X_N.data(), b.X_N_mod.data());
                c.chn->add_noise(b.CP.data(), b.X_N_mod.data(), b.Y_N.data());
                c.mdm->demodulate(b.CP.data(), b.Y_N.data(), b.LLR_N.data());

                if (!c.depuncture_indices.empty()) {
                    std::fill(b.LLR_N_depct.begin(), b.LLR_N_depct.end(), 0.0f);
                    for (size_t i = 0; i < c.depuncture_indices.size(); ++i) {
                        b.LLR_N_depct[c.depuncture_indices[i]] = b.LLR_N[i];
                    }
                } else {
                    std::copy(b.LLR_N.begin(), b.LLR_N.end(), b.LLR_N_depct.begin());
                }

                if (c.is_concat) {
                    // Decode inner
                    c.dec_inner->decode_siho(b.LLR_N_depct.data(), b.V_K_itl.data());

                    // Block deinterleave
                    const int* __restrict__ v_itl = b.V_K_itl.data();
                    int* __restrict__ v_rs = b.V_K_rs.data();
                    for (int d = 0; d < c.depth; d++) {
                        for (int b_idx = 0; b_idx < 255; b_idx++) {
                            std::copy_n(v_itl + (b_idx * c.depth + d) * 8, 8, v_rs + (d * 255 + b_idx) * 8);
                        }
                    }
                    // Convert to pseudo-LLR
                    for(size_t i = 0; i < b.V_K_rs.size(); i++) {
                        b.LLR_N_rs[i] = 1.0f - 2.0f * (float)b.V_K_rs[i];
                    }

                    // Decode outer
                    int K_rs = c.dec->get_K();
                    int N_rs = c.dec->get_N();
                    for (int i = 0; i < c.depth; ++i) {
                        c.dec->decode_siho(b.LLR_N_rs.data() + i * N_rs, b.V_K.data() + i * K_rs);
                    }
                } else if (c.name == "Uncoded") {
                    for (int i = 0; i < c.K; i++) {
                        b.V_K[i] = (b.LLR_N_depct[i] < 0.0f) ? 1 : 0;
                    }
                } else {
                    int K_dec = c.dec->get_K();
                    int N_dec = c.dec->get_N();
                    int frames_dec = c.K / K_dec;
                    for (int i = 0; i < frames_dec; ++i) {
                        c.dec->decode_siho(b.LLR_N_depct.data() + i * N_dec, b.V_K.data() + i * K_dec);
                    }
                }

                // Manual error check to avoid shared module state
                int be = 0;
                const int* __restrict__ u_k = b.U_K.data();
                const int* __restrict__ v_k = b.V_K.data();
                for (int i = 0; i < c.K; i++) {
                    be += (u_k[i] != v_k[i] ? 1 : 0);
                }
                local_be += be;
                local_fe += (be > 0 ? 1 : 0);
                local_be += be;
                local_fra += 1;

                // Batch global counter updates
                if (local_fra >= 100 || local_fe >= 15) {
                    global_be.fetch_add(local_be, std::memory_order_relaxed);
                    global_fe.fetch_add(local_fe, std::memory_order_relaxed);
                    global_fra.fetch_add(local_fra, std::memory_order_relaxed);

                    local_be = 0;
                    local_fe = 0;
                    local_fra = 0;

                    // Only thread 0 prints to avoid overlapping
                    if (t == 0) {
                        int current_fra = global_fra.load(std::memory_order_relaxed);
                        if (current_fra - last_printed_fra >= 1000) {
                            last_printed_fra = current_fra;
                            auto now = std::chrono::steady_clock::now();
                            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - point_start_time).count();
                            double elapsed_exact = std::chrono::duration<double>(now - point_start_time).count();

                            CPUStats curr_cpu_stats = get_cpu_stats();
                            float cpu_util = 0.0f;
                            if (curr_cpu_stats.total_time > prev_cpu_stats.total_time) {
                                cpu_util = 100.0f * (curr_cpu_stats.active_time - prev_cpu_stats.active_time) / (float)(curr_cpu_stats.total_time - prev_cpu_stats.total_time);
                            }
                            prev_cpu_stats = curr_cpu_stats;

                            int fps = (elapsed_exact > 0.0) ? (int)(current_fra / elapsed_exact) : 0;

                            std::cout << "\r    ~> Time: "
                                      << std::setfill('0') << std::setw(2) << (elapsed / 60) << ":"
                                      << std::setfill('0') << std::setw(2) << (elapsed % 60) << std::setfill(' ')
                                      << " | Frames: " << current_fra
                                      << " / " << SIM_MAX_FRAMES << " | FE: " << global_fe.load(std::memory_order_relaxed) << " / 1000"
                                      << " | FPS: " << std::left << std::setw(7) << fps << std::right
                                      << " | CPU: " << std::fixed << std::setprecision(1) << std::setw(5) << cpu_util << "%"
                                      << "   " << std::flush;
                        }
                    }
                }
            }

            // Flush any remaining local frames when the loop terminates
            if (local_fra > 0) {
                global_be.fetch_add(local_be, std::memory_order_relaxed);
                global_fe.fetch_add(local_fe, std::memory_order_relaxed);
                global_fra.fetch_add(local_fra, std::memory_order_relaxed);
            }
        };

        std::vector<std::thread> threads;
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(worker, t);
        }
        for (auto& th : threads) {
            th.join();
        }

        // Clear the live progress line
        std::cout << "\r                                                                                                            \r";

        auto point_end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(point_end_time - point_start_time).count();

        float ber = (double)global_be.load() / ((double)global_fra * (double)master.K);
        float fer = (float)global_fe / (float)global_fra;

        all_ber_data[master.name].push_back(ber);
        save_progress(ebno_range, system_names, all_ber_data, is_slave);

        // Print final result for Eb/No
        std::cout << std::fixed << std::setprecision(1) << std::setw(6) << ebno << "     | "
                  << std::scientific << std::setprecision(4) << ber << " | "
                  << std::scientific << std::setprecision(4) << fer << " | "
                  << std::left << std::setw(16) << global_fra << " | "
                  << std::right << std::setfill('0') << std::setw(2) << (elapsed / 60) << ":"
                  << std::setfill('0') << std::setw(2) << (elapsed % 60) << std::setfill(' ') << std::endl;

        if (ber == 0.0f && global_fra >= SIM_MAX_FRAMES) {
             std::cout << "  Zero errors encountered. Stopping early." << std::endl;
             int points_skipped = ebno_range.size() - all_ber_data[master.name].size();
             for (int i = 0; i < points_skipped; i++) {
                 all_ber_data[master.name].push_back(1e-7f);
             }
             save_progress(ebno_range, system_names, all_ber_data, is_slave);
             break;
        }
    }
}

int main(int argc, char** argv) {
    std::string target_sys = "";
    float target_ebno = -100.0f;
    bool is_slave = false;
    int num_threads_arg = 0;

    if (argc >= 3) {
        target_sys = argv[1];
        target_ebno = std::stof(argv[2]);
        is_slave = true;
    }
    if (argc >= 5 && std::string(argv[3]) == "--threads") {
        try { num_threads_arg = std::stoi(argv[4]); }
        catch (...) { std::cerr << "Invalid value for --threads argument." << std::endl; }
    }

    std::cout << "--- Starting Multi-Algorithm BER Simulation (AFF3CT C++) ---" << std::endl;

    auto total_start_time = std::chrono::steady_clock::now();

    // 0 to 12 in 0.1 steps
    std::vector<float> ebno_range;
    for (int i = 0; i <= 120; i += 1) {
        ebno_range.push_back(i / 10.0f);
    }

    std::vector<std::string> system_names;
    std::map<std::string, std::vector<float>> all_ber_data;
    std::map<std::string, float> system_rates;

    if (!is_slave) {
        load_progress(system_names, all_ber_data);
    }

    std::vector<std::unique_ptr<SystemBuilder>> system_registry;
    system_registry.push_back(std::make_unique<Uncoded_System>());

    std::vector<std::string> rates = {"1/2", "2/3", "3/4", "5/6", "7/8"};
    for (const auto& r : rates) {
        system_registry.push_back(std::make_unique<Conv_System>(r));
    }

    std::vector<int> depths = {1, 2, 4, 5, 8, 16};
    for (int d : depths) {
        system_registry.push_back(std::make_unique<RS_System>(d));
    }

    for (const auto& r : rates) {
        for (int d : depths) {
            system_registry.push_back(std::make_unique<Concat_System>(r, d));
        }
    }

    for (auto& sys_builder : system_registry) {
        if (!target_sys.empty()) {
            auto temp_chain = sys_builder->build(0);
            if (temp_chain->name != target_sys) continue;
        }

        std::vector<float> run_range = ebno_range;
        if (target_ebno > -99.0f) run_range = {target_ebno};

        run_simulation(*sys_builder, run_range, system_names, all_ber_data, system_rates, is_slave, num_threads_arg);
    }

    auto total_end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_elapsed = total_end_time - total_start_time;

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(2) << total_elapsed.count() << "s\n" << std::endl;

    if (!target_sys.empty()) return 0;

    std::stringstream report;
    report << std::string(105, '=') << "\n";
    std::string title = "FEC PERFORMANCE SUMMARY & BPSK CAPACITY LIMITS";
    report << std::string((105 - title.size()) / 2, ' ') << title << "\n";
    report << std::string(105, '=') << "\n";
    report << std::left << std::setw(25) << "Algorithm" << " | "
           << std::setw(12) << "BER = 1e-3" << " | "
           << std::setw(12) << "BER = 1e-5" << " | "
           << std::setw(12) << "Error-Free" << " | "
           << std::setw(12) << "BPSK Limit" << " | "
           << std::setw(12) << "Gap (1e-5)" << "\n";
    report << std::string(105, '-') << "\n";

    auto fmt_db = [](double val) {
        if (val <= -100.0) return std::string("N/A");
        std::stringstream s; s << std::fixed << std::setprecision(2) << val << " dB"; return s.str();
    };

    for (const auto& name : system_names) {
        if (all_ber_data.find(name) == all_ber_data.end()) continue;
        const auto& ber_list = all_ber_data[name];

        double ebno_3    = get_ebno_for_ber(ebno_range, ber_list, 1e-3);
        double ebno_5    = get_ebno_for_ber(ebno_range, ber_list, 1e-5);
        double ebno_free = get_ebno_for_ber(ebno_range, ber_list, 1.01e-7);

        double limit = std::numeric_limits<double>::infinity();
        if (system_rates.find(name) != system_rates.end()) limit = calculate_exact_bpsk_shannon_limit(system_rates[name]);

        std::string s_gap = "N/A";
        if (ebno_5 > -100.0 && limit != std::numeric_limits<double>::infinity()) s_gap = fmt_db(ebno_5 - limit);

        report << std::left << std::setw(25) << name << " | "
               << std::setw(12) << (ebno_3 > -100.0 ? fmt_db(ebno_3) : "> 12.0 dB") << " | "
               << std::setw(12) << (ebno_5 > -100.0 ? fmt_db(ebno_5) : "> 12.0 dB") << " | "
               << std::setw(12) << fmt_db(ebno_free) << " | "
               << std::setw(12) << (limit != std::numeric_limits<double>::infinity() ? fmt_db(limit) : "N/A") << " | "
               << std::setw(12) << s_gap << "\n";
    }
    report << std::string(105, '=') << "\n\n";
    std::cout << "\n" << report.str();
    std::ofstream("data/ber_summary_report.txt") << report.str();

    return 0;
}