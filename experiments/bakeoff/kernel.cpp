// SPEC.md (lidar) in C++ (C ABI). Build: g++ -O2 -fPIC -shared -ffp-contract=off -fno-fast-math -pthread
#include <algorithm>
#include <thread>
#include <vector>

static inline bool inside(const double* v, double px, double py) {
    bool in = false;
    double p1x = v[0], p1y = v[1];
    for (int i = 0; i < 5; ++i) {
        double p2x = v[(i % 4) * 2], p2y = v[(i % 4) * 2 + 1];
        if (py > std::min(p1y, p2y) && py <= std::max(p1y, p2y) && px <= std::max(p1x, p2x)) {
            double xin = (p1y != p2y) ? (py - p1y) * (p2x - p1x) / (p2y - p1y + 1e-8) + p1x : p1x;
            if (p1x == p2x || px <= xin) in = !in;
        }
        p1x = p2x; p1y = p2y;
    }
    return in;
}

static void cast(const double* tiles, const double* centers, int T, const double* pos, const double* dirs, int p,
                 int R, double step, double maxd, double* out, int* cand) {
    double x = pos[2 * p], y = pos[2 * p + 1], lim = (maxd + 10.0) * (maxd + 10.0);
    int nc = 0;
    for (int t = 0; t < T; ++t) {
        double dx = centers[2 * t] - x, dy = centers[2 * t + 1] - y;
        if (dx * dx + dy * dy <= lim) cand[nc++] = t;
    }
    int K = (int)(maxd / step), last = 0;
    for (int r = 0; r < R; ++r) {
        double dx = dirs[(p * R + r) * 2], dy = dirs[(p * R + r) * 2 + 1], res = 1.0;
        for (int k = 1; k <= K; ++k) {
            double px = x + dx * (step * k), py = y + dy * (step * k);
            bool hit = nc > 0 && inside(tiles + cand[last] * 8, px, py);
            if (!hit) for (int c = 0; c < nc; ++c) if (inside(tiles + cand[c] * 8, px, py)) { hit = true; last = c; break; }
            if (!hit) { res = step * (k - 1) / maxd; break; }
        }
        out[p * R + r] = res;
    }
}

extern "C" void lidar(const double* tiles, const double* centers, int T, const double* pos, const double* dirs, int P,
                      int R, double step, double maxd, double* out, int threads) {
    auto work = [&](int lo, int hi) {
        std::vector<int> cand(T);
        for (int p = lo; p < hi; ++p) cast(tiles, centers, T, pos, dirs, p, R, step, maxd, out, cand.data());
    };
    if (threads <= 1) { work(0, P); return; }
    std::vector<std::thread> pool;
    for (int k = 0; k < threads; ++k) pool.emplace_back(work, (long)P * k / threads, (long)P * (k + 1) / threads);
    for (auto& th : pool) th.join();
}
