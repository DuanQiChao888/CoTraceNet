#include <algorithm>
#include <cstdint>
#include <cstdlib>

// A reflected boundary lets all 256 x 256 sites contribute equally to each channel.
static int reflect(int v) {
    if (v < 0) return -v;
    if (v > 255) return 510 - v;
    return v;
}

extern "C" void spatial_grid(const uint8_t* image, uint16_t* counts) {
    std::fill(counts, counts + 144 * 16, 0);
    const int dy[] = {0,1,1,1}, dx[] = {1,0,1,-1};
    int16_t residual[256 * 256];
    int family = 0;
    for (int step : {1,2}) {
        for (int dir = 0; dir < 4; ++dir, ++family) {
            const int sy = dy[dir] * step, sx = dx[dir] * step;
            for (int y = 0; y < 256; ++y) {
                const auto* hi = image + 256 * reflect(y + sy);
                const auto* lo = image + 256 * reflect(y - sy);
                const auto* mid = image + 256 * y;
                for (int x = 0; x < 256; ++x) {
                    residual[y*256+x] = hi[reflect(x+sx)] + lo[reflect(x-sx)] - 2*mid[x];
                }
            }
            auto* base = counts + family * 18 * 16;
            for (int y = 0; y < 256; ++y) {
                const int nexty = reflect(y+dy[dir]);
                for (int x = 0; x < 256; ++x) {
                    const int cell = (y/64)*4 + x/64;
                    const int r = residual[y*256+x], q = residual[nexty*256+reflect(x+dx[dir])];
                    const int magnitude = std::min(std::abs(r), 4);
                    const int a = std::clamp(r,-2,2) + 2, b = std::clamp(q,-2,2) + 2;
                    const int word = a*5+b;
                    const int sign_pooled = std::min(word, 24-word);
                    ++base[magnitude*16+cell];
                    ++base[(5+sign_pooled)*16+cell];
                }
            }
        }
    }
}
