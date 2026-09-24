// buta-ppo/src/env/math_utils.hpp
#pragma once

#include <cmath>
#include <random>

namespace buta_ppo::env::math {

inline void euler_to_mat3(float pitch, float yaw, float roll, float out_mat[3][3]) {
    float sy = std::sin(yaw), cy = std::cos(yaw);
    float sp = std::sin(pitch), cp = std::cos(pitch);
    float sr = std::sin(roll), cr = std::cos(roll);

    // X-axis (forward).
    out_mat[0][0] = cy * cp;
    out_mat[0][1] = sy * cp;
    out_mat[0][2] = -sp;

    // Y-axis (right).
    out_mat[1][0] = cy * sp * sr - sy * cr;
    out_mat[1][1] = sy * sp * sr + cy * cr;
    out_mat[1][2] = cp * sr;

    // Z-axis (up).
    out_mat[2][0] = cy * sp * cr + sy * sr;
    out_mat[2][1] = sy * sp * cr - cy * sr;
    out_mat[2][2] = cp * cr;
}

inline void random_norm_vec(std::mt19937& rng, float out_vec[3]) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    while (true) {
        float x = dist(rng), y = dist(rng), z = dist(rng);
        float sq_len = x*x + y*y + z*z;
        if (sq_len >= 1e-6f && sq_len <= 1.0f) {
            float inv_len = 1.0f / std::sqrt(sq_len);
            out_vec[0] = x * inv_len;
            out_vec[1] = y * inv_len;
            out_vec[2] = z * inv_len;
            return;
        }
    }
}

}; // namespace buta_ppo::env::math