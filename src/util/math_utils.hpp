// buta-ppo/src/util/math_utils.hpp
#pragma once

#include <cmath>
#include <random>

namespace buta_ppo::util::math {

/**
 * @brief Subtracts two vectors and stores the result in a provided buffer.
 * 
 * @param vec_a The first vector.
 * @param vec_b The Second vector.
 * @param result The result location to store the subtraction.
 */
inline void sub_vec3(const float* vec_a, const float* vec_b, float* result) {
    result[0] = vec_a[0] - vec_b[0];
    result[1] = vec_a[1] - vec_b[1];
    result[2] = vec_a[2] - vec_b[2];
}

/**
 * @brief Adds two vectors and stores the result in a provided buffer.
 * 
 * @param vec_a The first vector.
 * @param vec_b The Second vector.
 * @param result The result location to store the addition.
 */
inline void add_vec3(const float* vec_a, const float* vec_b, float* result) {
    result[0] = vec_a[0] + vec_b[0];
    result[1] = vec_a[1] + vec_b[1];
    result[2] = vec_a[2] + vec_b[2];
}

/**
 * @brief Multiplies a vector by a scalar and stores the result in a provided buffer.
 * 
 * @param scalar The scalar for the opteraion.
 * @param vec_a The vector to scale.
 * @param result The result location to store the multiplication.
 */
inline void scalar_mult_vec3(const float scalar, const float* vec_a, float* result) {
    result[0] = scalar * vec_a[0];
    result[1] = scalar * vec_a[1];
    result[2] = scalar * vec_a[2];
}

/*
 * Euler angles follow the game's convention (Unreal rotators), the same as RLBot's `Rotator`, RLGym's
 * `euler_to_rotation`, and RocketSim: angles in radians, and
 *   - positive pitch raises the nose,
 *   - positive yaw turns the nose toward the car's right, from +X toward +Y,
 *   - positive roll lowers the car's right side.
 * The matrix rows are the car's forward, right and up directions in world space, like `PhysState::rot_mat`.
 * This is the textbook right-handed Z-Y-X rotation with pitch and roll negated, since the game's world is left-handed.
 */

/**
 * @brief Converts euler angles in the game's convention (see above) to a 3x3 rotation matrix.
 *
 * @param pitch The pitch angle, in radians. Positive raises the nose.
 * @param yaw The yaw angle, in radians. Positive turns the nose from +X toward +Y.
 * @param roll The roll angle, in radians. Positive lowers the right side.
 * @param out_mat The matrix to store the resulting conversion into: rows forward, right, up.
 */
inline void euler_to_mat3(float pitch, float yaw, float roll, float out_mat[3][3]) {
    const float sy = std::sin(yaw), cy = std::cos(yaw);
    const float sp = std::sin(pitch), cp = std::cos(pitch);
    const float sr = std::sin(roll), cr = std::cos(roll);

    // X-axis (forward).
    out_mat[0][0] = cp * cy;
    out_mat[0][1] = cp * sy;
    out_mat[0][2] = sp;

    // Y-axis (right).
    out_mat[1][0] = sr * sp * cy - cr * sy;
    out_mat[1][1] = sr * sp * sy + cr * cy;
    out_mat[1][2] = -sr * cp;

    // Z-axis (up).
    out_mat[2][0] = -(cr * sp * cy + sr * sy);
    out_mat[2][1] = sr * cy - cr * sp * sy;
    out_mat[2][2] = cr * cp;
}

/**
 * @brief Converts a 3x3 rotation matrix to euler angles in the game's convention (see above), the inverse of
 * `euler_to_mat3`.
 *
 * Pitch is in [-pi/2, pi/2], yaw and roll in [-pi, pi]. With the nose pointing straight up or down, yaw and roll
 * turn about the same axis, so only their combination is defined. The angles returned still rebuild the same matrix.
 *
 * @param mat The rotation matrix, rows forward, right, up, like `PhysState::rot_mat`.
 * @param out_pitch The pitch angle, in radians.
 * @param out_yaw The yaw angle, in radians.
 * @param out_roll The roll angle, in radians.
 */
inline void mat3_to_euler(const float mat[3][3], float& out_pitch, float& out_yaw, float& out_roll) {
    const float* forward = mat[0];
    const float* right = mat[1];
    const float* up = mat[2];

    out_pitch = std::atan2(forward[2], std::hypot(forward[0], forward[1]));
    out_yaw = std::atan2(forward[1], forward[0]);

    // Roll is measured against the right direction the same pitch and yaw would have with no roll.
    const float level_right[2] = { -std::sin(out_yaw), std::cos(out_yaw) };
    const float right_dot = right[0] * level_right[0] + right[1] * level_right[1];
    const float up_dot = up[0] * level_right[0] + up[1] * level_right[1];
    out_roll = std::atan2(up_dot, right_dot);
}

/**
 * @brief Creates a uniformly distributed random normalization vector via rejection sampling.
 * 
 * @param rng The rng device to use.
 * @param out_vec The vector reference to store the result to.
 */
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

} // namespace buta_ppo::util::math