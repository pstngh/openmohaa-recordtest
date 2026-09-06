/*
 * Native-control mohdm6 imitation. Copyright (C) 2026 OpenMoHAA contributors.
 * SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt.
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace imitation {
constexpr int features = 54, hidden = 64, components = 4, output = 144;
constexpr std::uint32_t mapChecksum = 1974169620u, tickMsec = 20;
using Features = std::array<float, features>;
using Hidden = std::array<float, hidden>;
using Output = std::array<float, output>;
struct Observation {
    std::array<float,3> position{}, velocity{};
    float pitch = 0, yaw = 0, lean = 0, viewHeight = 82;
    bool ducked = false, prone = false, walking = false;
    float health = 100, clipAmmo = 0, reserveAmmo = 0;
    int weaponClass = 0;
    std::array<int,3> previousMove{};
    unsigned previousButtons = 0;
    std::array<float,2> previousViewDelta{}; // Realized change, normalized to 20ms.
    bool target = false;
    std::array<float,3> targetRelative{}, targetVelocity{};
    float targetDistance = 0;
};
Features Encode(const Observation& observation);
int WeaponClass(std::string name);
float AngleDelta(float from, float to);
struct Action {
    int forward = 0, right = 0, up = 0;
    unsigned buttons = 0;
    float pitchDelta = 0, yawDelta = 0, mixtureWeight = 0;
    int component = 0;
};
Action Decode(const Output& output);
class Policy {
public:
    // A bounded, non-executable binary format. A failed load preserves the old model.
    bool Load(const void *data, std::size_t size, std::string& error);
    bool Loaded() const { return !weights.empty(); }
    Output Step(const Features& features, Hidden& state) const;
    Action Choose(const Output& output, const std::array<int,4>& previous,
                  const std::array<float,5>& uniform) const;
private:
    std::vector<float> weights;
};
// Firing is gated separately from gaze: pre-aim never requires a visible target.
struct FireGate {
    bool visible = false, reacted = false, aligned = false, clear = false, muzzle = false;
    bool Allows() const { return visible && reacted && aligned && clear && muzzle; }
};
} // namespace imitation
