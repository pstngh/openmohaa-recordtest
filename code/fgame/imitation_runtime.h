/* Native imitation control contracts. Copyright (C) 2026 OpenMoHAA contributors.
 * SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt. */
#pragma once
#include "imitation_policy.h"
#include <algorithm>
#include <cstdint>

namespace imitation {
// Human telemetry records requested inputs, not whether a weapon actually fired.
// Keep those two channels separate when fire/scope/interaction guards intervene.
struct ControlFeedback {
    Action intent{};
    unsigned sentButtons = 0;
    void Reset() { intent = {}; sentButtons = 0; }
    void Record(const Action& requested, unsigned executedButtons) {
        intent = requested;
        sentButtons = executedButtons;
    }
};

inline std::array<int,4> PreviousCategories(const Observation& observation) {
    const auto& move = observation.previousMove;
    const unsigned buttons = observation.previousButtons;
    const int lean = (buttons & 16) && !(buttons & 32) ? 0
        : (buttons & 32) && !(buttons & 16) ? 2 : 1;
    return {(move[0]/127+1)*3+move[1]/127+1, move[2]/127+1, lean,
        ((buttons&4)?1:0) | ((buttons&1)?2:0) | ((buttons&2)?4:0) | int(buttons&8)};
}

// Do not advance the recurrent policy multiple times for an unchanged server
// simulation tick. A discontinuity starts a fresh observation history.
inline std::int64_t Elapsed(bool initialized, int previous, int current) {
    return initialized ? std::int64_t(current)-std::int64_t(previous) : tickMsec;
}
inline bool Discontinuous(std::int64_t elapsed) { return elapsed < 0 || elapsed > 100; }

inline std::array<float,3> CommandView(float pitch, float yaw, const Action& action, int elapsed) {
    const float scale = std::clamp(elapsed, 0, 100)/float(tickMsec);
    // Match PM_UpdateViewAngles's native signed-short pitch limit, not +/-89.
    constexpr float maximumPitch = 16000.f*360.f/65536.f;
    return {std::clamp(AngleDelta(0,pitch)+action.pitchDelta*scale,-maximumPitch,maximumPitch),
            AngleDelta(0,yaw+action.yawDelta*scale), 0.f};
}
} // namespace imitation
