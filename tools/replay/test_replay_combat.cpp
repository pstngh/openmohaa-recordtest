/* Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later. No warranty; see COPYING.txt. */
#include "replay_combat.h"
#include "replay_track.h"
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

void Tests() {
    replay::CombatController controller;
    replay::LiveWeaponInput live;
    auto request = controller.Update(true, true, live);
    Check(!request.primary && !request.reload, "unarmed bot requested combat");
    live.active = true;
    live.ammoInClip = true;
    // Automatic fire remains held even between actual shots. Fire rate is engine-owned.
    for (int tick = 0; tick < 50; ++tick) {
        request = controller.Update(true, true, live);
        Check(request.primary && !request.reload, "automatic trigger interrupted");
    }
    Check(!controller.Update(true, false, live).primary, "invented firing intent");

    live.semiAutomatic = true;
    live.idle = live.canFire = true;
    controller.Reset();
    for (int tick = 0; tick < 8; ++tick) {
        request = controller.Update(true, true, live);
        Check(request.primary == (tick % 2 == 0), "semi-auto hold lacks release edges");
    }
    live.canFire = false;
    Check(!controller.Update(true, true, live).primary, "sniper bypassed live fire delay");
    live.canFire = true;
    live.idle = false;
    Check(!controller.Update(true, true, live).primary, "sniper interrupted its animation");
    live.idle = true;
    Check(controller.Update(true, true, live).primary, "sniper failed to resume when ready");

    live.active = false; // Pending draw/switch/reload; movement remains independent.
    request = controller.Update(true, true, live);
    Check(!request.primary && !request.reload, "pending weapon state was interrupted");
    live.active = true;
    Check(controller.Update(true, true, live).primary, "draw wait retained a stale trigger");

    live.ammoInClip = false;
    live.canReload = true;
    request = controller.Update(true, true, live);
    Check(!request.primary && request.reload, "empty live clip did not request normal reload");
    live.canReload = false;
    request = controller.Update(true, true, live);
    Check(!request.primary && !request.reload, "out-of-ammo weapon requested refill");
    live.ammoInClip = true;
    live.canReload = true; // Partial magazine can reload, but does not NEED a donor reload.
    request = controller.Update(true, false, live);
    Check(!request.primary && !request.reload, "partially loaded weapon reloaded without live need");

    live.ammoInClip = false;
    request = controller.Update(false, true, live);
    Check(!request.primary && !request.reload, "disabled/finished playback requested reload");
    live.ammoInClip = true;
    request = controller.Update(false, true, live);
    Check(!request.primary && !request.reload, "disabled/finished playback requested fire");
    controller.Reset(); // A real live weapon change cannot carry a latched trigger.
    Check(controller.Update(true, true, live).primary, "weapon reset retained old trigger");

    // A donor may switch team, gun, ammo, secondary mode or reload schedule.
    // None of those are inputs to this policy; movement stays byte-value identical.
    replay::Clip donor;
    donor.team = 4;
    donor.duration = 1000;
    donor.weapons = {{"MP40", 0, 32, 300}, {"Kar98 Sniper", 100, 5, 50}};
    donor.actions = {{0, replay::Action::Reload}, {100, replay::Action::SecondaryShot}};
    for (unsigned time = 0; time < 1000; time += 50) {
        replay::Frame frame;
        frame.time = time;
        frame.origin = {float(time), 25, 1};
        frame.velocity = {1000, 0, 0};
        frame.weapon = (time / 100) % 2;
        frame.buttons = 3; // Both donor attack buttons; only primary intent is admitted.
        donor.frames.push_back(frame);
    }
    for (bool sniper : {false, true}) {
        controller.Reset();
        live.semiAutomatic = sniper;
        for (const auto& frame : donor.frames) {
            const auto sampled = replay::Sample(donor, frame.time);
            request = controller.Update(true, (sampled.buttons & 1) != 0, live);
            Check(!request.reload, "donor event caused reload of loaded live weapon");
            Check(sampled.origin == frame.origin && sampled.velocity == frame.velocity,
                  "combat changed original movement");
            Check(sampled.weapon == frame.weapon, "combat rewrote recording metadata");
        }
    }
    std::cout << "Live-loadout combat tests passed (SMG, semi-auto, ammo, draw, reset, immutable samples).\n";
}
} // namespace

int main() {
    try { Tests(); }
    catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
