/*
 * Live-loadout combat for recorded movement. Part of OpenMoHAA.
 * Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Distributed without warranty; see COPYING.txt.
 */
#pragma once

// Added in OPM: the policy cannot read or change a recording's weapon/team,
// movement, health, inventory or ammo. The engine adapter supplies live facts.
namespace replay {
struct LiveWeaponInput {
    bool active = false; // False while unarmed, switching, drawing or reloading.
    bool semiAutomatic = false;
    bool idle = false;
    bool canFire = false; // Live weapon readiness, including delay/movement limits.
    bool ammoInClip = false;
    bool canReload = false;
};

struct CombatRequest {
    bool primary = false;
    bool reload = false;
};

class CombatController {
public:
    void Reset() { primaryDown = false; }

    CombatRequest Update(bool enabled, bool primaryIntent, const LiveWeaponInput& live) {
        if (!enabled || !live.active) {
            Reset();
            return {};
        }
        if (!live.ammoInClip) {
            Reset();
            // Only request a normal reload when the LIVE weapon needs one.
            return {false, live.canReload};
        }
        bool attack = primaryIntent;
        if (live.semiAutomatic) {
            // A recorded SMG trigger hold must not latch a sniper trigger forever.
            // Send a release between presses; native fire/draw/reload timing wins.
            attack = attack && live.idle && live.canFire && !primaryDown;
        }
        primaryDown = attack;
        // Automatic trigger holds are passed through; the engine still gates shots.
        return {attack, false};
    }

private:
    bool primaryDown = false;
};
} // namespace replay
