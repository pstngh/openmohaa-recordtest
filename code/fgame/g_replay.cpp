/*
 * Spawn-indexed replay bots. Part of OpenMoHAA.
 * Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Distributed without warranty; see COPYING.txt.
 */
#include "g_replay.h"
#include "replay_track.h"
#include "replay_combat.h"
#include "g_local.h"
#include "g_main.h"
#include "g_phys.h"
#include "player.h"
#include "weapon.h"
#include "g_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>

namespace {
constexpr std::size_t noClip = static_cast<std::size_t>(-1);
cvar_t *enabled, *liveAim, *actionsEnabled, *debugReplay, *spawnTolerance;
cvar_t *turnSpeed, *reactionMsec, *aimError, *replaySeed;

replay::Vec3 Packed(const Vector& v) { return {v[0], v[1], v[2]}; }
Vector Unpacked(const replay::Vec3& v) { return Vector(v[0], v[1], v[2]); }

struct ReplayState {
    SafePtr<Player> owner;
    replay::Vec3 spawn{};
    replay::Frame frame;
    Vector aim;
    SafePtr<Sentient> enemy;
    int born = 0, lastTime = 0, noticed = 0;
    int savedMoveType = MOVETYPE_WALK, savedContents = CONTENTS_BODY;
    int savedFlags = 0;
    std::size_t clip = noClip;
    SafePtr<Weapon> combatWeapon;
    replay::CombatController combat;
    bool eligible = false, tried = false, locked = false, finished = false;
};

struct ReplayWorld {
    replay::Library library;
    replay::SpawnSelector selector;
    std::map<Player *, ReplayState> states;
    bool attemptedLoad = false, loaded = false;
};
// Explicit lifetime: BotManager tears this down before module/global destruction.
ReplayWorld *replayWorld = nullptr;

void InitCvars() {
    enabled = gi.Cvar_Get("g_replay_bots", "0", 0);
    liveAim = gi.Cvar_Get("g_replay_aim", "1", 0);
    actionsEnabled = gi.Cvar_Get("g_replay_actions", "1", 0);
    debugReplay = gi.Cvar_Get("g_replay_debug", "0", 0);
    spawnTolerance = gi.Cvar_Get("g_replay_spawn_tolerance", "8", 0);
    turnSpeed = gi.Cvar_Get("g_replay_turn_speed", "360", 0);
    reactionMsec = gi.Cvar_Get("g_replay_reaction_ms", "200", 0);
    aimError = gi.Cvar_Get("g_replay_aim_error", "1.5", 0);
    replaySeed = gi.Cvar_Get("g_replay_seed", "1", 0);
    gi.Cvar_CheckRange(spawnTolerance, 0, 32, qfalse);
    gi.Cvar_CheckRange(turnSpeed, 1, 1440, qfalse);
    gi.Cvar_CheckRange(reactionMsec, 0, 2000, qtrue);
    gi.Cvar_CheckRange(aimError, 0, 20, qfalse);
}

ReplayState *Find(const Player *player) {
    if (!replayWorld || !player) return nullptr;
    auto it = replayWorld->states.find(const_cast<Player *>(player));
    return it == replayWorld->states.end() || it->second.owner.Pointer() != player ? nullptr : &it->second;
}

bool Living(Player *p) {
    return p && p->client && !p->IsDead() && !p->IsSpectator() && p->GetTeam() >= TEAM_FREEFORALL;
}

replay::Frame Snapshot(Player *p) {
    replay::Frame f;
    f.origin = Packed(p->origin);
    f.mins = Packed(p->mins); f.maxs = Packed(p->maxs);
    f.angles = Packed(p->GetViewAngles());
    f.eyeOffset = {0, 0, static_cast<float>(p->viewheight)};
    f.pmFlags = p->client->ps.pm_flags;
    f.pose = p->client->ps.walking ? 1 : 0;
    return f;
}

void Hold(ReplayState& state, const char *reason) {
    state.clip = noClip;
    state.frame.velocity = {};
    state.frame.buttons = 0;
    state.frame.forward = state.frame.right = state.frame.up = 0;
    state.enemy = nullptr;
    gi.Printf("Replay bot %d: %s; holding position until next spawn or replay disable.\n",
              state.owner->entnum, reason);
}

bool LoadLibrary() {
    auto& data = *replayWorld;
    if (data.attemptedLoad) return data.loaded;
    data.attemptedLoad = true;
    const auto checksum = static_cast<std::uint32_t>(
        std::strtoll(gi.Cvar_Get("sv_mapChecksum", "0", 0)->string, nullptr, 10));
    const int gameType = g_gametype->integer;
    if (g_protocol != 8 || (gameType != GT_FFA && gameType != GT_TEAM)) {
        gi.Printf("Replay: prototype supports protocol 8 FFA/team deathmatch only.\n");
        return false;
    }
    const str filename = va("replays/%s.%u.%d.rpl", level.mapname.c_str(), checksum, gameType);
    const long size = gi.FS_ReadFile(filename, nullptr, qtrue);
    if (size <= 0 || size > 128 * 1024 * 1024) {
        gi.Printf("Replay: missing/oversized library %s. Import matching recordings first.\n", filename.c_str());
        return false;
    }
    void *bytes = nullptr;
    const long readSize = gi.FS_ReadFile(filename, &bytes, qtrue);
    std::string error;
    const bool valid = bytes && readSize == size && replay::Load(bytes, size, data.library, error);
    if (bytes) gi.FS_FreeFile(bytes);
    if (!valid || !replay::Compatible(data.library, level.mapname.c_str(), checksum, gameType, g_protocol)) {
        gi.Printf("Replay: rejected %s: %s\n", filename.c_str(), error.empty() ? "identity mismatch" : error.c_str());
        return false;
    }
    const int fps = gi.Cvar_Get("sv_fps", "20", 0)->integer;
    if (fps <= 0 || 1000 % fps || 1000 / fps != static_cast<int>(replayWorld->library.sampleMsec)) {
        gi.Printf("Replay: sv_fps must match recording sample interval %u ms; restart map after changing it.\n",
                  data.library.sampleMsec);
        return false;
    }
    data.selector.Reset(static_cast<std::uint32_t>(replaySeed->integer));
    gi.Printf("Replay: loaded %zu spawn-anchored lives from %s.\n", data.library.clips.size(), filename.c_str());
    data.loaded = true;
    return true;
}

// Combat adapts only the primary-fire intent to the live loadout. Recorded
// weapon names, ammunition, reloads and secondary actions never mutate it.
replay::CombatRequest Combat(Player *p, ReplayState& state) {
    Weapon *weapon = p->GetActiveWeapon(WEAPON_MAIN);
    if (state.combatWeapon.Pointer() != weapon) {
        state.combatWeapon = weapon;
        state.combat.Reset();
    }
    replay::LiveWeaponInput live;
    const bool playable = actionsEnabled->integer && state.clip != noClip && !state.finished;
    if (playable && weapon && !p->GetNewActiveWeapon()) {
        const auto weaponState = weapon->GetState();
        live.active = weaponState == WEAPON_READY || weaponState == WEAPON_FIRING;
        live.semiAutomatic = weapon->IsSemiAuto();
        const int animation = p->client->ps.iViewModelAnim;
        live.idle = weaponState == WEAPON_READY && (animation == VM_ANIM_IDLE
            || (animation >= VM_ANIM_IDLE_0 && animation <= VM_ANIM_IDLE_2));
        live.ammoInClip = weapon->HasAmmoInClip(FIRE_PRIMARY);
        live.canReload = weaponState == WEAPON_READY && weapon->CheckReload(FIRE_PRIMARY);
        // ReadyToFire enforces the equipped weapon's delay and movement limits.
        // Automatic weapons keep the trigger held; the native state machine gates shots.
        live.canFire = live.semiAutomatic && live.idle && weapon->ReadyToFire(FIRE_PRIMARY, qfalse);
    }
    return state.combat.Update(playable, (state.frame.buttons & BUTTON_ATTACKLEFT) != 0, live);
}

bool PathClear(Player *p, const replay::Vec3& from, const replay::Frame& frame) {
    // CSV precision may place a bound fractionally in a surface. This epsilon does not alter the stored position.
    Vector mins = Unpacked(frame.mins) + Vector(.125f, .125f, .125f);
    Vector maxs = Unpacked(frame.maxs) - Vector(.125f, .125f, .125f);
    const trace_t trace = G_Trace(Unpacked(from), mins, maxs, Unpacked(frame.origin), p,
        MASK_PLAYERSOLID & ~CONTENTS_BODY, false, "Replay path validation");
    return !trace.startsolid && !trace.allsolid && trace.fraction == 1.0f;
}

void Aim(Player *p, ReplayState& state, int elapsed, int delta) {
    Vector desired = Unpacked(state.frame.angles);
    if (!liveAim->integer || state.clip == noClip) {
        state.aim = desired;
        state.enemy = nullptr;
        return;
    }
    const Vector eye = Unpacked(state.frame.origin) + Unpacked(state.frame.eyeOffset);
    Vector forward;
    state.aim.AngleVectors(&forward);
    Sentient *target = nullptr;
    float best = 4096.0f * 4096.0f;
    for (int i = 1; i <= SentientList.NumObjects(); ++i) {
        Sentient *other = SentientList.ObjectAt(i);
        if (other == p || other->IsDead() || other->hidden() || (other->flags & FL_NOTARGET)
            || !other->IsSubclassOfPlayer()) continue;
        Player *enemy = static_cast<Player *>(other);
        if (enemy->IsSpectator() || (g_gametype->integer != GT_FFA && enemy->GetTeam() == p->GetTeam())) continue;
        Vector direction = other->centroid - eye;
        const float dist = direction.lengthSquared();
        if (dist < .01f || dist > best) continue;
        direction.normalize();
        if (DotProduct(direction, forward) < .5f) continue;
        const trace_t trace = G_Trace(eye, vec_zero, vec_zero, other->centroid, p, MASK_OPAQUE, false, "Replay visibility");
        if (trace.fraction != 1.0f && trace.entityNum != other->entnum) continue;
        best = dist;
        target = other;
    }
    if (state.enemy.Pointer() != target) {
        state.enemy = target;
        state.noticed = elapsed;
    }
    if (target && elapsed - state.noticed >= reactionMsec->integer) {
        desired = (target->centroid - eye).toAngles();
        const float phase = float(elapsed + p->entnum * 173);
        desired[0] += std::sin(phase * .007f) * aimError->value;
        desired[1] += std::sin(phase * .005f) * aimError->value;
    }
    const float limit = turnSpeed->value * std::clamp(delta, 0, 100) / 1000.0f;
    for (int i = 0; i < 2; ++i) state.aim[i] += std::clamp(replay::AngleDelta(state.aim[i], desired[i]), -limit, limit);
    state.aim[2] = 0;
    state.aim.EulerNormalize();
}
} // namespace

// Only this adapter is a Player friend; the recording parser has no access to engine objects.
struct ReplayAccess {
    static void Apply(Player *p, ReplayState& state) {
        const auto& f = state.frame;
        const Vector destination = Unpacked(f.origin);
        p->movetype = MOVETYPE_NONE; // Kinematic, not NOCLIP (NOCLIP would disable damage).
        p->setContents(CONTENTS_WEAPONCLIP); // Hittable by weapons, not a player movement obstruction.
        p->setSize(Unpacked(f.mins), Unpacked(f.maxs));
        p->setOrigin(destination);
        p->velocity = Unpacked(f.velocity);
        auto& ps = p->client->ps;
        destination.copyTo(ps.origin);
        p->velocity.copyTo(ps.velocity);
        constexpr int posture = PMF_DUCKED | PMF_VIEW_PRONE | PMF_VIEW_DUCK_RUN | PMF_VIEW_JUMP_START | PMF_RESPAWNED;
        ps.pm_flags = (ps.pm_flags & ~posture) | (f.pmFlags & posture) | PMF_NO_PREDICTION;
        ps.pm_type = PM_NORMAL;
        ps.walking = (f.pose & 1) != 0;
        ps.groundPlane = ps.walking;
        ps.groundEntityNum = ps.walking ? ENTITYNUM_WORLD : ENTITYNUM_NONE;
        p->groundentity = ps.walking ? world->edict : nullptr;
        p->falling = !ps.walking;
        ps.fLeanAngle = f.lean;
        ps.viewheight = p->viewheight = f.eyeOffset[2];
        p->m_iMovePosFlags = (f.pmFlags & PMF_VIEW_PRONE) ? MPF_POSITION_PRONE
            : ((f.pmFlags & PMF_DUCKED) ? MPF_POSITION_CROUCHING : MPF_POSITION_STANDING);
        if (!ps.walking) p->m_iMovePosFlags |= MPF_POSITION_OFFGROUND;
        p->m_iMovePosFlags |= (f.buttons & BUTTON_RUN) ? MPF_MOVEMENT_RUNNING : MPF_MOVEMENT_WALKING;
        if (p->velocity[2] < 0 && !ps.walking) p->m_iMovePosFlags |= MPF_MOVEMENT_FALLING;
        p->SetViewAngles(state.aim);
        state.aim.copyTo(ps.viewangles);
        p->m_vViewAng = state.aim;
        p->m_vViewPos = destination + Unpacked(f.eyeOffset);
        p->m_vViewPos.copyTo(ps.vEyePos);
        p->link();
        // Network prediction also must not body-block the human against this kinematic bot.
        // Keep the SERVER bbox/contents intact for live weapon traces and damage.
        p->edict->s.solid = 0;
    }
    static void Move(Player *p, ReplayState& state, const usercmd_t *cmd) {
        p->oldorigin = p->origin;
        p->m_fLastDeltaTime = std::clamp(cmd->serverTime - p->client->ps.commandTime, 0, 100) * .001f;
        p->client->ps.commandTime = cmd->serverTime;
        Apply(p, state);
        p->CheckMoveFlags();
    }
};

void G_ReplayInit() {
    G_ReplayShutdown();
    InitCvars();
    replayWorld = new ReplayWorld;
}

void G_ReplayForget(Player *p) {
    ReplayState *state = Find(p);
    if (!state) return;
    if (state->locked) {
        if (p->movetype == MOVETYPE_NONE) p->movetype = state->savedMoveType;
        if (p->edict->r.contents == CONTENTS_WEAPONCLIP) p->setContents(state->savedContents);
        if (!(state->savedFlags & PMF_NO_PREDICTION)) p->client->ps.pm_flags &= ~PMF_NO_PREDICTION;
        p->link();
    }
    replayWorld->states.erase(p);
}

void G_ReplayShutdown() {
    if (!replayWorld) return;
    while (!replayWorld->states.empty()) {
        auto it = replayWorld->states.begin();
        if (it->second.owner) G_ReplayForget(it->first);
        else replayWorld->states.erase(it);
    }
    delete replayWorld;
    replayWorld = nullptr;
}

void G_ReplaySpawned(Player *p) {
    if (!replayWorld || !p || !p->client) return;
    G_ReplayForget(p);
    ReplayState state;
    state.owner = p;
    state.born = level.inttime;
    state.spawn = Packed(p->origin);
    state.frame = Snapshot(p);
    state.aim = p->GetViewAngles();
    state.savedMoveType = p->movetype;
    state.savedContents = p->edict->r.contents;
    state.savedFlags = p->client->ps.pm_flags;
    state.eligible = enabled->integer != 0;
    replayWorld->states.emplace(p, state);
}

bool G_ReplayLocked(Player *p) {
    const ReplayState *state = Find(p);
    return state && state->locked && enabled && enabled->integer && !p->IsDead() && !p->IsSpectator();
}

void G_ReplayRestore(Player *p) {
    if (G_ReplayLocked(p)) ReplayAccess::Apply(p, *Find(p));
}

bool G_ReplayClientMove(Player *p, const usercmd_t *cmd) {
    if (!G_ReplayLocked(p)) return false;
    ReplayAccess::Move(p, *Find(p), cmd);
    return true;
}

bool G_ReplayBuildCommand(Player *p, usercmd_t *cmd, usereyes_t *eyes) {
    if (!replayWorld) return false;
    if (!enabled->integer || !Living(p) || level.intermissiontime) {
        G_ReplayForget(p);
        return false;
    }
    if (!Find(p)) {
        G_ReplaySpawned(p);
        // An explicit spawn callback is required; enabling halfway through a life must not teleport it.
        Find(p)->eligible = false;
    }
    ReplayState& state = *Find(p);
    const int elapsed = level.inttime - state.born;
    if (!state.tried) {
        state.tried = state.locked = true;
        if (!state.eligible || elapsed < 0 || elapsed > 100) {
            state.frame = Snapshot(p);
            Hold(state, "replay enabled mid-life; respawn to select a recording");
        } else if (!LoadLibrary()) {
            Hold(state, "no compatible recording library");
        } else {
            std::set<std::size_t> busy;
            for (const auto& entry : replayWorld->states) if (entry.second.owner && entry.second.clip != noClip) busy.insert(entry.second.clip);
            state.clip = replayWorld->selector.Select(replayWorld->library, state.spawn, spawnTolerance->value, busy);
            if (state.clip == noClip) Hold(state, "uncovered/ambiguous spawn or spawn pool currently exhausted");
            else {
                const auto& clip = replayWorld->library.clips[state.clip];
                state.aim = Unpacked(clip.frames.front().angles);
                p->CancelEventsOfType(EV_SetViewangles);
                // Leave the live spawn's pending weapon draw/loadout events intact.
                gi.Printf("Replay bot %d: clip %s at (%.3f %.3f %.3f), %u ms\n", p->entnum,
                    clip.id.c_str(), clip.spawn[0], clip.spawn[1], clip.spawn[2], clip.duration);
            }
        }
    }
    const auto previous = state.frame.origin;
    if (state.clip != noClip) {
        const auto& clip = replayWorld->library.clips[state.clip];
        if (elapsed < state.lastTime || elapsed-state.lastTime > static_cast<int>(replayWorld->library.sampleMsec*2)) {
            Hold(state, "playback clock discontinuity");
        } else if (elapsed >= static_cast<int>(clip.duration)) {
            state.finished = true;
            state.frame.forward = state.frame.right = state.frame.up = 0;
            state.frame.velocity = {};
            state.frame.buttons = 0;
            // Reset a living replay endpoint without old damage, fake kill credit or forced round respawns.
            if (dmManager.AllowRespawn() && (g_gametype->integer == GT_FFA || p->AllowTeamRespawn())) {
                G_ReplayForget(p);
                p->Respawn(nullptr);
                return G_ReplayBuildCommand(p, cmd, eyes);
            }
        } else if (p->HasVehicle() || p->GetTurret() || p->GetLadder() || p->m_bFrozen || level.playerfrozen) {
            Hold(state, "unsupported attachment or scripted freeze");
        } else {
            auto next = replay::Sample(clip, static_cast<std::uint32_t>(elapsed));
            // Validate each crossed sample, not a shortcut across several samples/corners.
            auto from = previous;
            bool clear = true;
            auto sample = std::upper_bound(clip.frames.begin(), clip.frames.end(), state.lastTime,
                [](int t, const replay::Frame& f) { return t < static_cast<int>(f.time); });
            for (; sample != clip.frames.end() && sample->time <= static_cast<std::uint32_t>(elapsed); ++sample) {
                if (!PathClear(p, from, *sample)) { clear = false; break; }
                from = sample->origin;
            }
            if (clear) clear = PathClear(p, from, next);
            if (!clear) Hold(state, "recorded path blocked by the current world");
            else {
                state.frame = next;
            }
        }
    }
    Aim(p, state, elapsed, elapsed-state.lastTime);
    state.lastTime = elapsed;
    ReplayAccess::Apply(p, state); // Before ClientThink computes its eyes, use interactions and weapon targeting.
    *cmd = {};
    *eyes = {};
    cmd->serverTime = level.svsTime;
    cmd->forwardmove = state.frame.forward;
    cmd->rightmove = state.frame.right;
    cmd->upmove = state.frame.up;
    // Preserve locomotion inputs, not donor-specific USE, scope, melee or fire modes.
    cmd->buttons = state.frame.buttons & ~(BUTTON_USE | BUTTON_ATTACKLEFT | BUTTON_ATTACKRIGHT);
    const auto combat = Combat(p, state);
    if (combat.reload) p->PlayerReload(nullptr);
    if (combat.primary) cmd->buttons |= BUTTON_ATTACKLEFT;
    for (int i = 0; i < 3; ++i) {
        cmd->angles[i] = ANGLE2SHORT(state.aim[i]) - p->client->ps.delta_angles[i];
        eyes->ofs[i] = static_cast<signed char>(std::clamp(state.frame.eyeOffset[i], -127.0f, 127.0f));
    }
    eyes->angles[0] = state.aim[0]; eyes->angles[1] = state.aim[1];
    return true;
}

void G_ReplayValidate(Player *p) {
    if (!G_ReplayLocked(p)) return;
    const ReplayState& state = *Find(p);
    if (state.clip == noClip) return;
    const float originError = std::sqrt(replay::DistanceSquared(Packed(p->origin), state.frame.origin));
    const float stateError = std::sqrt(replay::DistanceSquared(Packed(Vector(p->client->ps.origin)), state.frame.origin));
    const float networkError = std::sqrt(replay::DistanceSquared(Packed(Vector(p->edict->s.netorigin)), state.frame.origin));
    if (debugReplay->integer || originError > .001f || stateError > .001f || networkError > .001f) {
        gi.Printf("replay_sample bot=%d clip=%s t=%d origin=%.3f,%.3f,%.3f error=%.9f ps_error=%.9f net_error=%.9f\n",
            p->entnum, replayWorld->library.clips[state.clip].id.c_str(), state.lastTime,
            p->origin[0], p->origin[1], p->origin[2], originError, stateError, networkError);
    }
}
