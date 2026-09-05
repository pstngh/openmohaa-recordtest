/*
 * Spawn-indexed replay data. Part of OpenMoHAA.
 * Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Distributed without warranty; see COPYING.txt.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <string>
#include <vector>

// Added in OPM: deterministic, engine-independent recording playback.
namespace replay {
using Vec3 = std::array<float, 3>;

struct Frame {
    std::uint32_t time = 0;
    Vec3 origin{}, velocity{}, angles{}, mins{}, maxs{}, eyeOffset{};
    std::uint32_t pmFlags = 0;
    int forward = 0, right = 0, up = 0;
    std::uint32_t buttons = 0, pose = 0, weapon = 0;
    float lean = 0; // Reconstructed by the importer, not losslessly recorded.
};

struct Weapon {
    std::string name;
    std::uint32_t firstSeen = 0;
    int clipAmmo = -1, reserveAmmo = -1;
};

struct Action {
    enum Kind : std::uint32_t { Reload = 1, PrimaryShot = 2, SecondaryShot = 3 };
    std::uint32_t time = 0, kind = 0;
};

struct Clip {
    std::string id;
    int team = 0;
    Vec3 spawn{};
    std::uint32_t duration = 0;
    std::vector<Weapon> weapons;
    std::vector<Frame> frames;
    std::vector<Action> actions;
};

struct Library {
    std::string map;
    std::uint32_t checksum = 0, gameType = 0, protocol = 0, sampleMsec = 0;
    std::vector<Clip> clips;
};

// Transactional load: out is unchanged on failure. Explicit little-endian wire format.
bool Load(const void *data, std::size_t size, Library& out, std::string& error);
bool Compatible(const Library& library, const std::string& map, std::uint32_t checksum,
                std::uint32_t gameType, std::uint32_t protocol);
Frame Sample(const Clip& clip, std::uint32_t time);
float DistanceSquared(const Vec3& a, const Vec3& b);
float AngleDelta(float from, float to);

// All events up to and including time are delivered once, including time zero.
std::vector<Action> DueActions(const Clip& clip, std::size_t& cursor, std::uint32_t time);

class SpawnSelector {
public:
    explicit SpawnSelector(std::uint32_t seed = 1) : generator(seed) {}
    void Reset(std::uint32_t seed);
    // SIZE_MAX means uncovered/ambiguous spawn or all remaining clips in use.
    // Teams are ignored only for FFA; positions are never translated or rotated.
    std::size_t Select(const Library& library, const Vec3& spawn, int team, float tolerance,
                       const std::set<std::size_t>& busy = {});
private:
    struct Bag {
        Vec3 spawn{};
        int team = 0;
        std::vector<std::size_t> members, order;
        std::size_t cursor = 0, last = static_cast<std::size_t>(-1);
    };
    std::vector<Bag> bags;
    std::mt19937 generator;
};
} // namespace replay
