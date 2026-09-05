/*
 * Spawn-indexed replay data. Part of OpenMoHAA.
 * Copyright (C) 2026 OpenMoHAA contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Distributed without warranty; see COPYING.txt.
 */
#include "replay_track.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace replay {
namespace {
constexpr std::size_t noClip = static_cast<std::size_t>(-1);
constexpr std::size_t maxFile = 128 * 1024 * 1024;
constexpr std::uint32_t maxTime = 3600000;

class Reader {
public:
    Reader(const void *data, std::size_t length)
        : data(static_cast<const unsigned char *>(data)), length(length) {}
    std::uint32_t U32() {
        Need(4);
        const auto *p = data + offset;
        offset += 4;
        return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8)
             | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    }
    int I32() {
        const auto n = U32();
        return n <= 0x7fffffffU ? static_cast<int>(n) : -1 - static_cast<int>(0xffffffffU - n);
    }
    float Float(float limit) {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                      "Replay format requires IEEE-754 binary32");
        const auto bits = U32();
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value) || std::fabs(value) > limit) {
            throw std::runtime_error("non-finite or out-of-range float");
        }
        return value;
    }
    Vec3 Vector(float limit) { return {Float(limit), Float(limit), Float(limit)}; }
    std::string Text(std::size_t maximum) {
        auto count = U32();
        if (count > maximum) throw std::runtime_error("string too long");
        Need(count);
        std::string value(reinterpret_cast<const char *>(data + offset), count);
        offset += count;
        for (unsigned char c : value) {
            if (c < 32 || c > 126) throw std::runtime_error("invalid string character");
        }
        return value;
    }
    std::uint32_t Count(std::uint32_t maximum, std::size_t minimumBytes) {
        auto count = U32();
        if (count > maximum || count > (length - offset) / minimumBytes) {
            throw std::runtime_error("invalid count or truncated recording");
        }
        return count;
    }
    void Need(std::size_t count) const {
        if (count > length - offset) throw std::runtime_error("truncated recording");
    }
    bool Done() const { return offset == length; }
    std::size_t offset = 0;
private:
    const unsigned char *data;
    std::size_t length;
};

bool SafeMap(const std::string& map) {
    if (map.empty() || map.front() == '/' || map.find("..") != std::string::npos) return false;
    for (unsigned char c : map) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
            || c == '_' || c == '-' || c == '/')) return false;
    }
    return true;
}
} // namespace

bool Load(const void *data, std::size_t size, Library& out, std::string& error) {
    error.clear();
    try {
        if (!data || size < 8 || size > maxFile || std::memcmp(data, "OMRPL001", 8) != 0) {
            throw std::runtime_error("invalid replay magic, version or file size");
        }
        Reader in(data, size);
        in.offset = 8;
        Library library;
        library.map = in.Text(128);
        library.checksum = in.U32();
        library.gameType = in.U32();
        library.protocol = in.U32();
        library.sampleMsec = in.U32();
        if (!SafeMap(library.map) || library.gameType < 1 || library.gameType > 6
            || library.protocol != 8 || library.sampleMsec < 1 || library.sampleMsec > 1000) {
            throw std::runtime_error("unsupported map, game type, protocol or sample interval");
        }
        const auto count = in.Count(50000, 36);
        if (!count) throw std::runtime_error("empty replay library");
        std::set<std::string> ids;
        std::size_t totalFrames = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            Clip clip;
            clip.id = in.Text(64);
            if (clip.id.empty() || !ids.insert(clip.id).second) throw std::runtime_error("duplicate/empty clip ID");
            clip.team = in.I32();
            clip.spawn = in.Vector(131072);
            clip.duration = in.U32();
            if (clip.team < 0 || clip.team > 4 || !clip.duration || clip.duration > maxTime) {
                throw std::runtime_error("invalid team or duration");
            }
            const auto weapons = in.Count(64, 16);
            if (!weapons) throw std::runtime_error("missing weapon table");
            for (std::uint32_t j = 0; j < weapons; ++j) {
                Weapon weapon;
                weapon.name = in.Text(64);
                weapon.firstSeen = in.U32();
                weapon.clipAmmo = in.I32();
                weapon.reserveAmmo = in.I32();
                if (weapon.firstSeen >= clip.duration || weapon.clipAmmo < -1 || weapon.clipAmmo > 10000
                    || weapon.reserveAmmo < -1 || weapon.reserveAmmo > 100000) {
                    throw std::runtime_error("invalid weapon state");
                }
                clip.weapons.push_back(std::move(weapon));
            }
            const auto frames = in.Count(1000000, 108);
            const auto actions = in.Count(1000000, 8);
            totalFrames += frames;
            if (frames < 2 || totalFrames > 1000000) throw std::runtime_error("invalid total frame count");
            // The count checks above are lower bounds, not permission to read unchecked bytes.
            for (std::uint32_t j = 0; j < frames; ++j) {
                Frame f;
                f.time = in.U32();
                f.origin = in.Vector(131072);
                f.velocity = in.Vector(10000);
                f.angles = in.Vector(3600);
                f.mins = in.Vector(256);
                f.maxs = in.Vector(256);
                f.eyeOffset = in.Vector(256);
                f.pmFlags = in.U32();
                f.forward = in.I32(); f.right = in.I32(); f.up = in.I32();
                f.buttons = in.U32(); f.pose = in.U32(); f.weapon = in.U32();
                f.lean = in.Float(45);
                if ((j == 0 && f.time != 0) || f.time >= clip.duration
                    || (j && (f.time <= clip.frames.back().time
                              || f.time - clip.frames.back().time > library.sampleMsec * 2))
                    || f.buttons > 63 || f.pose > 7 || f.weapon >= weapons
                    || f.forward < -127 || f.forward > 127 || f.right < -127 || f.right > 127
                    || f.up < -127 || f.up > 127) {
                    throw std::runtime_error("invalid frame time, gap, command or weapon index");
                }
                for (int axis = 0; axis < 3; ++axis) {
                    if (f.maxs[axis] <= f.mins[axis]) throw std::runtime_error("invalid bounds");
                }
                if (f.pose & 2) throw std::runtime_error("ladder playback is not supported in format 1");
                clip.frames.push_back(f);
            }
            if (DistanceSquared(clip.frames.front().origin, clip.spawn) > 0.01f
                || clip.duration - clip.frames.back().time > library.sampleMsec * 2) {
                throw std::runtime_error("invalid spawn anchor or unsampled tail");
            }
            for (std::uint32_t j = 0; j < actions; ++j) {
                Action action{in.U32(), in.U32()};
                if (action.time >= clip.duration || (j && action.time < clip.actions.back().time)
                    || action.kind < Action::Reload || action.kind > Action::SecondaryShot) {
                    throw std::runtime_error("invalid action");
                }
                clip.actions.push_back(action);
            }
            library.clips.push_back(std::move(clip));
        }
        if (!in.Done()) throw std::runtime_error("trailing replay data");
        out = std::move(library);
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool Compatible(const Library& library, const std::string& map, std::uint32_t checksum,
                std::uint32_t gameType, std::uint32_t protocol) {
    return library.map == map && library.checksum == checksum && library.gameType == gameType
        && library.protocol == protocol;
}

float DistanceSquared(const Vec3& a, const Vec3& b) {
    float result = 0;
    for (int i = 0; i < 3; ++i) result += (a[i] - b[i]) * (a[i] - b[i]);
    return result;
}

float AngleDelta(float from, float to) {
    float delta = std::fmod(to - from + 180.0f, 360.0f);
    if (delta < 0) delta += 360.0f;
    return delta - 180.0f;
}

Frame Sample(const Clip& clip, std::uint32_t time) {
    if (clip.frames.empty()) throw std::invalid_argument("cannot sample an empty clip");
    auto after = std::upper_bound(clip.frames.begin(), clip.frames.end(), time,
        [](std::uint32_t t, const Frame& f) { return t < f.time; });
    if (after == clip.frames.begin()) return clip.frames.front();
    const auto& a = *(after - 1);
    // Preserve exact stored float values at sample timestamps; no rounding via interpolation.
    if (a.time == time || after == clip.frames.end()) return a;
    const auto& b = *after;
    Frame result = a; // discrete posture, weapon and input are sample-and-hold
    result.time = time;
    const float fraction = float(time - a.time) / float(b.time - a.time);
    for (int i = 0; i < 3; ++i) {
        result.origin[i] = a.origin[i] + (b.origin[i] - a.origin[i]) * fraction;
        result.velocity[i] = a.velocity[i] + (b.velocity[i] - a.velocity[i]) * fraction;
        result.angles[i] = a.angles[i] + AngleDelta(a.angles[i], b.angles[i]) * fraction;
        result.eyeOffset[i] = a.eyeOffset[i] + (b.eyeOffset[i] - a.eyeOffset[i]) * fraction;
    }
    result.lean = a.lean + (b.lean - a.lean) * fraction;
    return result;
}

std::vector<Action> DueActions(const Clip& clip, std::size_t& cursor, std::uint32_t time) {
    std::vector<Action> result;
    while (cursor < clip.actions.size() && clip.actions[cursor].time <= time) {
        result.push_back(clip.actions[cursor++]);
    }
    return result;
}

void SpawnSelector::Reset(std::uint32_t seed) {
    bags.clear();
    generator.seed(seed);
}

std::size_t SpawnSelector::Select(const Library& library, const Vec3& spawn, float tolerance,
                                 const std::set<std::size_t>& busy) {
    if (!std::isfinite(tolerance) || tolerance < 0 || tolerance > 32) return noClip;
    for (float coordinate : spawn) if (!std::isfinite(coordinate)) return noClip;
    std::vector<std::size_t> matches;
    Vec3 anchor{};
    bool found = false;
    for (std::size_t i = 0; i < library.clips.size(); ++i) {
        const auto& clip = library.clips[i];
        if (DistanceSquared(spawn, clip.spawn) > tolerance * tolerance) continue;
        // Never collapse two distinct nearby spawn points into one pool.
        if (found && DistanceSquared(anchor, clip.spawn) > 0.0001f) return noClip;
        anchor = clip.spawn;
        found = true;
        matches.push_back(i);
    }
    if (!found) return noClip;
    auto bag = std::find_if(bags.begin(), bags.end(), [&](const Bag& b) {
        return b.spawn == anchor && b.members == matches;
    });
    if (bag == bags.end()) {
        bags.push_back({anchor, matches, {}, 0, noClip});
        bag = bags.end() - 1;
    }
    if (bag->cursor == bag->order.size()) {
        bag->order = bag->members;
        std::shuffle(bag->order.begin(), bag->order.end(), generator);
        if (bag->order.size() > 1 && bag->order.front() == bag->last) {
            std::swap(bag->order[0], bag->order[1]);
        }
        bag->cursor = 0;
    }
    for (std::size_t i = bag->cursor; i < bag->order.size(); ++i) {
        if (!busy.count(bag->order[i])) {
            std::swap(bag->order[i], bag->order[bag->cursor]);
            bag->last = bag->order[bag->cursor++];
            return bag->last;
        }
    }
    return noClip;
}
} // namespace replay
