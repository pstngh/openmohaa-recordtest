# Spawn-indexed replay bots (experimental prototype)

This fork adds recorded-world-position playback with independently retargeted live aim.
It is not machine learning and does not replay the original match's damage or kills.
The original bot implementation is unchanged when `g_replay_bots` is zero (the default).

## First playable target

Protocol 8 (Allied Assault), FFA or respawning team deathmatch. Start with one bot on
`dm/crnodoors` using the same BSP as the recording. Round/objective modes are not
supported by this first runtime. An incompatible library fails closed; it does not
silently use ordinary navigation or an unrelated spawn's route.

A server-side rebuild is required. Use this branch's dedicated server and game module
with a normal licensed game installation and the matching map assets. No game assets
or player recordings are distributed in this repository.

The team/weapon-neutral runtime fix also requires a rebuilt game module: old builds
still restrict team-deathmatch pools and equip recorded weapons. Existing `OMRPL001`
replay packs are compatible; no reimport or new recording data is needed for this fix.

## Import recordings

Python 3.10 or newer; no third-party Python packages are required.

```sh
python tools/replay/import_recordings.py Archive.zip /path/to/MOHAA/main/replays --map dm/crnodoors
```

An unpacked directory is also accepted. Frame, event and metadata files are paired by
directory and filename suffix, including names such as `movement_frames (2).csv`.
Use an empty output directory when rebuilding a pack so old unreferenced libraries
cannot be confused with the new manifest. Restart the map after replacing libraries.

For the provided archive, the default quality-filtered first-map import produces
150 human lives / 76,800 frames in four spawn pools: 34, 45, 8 and 63 lives. The
previous 600-life pack is reduced to those 150 unmodified lives. Two retained clips
have an explicit synthetic spawn boundary before the first observed frame; observed
timestamps and positions are retained. These are input-specific validation results,
not a guarantee about other archives. The quality audit sees 601 structurally valid
candidates, including one sub-500-ms life the old default already excluded.

The resulting first-map file is:

```text
main/replays/dm/crnodoors.4277373344.1.rpl
```

The unsigned checksum in this name corresponds to the engine's signed checksum
`-17593952`. A different map build requires its own recorded library, even when the
map has the same name. The manifest lists the accepted clips, spawn pools, rejected
categories, end reasons and file SHA-256 hashes, without player names.

Without `--map`, the importer writes each available map/checksum/game-mode library.
The runtime still enforces its FFA/team-deathmatch and protocol-8 restrictions.
`--include-bots` additionally imports recorded bot lives; the default is humans only.
Metadata lacking a BSP checksum is skipped, not assigned the checksum of a newer map.

## Filter out short or inactive lives

Filtering is performed by the importer, not by changing playback. A rejected life
is removed from its spawn's pool in its entirety. An accepted life's positions,
timestamps, spawn, clip ID, weapons and actions are unchanged. Pauses are never
fast-forwarded or cut out, and a life is never restarted from its first moving frame.

The new defaults are:

| Import option | Default | Rule |
|---|---:|---|
| `--min-duration-ms` | `10000` | Reject lives shorter than 10 seconds of available playback. |
| `--max-stationary-ms` | `3000` | Reject any continuous stationary stretch longer than 3 seconds. Set `0` to disable this limit. |
| `--max-stationary-fraction` | `0.60` | Reject lives stationary for more than 60% of their playback time, even in several shorter pauses. Set `1` to disable this limit. |
| `--stationary-speed` | `10` | Horizontal game units/second at or below this threshold count as stationary. |

Activity is measured from horizontal position changes divided by elapsed time,
not recorded velocity, input buttons, aim changes, frame counts or net start/end
distance. Strafing back and forth counts as movement. Turning, shooting, leaning,
crouching and vertical-only jumping do not; slow drift and small positional jitter
at or below the threshold do not reset the stationary timer. Each sample interval
is weighted by its duration. The final held-position interval before the clip ends
is stationary even if the recorded velocity is nonzero. Thresholds are inclusive:
exactly 10 seconds of duration, a 3-second pause, or 60% stationary time is allowed.
Synthetic spawn interpolation, when present, is assessed as the playback it creates.

For example, explicitly select those defaults:

```sh
python tools/replay/import_recordings.py Archive.zip /path/to/new-pack/main/replays --map dm/crnodoors --min-duration-ms 10000 --max-stationary-ms 3000 --max-stationary-fraction 0.60 --stationary-speed 10
```

For more variety, a less restrictive preset is `--min-duration-ms 8000
--max-stationary-ms 4000`. The stationary-fraction limit still applies. To restore
the previous filtering policy, explicitly use `--min-duration-ms 500
--max-stationary-ms 0 --max-stationary-fraction 1`.

`manifest.json` now includes a `quality` section with the settings, per-clip activity
metrics and all failure reasons, plus spawn-pool counts before and after filtering.
Clip records contain hashed IDs, not player names. `quality.failure_counts` can
count a life against more than one failed rule; the main `rejected` totals count
each life only once, in duration / longest pause / total stationary time order.
Duplicate input clips are counted only once in the quality audit.

Empty spawn pools remain visible in the report and generate CLI warnings. Filters
are never silently relaxed to populate them. Bots at uncovered spawns still hold
and log a diagnostic rather than taking a different spawn's route. An import that
rejects every life fails with rejection counts and leaves existing output untouched.
Use a fresh output directory for each build: when a previously covered map is fully
filtered out, an existing library for that map is an error, not silently reusable data.

Filtering the original recordings does not fix a runtime collision-stop or missing
spawn match. Inspect the server's `Replay bot ... holding position` diagnostic when
a filtered clip stops unexpectedly.

**No new server build is needed for these filters.** The `OMRPL001` binary format and
engine are unchanged. Reimport or install the newly filtered `.rpl` plus manifest,
replacing the old library, then restart the map. Merely updating the Python importer
does not alter recordings already installed on a server.

## Start a local test

Before loading the map, execute these console commands (or put them in a server cfg):

```text
set g_gametype 1
set sv_maxclients 2
set sv_maxbots 1
set sv_numbots 1
set sv_fps 20
set g_replay_bots 1
set g_replay_aim 1
set g_replay_actions 1
map dm/crnodoors
```

Join normally. `sv_maxbots`, `sv_maxclients` and the game type are latched settings,
so apply them before loading/restarting the map. `g_replay_bots 1` applies to all
ordinary multiplayer bots; the settings above request one. Ordinary spawn selection
still chooses the spawn. The replay selector then uses only the pool matching that
actual spawned position. It never moves a route from a different spawn to fit.

Successful initialization prints `Replay: loaded ...` and each selected clip ID,
spawn position and duration. On an uncovered or ambiguous spawn, or a missing library,
the bot holds still and remains damageable. It does not invent a replacement route.
Enabling replay halfway through an existing life also holds until its next spawn.
Disabling `g_replay_bots` returns bots to ordinary control on their next think.

## Lifecycle and random selection

A clip starts at an explicit spawn event. Its timestamps are relative to that spawn.
Death events end the **target/victim's** life, not the actor/killer's. Spectator frames,
missing starts, big sampling gaps, teleports and ladder clips are not joined into a
fabricated continuous route. A sampling gap retains only a usable spawn-origin prefix.
Duplicate input clips are deduplicated.

Recorded teams and weapons are metadata, not movement restrictions. In both FFA and
team deathmatch, every compatible life anchored at the actual spawn enters one shared
pool, regardless of which team or weapon produced it. The live game still chooses a
legal spawn for the current team; recorded routes are not moved to other spawns.
Enemy selection still uses the live bot's team, not the recorded player's team.
Older importer manifests may list team-specific coverage counts; those describe the
source recordings, not separate runtime pools. Map/checksum/mode/protocol checks
remain in force; this does not introduce cross-game-mode playback or live stitching.

At each spawn, compatible lives enter a shuffled pool. The selector consumes the pool
before reshuffling and avoids an immediate repeat across cycles when possible. Clips
already in use by another bot are not handed out simultaneously. If all remaining
clips are busy, that spawn fails closed until the next spawn; this prototype does not
queue waiting bots or relax the no-repeat rule.

A live kill ends playback and leaves the normal game death/respawn rules in control.
If a living bot reaches the clip endpoint, the runtime resets it through `Respawn`
only in supported respawning modes when respawning is permitted. This is a replay reset,
not a fabricated death event: no attacker is awarded the original kill. If respawning
is currently forbidden, the bot holds at the endpoint. The next spawn picks a new life.

## Movement, aiming and actions

The replay controller owns the authoritative root position, velocity and stored bounds.
Normal `Pmove`, navigation, jump impulses and animation root motion cannot redirect it.
The adapter applies the replay before `ClientThink` targeting, inside `ClientMove`,
after animation-state evaluation, and around `FinishMove`. Server collision state,
player state and entity network origin are kept synchronized.

Replay bodies are kinematic and cannot be displaced by ordinary knockback or pushers.
They do not body-block players, including client prediction. The server retains a
weapon-hittable bounding box and normal live damage/death. This is deliberate: ordinary
physical displacement and an immutable recorded trajectory cannot both take priority.

Live aim chooses a visible enemy in the current match, with a field of view, reaction
delay, bounded turn rate and configurable aim error. It does not call the ordinary
attack state, which can change movement. No target means returning toward recorded aim.

The live game owns the bot's assigned weapon and ammunition. Replay does not grant,
equip or switch recorded weapons, cancel the spawn's pending weapon draw, or restore
recorded ammunition. A live Allied sniper can follow an Axis SMG recording and remain
an Allied sniper. Normal live loadout changes are respected. An unavailable, switching
or reloading live weapon suppresses combat requests, not movement playback.

With `g_replay_actions 1`, recorded primary-fire intent is adapted to the current
weapon: automatic weapons receive trigger holds; semi-automatic weapons receive
press/release edges only while their own idle/readiness rules permit. Native fire
rate, ammo use, draw/reload animations and movement-dependent firing restrictions
remain authoritative. The controller never stops the route to let a sniper shoot.
It requests normal reloads only when the live weapon lacks loaded ammunition and
can reload; it does not inject ammo or replay the donor's reload schedule.

Recorded weapon changes, secondary fire/melee/scope inputs, zoom state and shot/reload
events remain in the data for analysis but do not execute combat actions. In particular,
a donor's scope press cannot become a melee attack on an SMG. This version does not
automatically scope a live sniper based on donor input. Existing live scope state is
left to the native weapon system. Recorded damage, health and kill credit are never
applied. `g_replay_actions 0` suppresses all replay fire/reload requests.

`USE` input is deliberately not executed: with a different aim it could attach the bot
to the wrong ladder, turret or vehicle. The original world is not being replayed.
Interactive doors, triggers/pickups, ladders, water/vehicle traversal, moving platforms,
objective interactions and script-controlled attachments are not validated features.
The runtime conservatively stops a clip if its swept path is blocked by current world
geometry. It does not clip through a newly closed door or reroute. Coarse recorded
samples can also make this check reject a stair/corner segment; inspect that clip rather
than disabling collision checks to claim it was faithfully reproduced.

## Fidelity boundary

At an observed sample timestamp, the pure playback sampler returns the exact stored
float32 position/velocity without interpolating it. Between observations it interpolates
position/velocity and uses shortest-angle interpolation. Discrete posture and buttons
are held until the next sample. The server tick interval must match the library's sample
interval (`sv_fps 20` for this archive). A playback-clock jump stops the clip.

This is exact playback **of the retained samples**, not reconstruction of unrecorded
engine substeps. CSV decimal rounding, inferred lean (the source stores lean inputs,
not authoritative lean angles), synthetic spawn boundaries, animation selection,
network quantization and client rendering are not lossless. Weapon timing and the
whole rendered pose are not promised bit-identical. Changes in aim necessarily change
some orientation/animation. The library is immutable; targeting does not edit its route.

## Settings

| Setting | Default | Meaning |
|---|---:|---|
| `g_replay_bots` | `0` | Enable replay-controlled multiplayer bots. Set before map/spawn. |
| `g_replay_aim` | `1` | Retarget visible live enemies; `0` uses recorded angles. |
| `g_replay_actions` | `1` | Adapt primary-fire intent to the live weapon and reload on live ammo need; `0` disables requests. |
| `g_replay_debug` | `0` | Log selected clip and end-of-frame position checks each replay tick. |
| `g_replay_spawn_tolerance` | `8` | Spawn-match radius, bounded to 0..32 units. Multiple distinct matching anchors are rejected. |
| `g_replay_turn_speed` | `360` | Maximum aiming turn speed in degrees/second. |
| `g_replay_reaction_ms` | `200` | Delay before aiming at a newly noticed target. |
| `g_replay_aim_error` | `1.5` | Amplitude of deterministic aiming error in degrees. |
| `g_replay_seed` | `1` | Reproducible pool shuffling within the same build; loaded on map/library initialization. |

For baseline movement validation, set aim and actions to `0`, and debug to `1`.
`replay_sample` prints the clip-local time, live origin, and separate root, player-state
and network-origin errors at the **end of the player frame**, after normal animation
and view updates. Drift above 0.001 units is printed even when verbose debug is off.
Repeat with live aim/actions enabled and compare those errors; targeting should not
introduce trajectory drift. Client-side rendered positions are a separate test.

## Automated tests and build

The independent tests do not require game assets, Flex, Bison or the full engine build:

```sh
cmake -S tools/replay -B build-replay-tests
cmake --build build-replay-tests
ctest --test-dir build-replay-tests --output-on-failure
```

Validate a generated pack with the compiled test executable:

```sh
./build-replay-tests/replay_tests /path/to/MOHAA/main/replays/dm/crnodoors.4277373344.1.rpl
```

It validates the binary format and checks every retained sample for exact playback.
The Python suite covers segmentation, victim attribution, missing starts/checksums,
gaps, spectator/bot filtering, duplicate sources, ZIP input, added CSV columns, unsafe
map names, invalid numbers and event boundaries. Quality-filter tests cover exact
thresholds, time-weighted inactivity, stationary tails, XY-only motion, jitter, empty
spawn pools, stale libraries, invalid settings and unchanged accepted binary payloads. The C++ suite also checks malformed
and truncated files, sample interpolation, action delivery, team/weapon-neutral spawn pools,
busy-clip exclusion, reproducible shuffling and non-repeating pool cycles. The live
combat policy tests cover automatic holds, semi-auto releases, native readiness,
draw/switch waits, empty versus partial magazines, disable/reset and immutable samples.

Full server/game build (requires the project's normal build dependencies):

```sh
cmake -S . -B build-replay -DCMAKE_BUILD_TYPE=Debug -DBUILD_CLIENT=OFF -DBUILD_RENDERER_GL1=OFF -DBUILD_RENDERER_GL2=OFF -DBUILD_GAME_QVMS=OFF
cmake --build build-replay --parallel 2
```

The `Replay validation` workflow runs the replay, importer and live-combat test suites and the server/game build.
Automated sampling and compilation do not substitute for an in-game smoke test with
the original assets. Validate collision, visible animation, weapon timing, death,
respawn and pool selection on the target installation before increasing bot count.

## Files and format

`replay_track.*` is an engine-independent bounded parser/sampler/selector.
`g_replay.*` owns the engine adapter; narrow hooks in `player.cpp`, `player.h` and
`playerbot.cpp` select or protect replay motion. The importer is in `tools/replay`.

`OMRPL001` uses explicit little-endian integers and IEEE-754 float32 values, not native
C++ struct dumps. Strings are length-prefixed printable ASCII. The parser bounds file
size (128 MiB), counts, times, coordinates, commands, tables and finite numeric values;
a failed parse does not replace a previously valid library. Maps are path-validated.
The matching identity includes map, checksum, mode and protocol. The format's source
of truth is the paired parser/importer and their binary-contract tests. Libraries are
server-local data and are never executed as script or console commands.
