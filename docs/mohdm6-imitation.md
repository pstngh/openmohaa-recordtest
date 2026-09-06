# mohdm6 native-control imitation — experimental

This is a separate experiment based on the native bot implementation. It does not
load replay lives, write player positions, change collision solidity, or respawn a
living bot because a recording finished. The ordinary game simulates commands and
owns animation, collision, damage, ammunition and real respawns.

The goal is resemblance to the supplied mohdm6 demonstrations, including movement
and anticipatory crosshair placement. There is no tactical planner, route scoring,
reinforcement-learning reward, nearest-enemy aim override, or other-map training data.
This does NOT establish that autonomous bots already reproduce a human faithfully.
One-step prediction metrics and successful builds are not an in-game evaluation.

## Architecture

A 64-unit causal GRU consumes 54 current/past observation features. Four mixture
components jointly describe XY movement, vertical input, lean, run/fire/secondary/use
buttons, and two view-angle changes. A training-only future-view auxiliary head helps
learn temporal context; future information is never supplied to runtime observations.

Position, velocity, view direction, posture, lean, live weapon class/ammunition,
previous executed controls and realized view change provide the primary features.
Map-local Fourier position features allow learning location-dependent gaze patterns.
Visible near-crosshair target estimates provide limited encounter context. Team is
not a policy feature; native team checks determine which visible players are enemies.
The runtime target approximation uses centroid, FOV and line of sight. It is not
proven identical to the client recorder's inferred target selection.

Movement and camera originate in the SAME learned policy. The native movement,
rotation and attack state routines do not overwrite it while imitation is active.
Per-bot recurrent state and random generators are independent; immutable weights are
shared. State resets on actual spawn/death/removal, not a demonstration endpoint.
There is no life playback cursor or requirement to find matching trajectory endpoints.

Argmax/MAP decoding tends to repeat the previous input because changes are rare at
50 Hz. The default therefore samples learned joint component/category probabilities.
Four persistence offsets are fitted on validation demonstrations; no arbitrary hold
timers or random destinations are added. View deltas use component means, NOT Gaussian
noise samples. This is not proof of smooth autonomous movement or correctly timed
turns. `g_imitation_sampling 0` exposes MAP decoding for diagnosis, not a recommended
fix for navigation. `mode_weight` in logs is mixture probability, not a reliability score.

## What the recordings do and do not provide

Only `maps/dm/mohdm6.bsp`, checksum `1974169620`, client-predicted schema 1 captures
are accepted. Alive non-spectator observations are used; death, special movement,
teleports and timestamp discontinuities break recurrent sequences. Legitimate short
lives and stationary aiming are NOT discarded using the replay duration filters.

The input files are change logs and OMIT mouse-only changes. Movement/buttons are
validated against forward-filled command changes. Camera labels are successive
realized VIEW changes, not sparse `cmd_pitch`/`cmd_yaw` entries or raw mouse intent.
`cmd_msec` is zero in this archive and is not used. Command and client timestamps are
checked; labels summarize the following 10–40 ms, normalized to 20 ms for the camera.
This cannot reconstruct every original input substep or separate all recoil effects.

Complete capture UUID groups are assigned to training, validation or test; multiple
files from one UUID cannot leak across splits. Normalization is fitted on training
only. Validation selects the checkpoint and persistence calibration. Test data do
not update weights or select hyperparameters. Reports include persistence baselines,
change-event statistics and camera errors, not just flattering overall accuracy.

Directional trace features are deliberately omitted because the recorder's exact
geometry implementation was not supplied. Pixels, audio, intended target labels and
corner annotations are not reconstructed. Learning pre-aim is an objective, not an
already verified result; per-corner and autonomous comparisons require the actual BSP
and game installation. There is client-to-server observation distribution shift.

## Native combat safeguards

The policy can pre-aim at corners without a visible enemy. A separate firing guard
requires a currently visible live enemy, aligned aim, clear firing line and muzzle.
It blocks teammate/geometry obstruction; it NEVER turns the camera toward a target.
Optional `g_imitation_reaction_ms` adds a minimum firing delay (default 0: response
timing comes from the learned policy). Actual bullets and native weapon timing still
follow the normal game; already launched shots are not recalled.

The bot retains its live loadout. This prototype does not learn inventory selection
or donor weapon changes. Native ammo-based reload on an empty clip is a safeguard,
not learned partial-reload timing. Secondary intent can use a live sniper scope, not
an SMG melee/launcher action. Unsupported attachments/script freezes release controls.
Native physical collision still applies; no learned navigation reliability is promised.
There is no forced respawn, teleport or tactical fallback to hide an imitation failure.

When imitation is disabled, model loading fails, or the map/mode/tick rate is unsupported,
ordinary native bots remain active with an explicit diagnostic. Do not mistake that
fallback for successful model inference. A valid startup MUST print `Imitation: loaded`.

## Installation

Build the separate `feature/mohdm6-imitation` branch. It starts from native `main`,
NOT from the replay branch. Preserve custom Grok changes before switching. Do not merge
replay positioning/recovery hooks into the learned-control path. Existing `.rpl` packs
are neither needed nor read. `main` and the old replay branch are left separate.

Install the externally supplied model as:

```
main/bots/imitation/mohdm6.omim
main/mohdm6_imitation.cfg
```

Trained weights and raw user data are intentionally not included in the public repo.
The native runtime has no Python, PyTorch, ONNX or external inference-server dependency.
The model uses a bounded explicit little-endian float32 format (`OMIM0001`), with exact
feature/shape/map/tick contract checks and transactional loading. No serialized code is
executed by the server.

Run `exec mohdm6_imitation.cfg` in the rebuilt server console. It loads FFA mohdm6 at
50 Hz with ONE bot. Check startup and `imitation_control` logs. The same map's TDM is
accepted but was not the training mode; do not assume equivalent behaviour. Other maps,
protocols, game modes and tick rates are rejected. Restart the map after replacing weights.

The config reserves eight separate bot slots. Increase `sv_numbots` only after a
one-bot smoke test; multi-bot inference and collision behaviour require evaluation.
A human client and matching licensed map/game assets are not included in this project.

## Training and validation

Python 3.10+ with the requirements in `tools/imitation/requirements-train.txt` is needed
ONLY to train or check the export. CPU training is supported. Use a separate directory:

```sh
python tools/imitation/prepare.py client_telemetry.zip /path/to/private-data
python tools/imitation/train.py /path/to/private-data /path/to/private-model --epochs 18
```

The default seed and group split are fixed. Repeatability is for the same software
stack, not a guarantee of bit-identical training across every platform. `best.pt` is
selected before calibration; deploy the final `mohdm6.omim`, not an intermediate export.

Asset-free native/data tests (NumPy needed for Python tests):

```sh
cmake -S tools/imitation -B build-imitation-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build-imitation-tests --config Release --parallel 2
ctest --test-dir build-imitation-tests -C Release --output-on-failure
```

Check a locally produced checkpoint and its native export, adjusting executable paths:

```sh
python tools/imitation/check_export.py /path/to/private-model/best-calibrated.pt /path/to/private-model/mohdm6.omim /path/to/private-data/test.npz ./build-imitation-tests/imitation_tests export-parity.json
```

Full server/game build, using the project's normal dependencies:

```sh
cmake -S . -B build-imitation -DCMAKE_BUILD_TYPE=Debug -DBUILD_CLIENT=OFF -DBUILD_RENDERER_GL1=OFF -DBUILD_RENDERER_GL2=OFF -DBUILD_GAME_QVMS=OFF
cmake --build build-imitation --config Debug --parallel 2
```

## Required gameplay evaluation

Compare actual autonomous approaches and crosshair placement against held-out human
sequences. Measure starting a turn before a corner, crosshair height, strafe/lean timing,
route departure, stuck time, shooting permissions and lifecycle behaviour. Also inspect
actual animations and server CPU cost with multiple bots. A sampled conditional policy
may drift off the demonstration distribution or remain stuck; high one-step accuracy
does not rule this out. No near-perfect imitation or bug-free rollout claim is made.

Keep the old native/replay builds and configs for rollback. Do not replace a production
server based only on the offline scores. New corrections should be recorded as human
demonstrations and evaluated on separate capture groups, not hidden behind teleports.
