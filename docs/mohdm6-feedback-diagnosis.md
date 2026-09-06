# v1 imitation feedback failure and runtime corrections

The delivered v1 checkpoint is not a satisfactory autonomous imitation controller.
Reports of circling, little useful movement, downward gaze and little shooting must
not be answered by claiming that the earlier one-step/export tests proved gameplay.
The following changes repair adapter contracts and expose live evidence. They do
NOT replace the model or establish that these behavioral failures are solved.

## Verified adapter corrections

The training features contain the preceding requested button input. The old native
adapter instead fed back its fire/scope/USE-filtered command. A denied primary-fire
request therefore became 'not firing' in the policy's history. ControlFeedback now
keeps intent separate from the actual sent buttons; weapon and visibility guards
still determine the actual command. No denied shot is reinstated by this fix.

Identical simulation timestamps reuse the cached command instead of advancing the
GRU again. Disabling imitation clears that bot's state; attachment waits clear
trigger history. Camera increments scale with elapsed time and use the native
PM_UpdateViewAngles signed-short pitch clamp, with wrapped yaw. These repairs do
not force horizontal aim or steer toward an enemy.

A model fingerprint and decoder label distinguish the loaded data from the new
runtime. 'feedback-v2' is the runtime version, NOT a claim that new weights loaded.
Existing OMIM0001 weights remain readable. There is no replay, living respawn,
teleport, tactical planner or new auto-aim fallback.

## Model failure is a separate issue

The v1 policy strongly depends on previous controls/view change, and its calibration
adds positive persistence offsets. Accurate one-step copying is not reliable control
when errors feed back. Its view deltas can accumulate into sustained turning and
pitch drift. Bad gaze can then cause a valid firing guard to deny most shots.
Removing the guard would restore shooting without fixing aiming or safe firing.

feedback_probe.py runs an explicitly synthetic fixed-body observation stress test:
position fixed, velocity zero, grounded, no target, then recurrent predictions and
quantized camera updates are fed back. It does not load a BSP or simulate physics,
weapons or target perception. It can expose feedback instability, but cannot certify
navigation, attack quality or corner pre-aim. Both sampled and MAP decoding should
be examined; disabling sampling is not an established repair for this checkpoint.

Example, after preparing the data and using the externally supplied checkpoint:

```sh
python tools/imitation/feedback_probe.py best-calibrated.pt validation.npz probe.json --decoder sampled
```

A subsequent observation-perturbation/history-dropout training experiment reduced
some stress-test failures but still showed excessive turning and weak movement
transition prediction. Its checkpoint is not promoted or included in this commit.
The training labels in that experiment included synthetic orientation corrections;
it must not be described as expert-labelled on-policy recovery data or DAgger.

## Capture the actual failure in a rebuilt diagnostic installation

Preserve custom changes and use the native imitation branch, not a replay build.
Rebuild the game module with this commit. Keep the current model identifiable.
Start ONE bot, then enable:

```text
set logfile 1
set g_imitation_debug 2
```

Check for 'Imitation runtime feedback-v2'. Level 2 emits one imitation_frame line
per inference tick with model ID, per-spawn sequence, current view, requested/sent
controls, visible/aligned target flags, fire permission and all 54 input features.
Use it only for a short diagnostic capture because 50-Hz logging has overhead.
Turn it back off afterward with 'set g_imitation_debug 0'.

```sh
python tools/imitation/audit_runtime_log.py qconsole.log imitation-runtime-report.json
```

Provide the original diagnostic lines along with the summary. The analyzer does
not infer actual bullets from sent trigger inputs or call looking down a bug in
every context. It separates intent, gating, motion and observation-history errors.
A short spectator video is useful to align the numbers with visible behavior.

The committed tests cover feedback contracts, tick/angle edge cases and logging.
They are not a substitute for an actual match. A replacement model needs explicit
multi-step view stability, movement-transition and in-game rollout evaluation;
not just more epochs or a successful native build.
