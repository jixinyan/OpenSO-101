# Development Diary

> _Ported from `safe_sim2real/docs/development_diary.md` during the OpenSO-101 v0.1.0 refactor. Code references updated; historical narrative preserved._

A running log of project progress, decisions, and milestones. **Append the newest entry at the top, oldest at the bottom.** Keep entries short and dated: what changed, why, and what's next. This is the canonical source for "what has been done recently" — anyone picking up the project should read from the top until they've built enough context.

Commit this file alongside feature work so each entry lands with the code it describes.

---

## 2026-05-14: RL pipeline restoration after USD-canonical regression

### Findings

User reported the RL pipeline broken: floating arm base, robot whipping at high speed, gripper stuck open, training metrics stagnant, losses not converging. Investigation traced the regression to commit `5d01353` (2026-05-13) "refactor(robots): make upstream SO101 canonical", which swapped the canonical `SO_ARM101_CFG` from a URDF spawn (with `fix_base=True` and proven actuator gains) to a USD spawn that kept weak hand-tuned actuator values from a less-tested USD variant.

### Plan executed

Implemented in commits `c287ac4..427e07c`, subagent-driven per `docs/superpowers/plans/2026-05-14-canonical-so101-actuator-restoration.md`:

| Commit | Change |
|---|---|
| `c287ac4` | Restored URDF-era arm + gripper actuator gains on canonical `SO_ARM101_CFG`: effort 1.9 (arm) / 2.5 (gripper), velocity 1.5 everywhere, stiffness 200/170/120/80/50 arm + 60 gripper, damping 80/65/45/30/20 arm + 20 gripper |
| `805794c` | Restored articulation properties: `enabled_self_collisions=True`, solver iters `8/0`, `soft_joint_pos_limit_factor=0.9`; kept `fix_root_link=True` |
| `1fed937` | Dropped the now-redundant teleop actuator override (canonical is the proven config now) |
| `4e2d0e5` | Disabled silent `enable_corruption = True` no-op in `composite/lift` |
| `d960d4a` | Same `enable_corruption` cleanup in 6 atomic/composite tasks (grasp, place, transport, return_home, atomic-lift, stack). `composite/reach` correctly skipped — its ObsTerm sets `noise=` |
| `70ab579` | PPO `noise_std_type="log"` to prevent the negative-sigma crash that hit iter 79 of a verification run |
| `427e07c` | Reverted experimental init-pose change (URDF→USD joint zeros don't map 1:1 by angle) |

### Verification

- All 87 static tests pass / 6 skip (Isaac Sim-dependent skips, expected).
- Restored values confirmed live in `logs/rsl_rl/lift/2026-05-14_14-09-21/params/env.yaml`.
- Pick-place 200-iter training (`logs/rsl_rl/pick_and_place/2026-05-14_14-36-*`): at iter 140 mean reward 20.78 (vs 2.04 at iter 24), reach 0.80 (EE ~2 cm from cube), close-gripper-proxy max'd, `object_dropping: 0`, losses healthy. `grasped` (contact-confirmed pinch) not yet firing at 200 iters — that is reward-shaping, not pipeline correctness.

### Codebase health (separately surveyed)

- No URDF joint-name leaks in active code (only in comments / lerobot teleop side, where they refer to physical leader-arm joints).
- No import cycles. `composite → atomic/common` is one-direction.
- All 9 `[project.scripts]` entrypoints resolve to existing `main()` functions.
- No `fix_base=True` / `disable_gravity=True` legacy flags. No BLOCKER findings.
- Three restored-actuator pin tests skip under `importorskip("isaaclab.sim")` since the sim app isn't running in CI — values confirmed live in persisted env.yaml.

### Next

Reward shaping to bridge the close-gripper-proxy → contact-confirmed-grasp gap (the policy converges to "close near cube" but doesn't pinch). Likely the `grasped` weight should grow during training (curriculum), or a third reward term should reward the *cube-between-jaws* geometry. Separate from this restoration work.

---

- 2026-05-13: SO101 object physics unified for RL and teleop.
  All SO101 manipulation task configs now construct cube-like objects through a
  shared Isaac Lab object factory. The default is a 3 cm `CuboidCfg` rigid
  object with explicit mass, contact offsets, friction, and restitution, so RL
  and teleop no longer drift into different cube physics. The old heavy 100 g
  cube configs, the DexCube USD path, and the teleop-only cube physics override
  were removed. The upstream SO101 gripper/jaw collision prims now get stable
  convex contact geometry, contact offsets, and a high-friction contact
  material at spawn time. The experimental prebuilt-block teleop option was
  removed after the block mesh/body layout proved unreliable with the SO101
  gripper.

- 2026-05-13: SO101 config surface simplified across the whole task suite.
  Removed the legacy URDF articulation config, the separate upstream alias, the
  `--mapping-profile` teleop flag, and the duplicate `SO-ARM101-USD-*`
  PickPlace task IDs. Shared SO101 joint names and gripper open/close values now
  live in a pure robot constants module used by all task action/reward/terminal
  configs and by teleop data code. RL no longer imports teleop mapping code;
  teleop still records semantic LeRobot joint names plus actual simulator joint
  names.

- 2026-05-13: SO101 teleop stabilized on the upstream USD robot. The canonical
  `SO_ARM101_CFG` now uses the upstream LeRobot SO101 USD asset, including the
  gripper camera mount at `Robot/gripper/gripper_cam`; the overhead camera is
  still provided by Safe Sim2Real. Teleop data remains local HDF5 under
  `teleop_data/` with state/action plus wrist and overhead images. The gripper
  and jaw collision groups are rewritten from SDF to convex decomposition at
  spawn time to avoid Isaac Sim 5.1 GPU PhysX narrowphase failures during cube
  contact. Operating notes and troubleshooting are in
  `docs/lerobot_so101_teleop.md`.

- 2026-05-12: Safe-RL pipeline complete — PPO-Lag, CPO, FOCOPS all train
  end-to-end on PickPlace. Cost signal: joint-velocity-violation +
  cube-drop event. New env subclass ManagerBasedSafeRLEnv attaches a
  sibling cost_manager. Six new task IDs registered. Spec at
  docs/superpowers/specs/2026-05-12-safe-rl-omnisafe-integration-design.md.

- 2026-05-12: Safe-RL PPO-Lag pipeline trains end-to-end on PickPlace (smoke @ 30 iters).

## 2026-05-10 — LeRobot teleop boundary

- **What:** Added a native LeRobot boundary layer for real SO101 leader-arm teleoperation of Safe Sim2Real Isaac Lab tasks. Safe Sim2Real remains the source of truth for the simulated URDF robot, task IDs, camera factories, rewards, and domain-randomization stack.
- **First target:** Real SO101 leader arm -> LeRobot action dict -> Safe Sim2Real absolute joint targets -> simulated SO-ARM101 PickPlace teleop task. Launch with `python -m safe_sim2real.scripts.lerobot.teleop_agent --task SafeSim2Real-SO-ARM101-PickPlace-Teleop-Vision-v0 --leader-port /dev/ttyACM0 --leader-id leader_arm_1`.
- **Data collection:** Teleop records automatically by default into local HDF5 under `teleop_data/`. Episodes include `observations/qpos`, `observations/qvel`, `action`, `timestamps`, `observations/images/wrist_camera`, and `observations/images/overhead_camera`; `C` creates an in-memory checkpoint, `R` restores the sim and recording buffer to the last checkpoint, `Q` quits while cancelling the active episode, and `S` saves/starts episodes manually. Reaching the final goal region prompts for save/discard and marks saved episodes with `success=True`. Graphical launches try to open default, wrist, and overhead viewports automatically. Hub upload is intentionally separated into `python -m safe_sim2real.scripts.lerobot.push_dataset`, which exports HDF5 episodes to LeRobot format before upload.
- **Transfer rationale:** Teleop variants use direct six-joint position targets so simulated demonstrations can later be exported in LeRobot's SO101 schema without mixing hardware actions with PPO-normalized offset actions.

## 2026-05-10 — Lift and PickPlace moving-goal curriculum

- **What:** Reworked atomic Lift around a visible green goal sphere that is higher than before and slightly offset from the cube spawn, with a sparse `success` reward when the cube enters the goal region.
- **PickPlace:** Simplified the composite task to stay close to the Lift reward layout. The only intended difference is the custom three-stage goal command: lift above cube spawn, carry above final placement XY, then place on the table at that same XY. Stage-specific rewards are gated by the command's stage so later rewards do not accumulate before earlier stages succeed.
- **Usability:** Tightened the PickPlace cube reset range so starts stay near the SO-101 graspable workspace, and updated training-curve labels for the new stage rewards.
- **Docs:** Added a repo-specific Isaac Sim / Isaac Lab learning guide and kept command/reward examples aligned with the current curriculum API.
- **Next:** Run short Isaac Lab smoke training for `SafeSim2Real-SO-ARM101-Atomic-Lift-v0` and `SafeSim2Real-SO-ARM101-PickPlace-v0`, then inspect stage reward curves and success termination rates.

## 2026-04-24 — PickPlace anti-push reward gate

- **What:** Tightened composite PickPlace against the observed push-to-goal exploit. Lift credit now requires several consecutive controlled-lift steps, and placement/release/success require a stored "carried under gripper control near the goal" flag instead of only "was lifted once".
- **Anti-cheat:** Added a shaped penalty and failure termination for moving the cube too far in XY while it is not controlled by the gripper and has not already been carried near the goal. Small incidental contact remains inside a deadband so grasp attempts are not punished immediately.
- **Docs:** Updated the composite task design guide with the reward-design rule: prove the intended manipulation sequence, not just final object pose.
- **Next:** Run a short PickPlace training/play smoke test and check that `object_pushed_without_grasp` fires on shove attempts while successful trajectories still reach `released_at_goal`.

## 2026-04-24 — Domain randomization branch plan

- **What:** Started the next feature branch from the merged reward-gated `main` and clarified the domain-randomization plan in the README and shared DR module notes.
- **Approach change:** Treat DR as a staged, centralized robustness stack instead of per-task ad hoc ranges. Phase 1 physics DR remains the shared base; Phase 2 will add state/action robustness before any render-dependent visual work; Phase 3 visual DR stays scoped to `-Vision` variants so state PPO training does not pay unnecessary rendering cost.
- **Coverage rule:** Make task wiring explicit and auditable. Atomic skills and the Stack pilot are the first coverage target, then remaining composite tasks once the reward curves still match no-DR baselines closely enough.
- **Next:** Audit which task cfgs already call `attach_all_physics_dr`, wire any missing intended coverage, then add the first observation/action DR helpers behind conservative ranges.

## 2026-04-24 — Reward-gated task cleanup across atomic and composite tasks

- **What:** Scanned every task reward and tightened shortcut-prone terms around explicit gripper control. Added shared helpers for gripper joint position, object speed/static checks, close-near-object rewards, and "object controlled by gripper" checks in [`src/safe_sim2real/tasks/atomic/common/rewards.py`](../src/safe_sim2real/tasks/atomic/common/rewards.py).
- **Task updates:** Grasp, Lift, Transport, Place, composite Lift, PickPlace, and Stack now use stricter reward/success gates that require controlled lift, release, and/or settled objects where appropriate. Reach now has an early success termination instead of timeout-only episodes. Free-object manipulation tasks now reset with the gripper open so close-gripper rewards reflect learned behavior rather than the default pose.
- **Why:** Several previous rewards could be satisfied by bumping, flicking, height-only object motion, or position-only placement without proving a real grasp/place sequence. The new structure makes success predicates stricter than shaping rewards and reduces obvious reward hacks before sim-to-real evaluation.
- **Next:** Run short Isaac Lab smoke tests for the changed task IDs, then retrain affected baselines and tune thresholds/weights from the logged success and termination curves.

## 2026-04-22 — Pick-place anti-cheat reward shaping

- **What:** Tightened the composite PickPlace task around a fixed table goal and open-gripper reset. Rewards now require a gripper-controlled lift before carry/place/release rewards activate, and success requires prior lift, goal placement, open gripper, and settled object/robot state.
- **Why:** The previous success predicate could be satisfied by object-at-goal plus arm-returned, and the sampled goal overlapped the randomized cube starts. That allowed PPO to exploit shortcuts instead of learning the intended pick-lift-carry-place behavior.
- **Also:** Updated the training plot helper so the new close-gripper and release-at-goal reward terms show up in per-stage curves.

## 2026-04-13 — Camera calibration, Phase-1 physics DR, training videos default-on

- **Cameras re-calibrated against the real rig.** The `cameras.py` constants now come from a GUI calibration pass: overhead at `pos=(0.39028, -0.01593, 0.30459)` / quat `(0.71660, 0.06054, 0.06269, 0.69201)` (focal 10), wrist at `pos=(0.00758, 0.03617, -0.02181)` / quat `(-0.70242, 0.08132, 0.06176, 0.70440)` (focal 8). Camera convention flipped from `world` to `opengl` in both factories because USD cameras are OpenGL-native, and `world` was introducing a 90° frame rotation that made GUI tweaks not match rendered frames. Workflow for future re-calibration: tweak in Isaac Sim GUI → read quaternion directly via Script Editor (`UsdGeom.Xformable(prim).GetLocalTransformation().ExtractRotationQuat()`) → paste into `cameras.py`. No Euler conversions involved.
- **Object spawn bounds widened in +X** so blocks land "beyond the gripper" inside the overhead camera frame. Applied to stack, pick_place, lift, grasp, return_home. Y ranges unchanged.
- **Phase 1 domain randomization landed.** New shared module [`src/safe_sim2real/tasks/domain_randomization.py`](../src/safe_sim2real/tasks/domain_randomization.py) exposes tunable range constants and `attach_all_physics_dr(events, ...)` helpers. Covers robot link masses, actuator PD gains, joint friction + armature, per-object mass and contact material, and small gravity noise. Wired into the Stack task as the pilot; reward curve to be validated before replicating to other tasks. Known gotcha: Isaac Lab 2.3's `randomize_joint_parameters` / `randomize_actuator_gains` need an explicit `joint_names=".*"` on the `SceneEntityCfg` to avoid a tensor-shape bug on the `slice(None)` path — baked into the helpers.
- **Training videos default-on.** `scripts/rsl_rl/train.py` switched `--video` to `BooleanOptionalAction` with `default=True`. Every run writes clips into `logs/rsl_rl/<experiment>/<run>/videos/train/` every `--video_interval` env steps. Pass `--no-video` to suppress. State-based tasks now implicitly enable cameras to render the video buffer — slight throughput cost, but the record-what-we-trained-on trail is worth it.
- **Next:** Validate Stack DR training reward ≈ no-DR baseline. Once good, replicate to the other 4 tasks. Then Phase 2 (observation DR: higher joint-state noise, action latency, action noise) and Phase 3 (visual DR: camera pose jitter, lighting, textures) — visual DR will piggyback on the state/vision scene split so it only applies to `-Vision` variants.

## 2026-04-13 — State/vision camera split

- **What:** Centralized the SO-101 camera rig defaults as named constants in [`src/safe_sim2real/robots/trs_so101/cameras.py`](../src/safe_sim2real/robots/trs_so101/cameras.py). The wrist camera remains parented under `Robot/gripper_link/wrist_cam` so it follows the gripper, while the overhead camera remains fixed in the per-env frame.
- **Also:** Split every task scene into a state-only base `SceneCfg` and a `SceneCfgWithCameras` variant. Existing `-v0` / `-Play-v0` task IDs are now state-only; new `-Vision-v0` / `-Vision-Play-v0` IDs opt into the overhead + wrist cameras. Launch scripts only force `--enable_cameras` for video or `-Vision` tasks, and `preview_cameras.py` now defaults to `SafeSim2Real-SO-ARM101-Grasp-Vision-Play-v0`.
- **Note:** No new measured hardware extrinsics were added in this pass; the current constants are the single source of truth until the physical rig is re-measured and verified with `preview_cameras.py`.
- **Next:** Use the real rig measurements to update the camera constants if needed, then start visual/randomization and state → vision distillation work.

## 2026-04-12 — Sim2real camera rig in every scene

- **What:** Added the real-rig dual-camera setup to every task scene. Two factory functions live in [`src/safe_sim2real/robots/trs_so101/cameras.py`](../src/safe_sim2real/robots/trs_so101/cameras.py):
  - `overhead_camera_cfg()` — world-fixed (per-env) `TiledCameraCfg` sitting in front of and between the leader/follower arms, pitched down toward the table.
  - `wrist_camera_cfg()` — `TiledCameraCfg` parented to the follower arm's `gripper_link`, optical axis along the fingertip direction.

  Both are instantiated directly as fields on every `InteractiveSceneCfg` across `tasks/atomic/*` and `tasks/composite/*`. `PickPlaceSceneCfg_WithCameras` was consolidated into the base `PickPlaceSceneCfg`. Defaults use a ~47° HFOV pinhole at 128×128 RGB.

  A new script [`src/safe_sim2real/scripts/preview_cameras.py`](../src/safe_sim2real/scripts/preview_cameras.py) dumps still frames from both cameras to PNG and accepts `--overhead_pos/rpy` and `--wrist_pos/rpy` CLI overrides for fast pose iteration without editing Python.

- **Why:** Everything required for sim2real vision transfer — known intrinsics, fixed extrinsics, GPU-parallel rendering — should be baked into the scene so state-based policies and later distilled vision policies share the exact same world. Same principle as keeping the robot URDF in every task: the sim should match the hardware, full stop.

- **Trade-off / known issue:** Because cameras are in every scene by default, every run now requires `--enable_cameras`, and state-based PPO training pays a rendering cost it doesn't use. The clean fix is to split each task into a state `SceneCfg` and a `SceneCfg_WithCameras` subclass (registered under `-Play` / vision gym IDs). Not yet done — flagged for next session.

- **Next:**
  1. Measure the real overhead tripod + wrist mount and lock in the default poses in `cameras.py`. Use `preview_cameras.py` to iterate.
  2. Split scenes into state vs. vision variants so training stays fast.
  3. Start on domain randomization: physics (mass, friction, actuator gains), observation (noise, latency), visual (lighting, textures, camera pose jitter).
  4. Set up a state → vision distillation pipeline so the policies we already have (PPO state) can be reused as teachers.

## 2026-04-09 — Atomic skills + repository reorganization

- **What:** Introduced an `atomic/` task family with five primitive skills — Grasp, Lift, Transport, Place, ReturnHome — each with its own gym registration, PPO config, custom MDP terms, and reset events. Existing multi-stage tasks (Reach, Lift, PickAndPlace, Stack) were grouped under `composite/`. Added a unified 24-dim observation space across atomic skills so a sequencer can switch policies mid-episode without reshaping.
- **Why:** Long-horizon tabletop tasks are easier to learn and evaluate when decomposed into reusable primitives. Splitting `atomic/` from `composite/` makes the building-block vs. end-to-end distinction explicit.
- **Also:** License migrated to MIT (Copyright (c) 2026 Jixin Yan); license headers updated across all `.py` files. New documentation: [`docs/how_to_add_an_atomic_skill.md`](how_to_add_an_atomic_skill.md). Updated [`docs/how_to_design_a_task.md`](how_to_design_a_task.md) for the new `composite/` path.
- **Next:** Train each atomic skill end-to-end and collect baseline learning curves; design a simple sequencer that chains Grasp → Lift → Transport → Place; revisit the hardcoded "already-grasped" reset poses with proper FK.

## 2026-04-08 — Initial setup

- **What:** Bootstrapped the repository with Isaac Lab–based SO-ARM101 task configurations (reach, lift, pick-and-place, stack), RSL-RL training/playback scripts, and CI (Pylint workflow, GitHub Actions).
- **Why:** Establish a minimal working baseline before layering safety evaluation on top.
- **Next:** Decompose monolithic tasks into atomic skills.
