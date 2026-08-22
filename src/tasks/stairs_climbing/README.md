# Stair Climbing Task (`stairs_climbing`)

A Unitree Go2 stair-climbing locomotion task built on top of the velocity-tracking
baseline (`src/tasks/velocity`). It reuses the same PPO setup and only changes
the **terrain**, the **command distribution**, and the **reward function**.

Registered task ID: **`Unitree-Go2-Stairs-Climbing`**

---

## Quick start

```bash
# Activate the project environment
conda activate unitree_rl_mjlab

# List all registered tasks (the new one should appear)
python scripts/list_envs.py

# Train (PPO)
python scripts/train.py Unitree-Go2-Stairs-Climbing

# Preview the stairs terrain without training (dummy agents)
python scripts/play.py Unitree-Go2-Stairs-Climbing --agent zero

# Play a trained checkpoint
python scripts/play.py Unitree-Go2-Stairs-Climbing \
  --agent trained \
  --checkpoint-file logs/rsl_rl/go2_stairs_climbing/<run>/model_500.pt

# Interactive terrain gallery (all built-in terrains incl. stairs)
python scripts/visualize_terrain.py
```

Override config from the CLI (tyro, kebab-case):

```bash
python scripts/train.py Unitree-Go2-Stairs-Climbing \
  --env.scene.num-envs 4096 \
  --env.rewards.stair-incline.weight -2.0 \
  --env.scene.terrain.terrain-generator.sub-terrains.open-stairs.step-height-range 0.05 0.2 \
  --agent.experiment-name go2_stairs_climbing_v2
```

---

## Folder structure

```
src/tasks/stairs_climbing/
├── __init__.py                      # package docstring
├── stairs_climbing_env_cfg.py       # base env factory + stairs terrain config
├── mdp/                             # task MDP functions (rewards, etc.)
│   ├── __init__.py                  # re-exports everything
│   ├── rewards.py                   # reward functions (added: stair_incline, forward_progress)
│   ├── observations.py              # observations
│   ├── terminations.py              # terminations
│   ├── curriculums.py               # curricula
│   └── velocity_command.py          # velocity command term
├── config/
│   └── go2/
│       ├── __init__.py              # registers the task
│       ├── env_cfgs.py              # Go2-specific overrides
│       └── rl_cfg.py                # PPO runner config
└── rl/
    ├── __init__.py
    └── runner.py                    # StairsClimbingOnPolicyRunner (ONNX export)
```

---

## What changed vs. the velocity baseline

| Area | velocity | stairs_climbing |
|---|---|---|
| Task ID | `Unitree-Go2-Rough` / `Flat` | `Unitree-Go2-Stairs-Climbing` |
| Env factory | `make_velocity_env_cfg()` | `make_stairs_climbing_env_cfg()` |
| Terrain | `ROUGH_TERRAINS_CFG` (mixed) | `BoxOpenStairsTerrainCfg(inverted=True)` — bowl staircase |
| Commands | `lin_vel_x=(-1,2)`, `lin_vel_y=(-1,1)`, `ang_vel_z=(-1,1)` | `lin_vel_x=(0,0.8)`, `lin_vel_y=(-0.2,0.2)`, `ang_vel_z=(-0.3,0.3)` (forward-focused) |
| Rewards | velocity + gait set | same set **+ `stair_incline` + `forward_progress`**; `body_orientation_l2` relaxed to `-0.5` |
| Runner class | `VelocityOnPolicyRunner` | `StairsClimbingOnPolicyRunner` |
| Experiment name | `go2_velocity` | `go2_stairs_climbing` |
| Other robots | a2/g1/h1_2/h2/r1/as2 configs | removed (Go2 only) |

> **Why `inverted=True`?** `BoxOpenStairsTerrainCfg` builds *concentric* steps.
> `inverted=True` creates a **bowl**: the robot spawns on the flat platform at
> the bottom and must climb **up** the steps to the flat border at ground level.
> `inverted=False` would spawn at the top (a pyramid) — that's a *descend* task.

---

## Terrain

Defined in `make_stairs_climbing_terrain_cfg()` inside `stairs_climbing_env_cfg.py`:

- Grid: `size=(8,8)`, `num_rows=10`, `num_cols=10`, `border_width=10`
- Sub-terrain: `BoxOpenStairsTerrainCfg`
  - `step_height_range=(0.05, 0.15)` — step height grows with difficulty (curriculum rows)
  - `step_width_range=(0.5, 0.8)` — step width shrinks with difficulty
  - `platform_width=3.0`, `border_width=1.0`
  - `inverted=True` — climb-up bowl

Built-in alternatives (from `mjlab.terrains`):

- `BoxRandomStairsTerrainCfg` — random step heights
- `BoxPyramidStairsTerrainCfg` / `BoxInvertedPyramidStairsTerrainCfg` — solid pyramid/bowl
- A custom straight staircase: subclass `SubTerrainCfg` (in `mjlab.terrains.terrain_generator`)
  and implement `function(difficulty, spec, rng) -> TerrainOutput`.

### ⚠️ Spawn-origin note
The reset event places the robot at the terrain's `origin`. `BoxOpenStairsTerrainCfg`
returns the platform origin, so with `inverted=True` the robot spawns at the bottom
(bowl) — correct for climbing up. If you switch to a terrain whose origin is at the
top (e.g. `inverted=False`), the robot will spawn on top of the stairs. For custom
terrains you can also use `FlatPatchSamplingCfg` (see `SubTerrainCfg`) to designate
flat spawn patches at the base.

---

## Rewards

All reward functions live in `mdp/rewards.py`. Weights are in
`stairs_climbing_env_cfg.py` (base) and `config/go2/env_cfgs.py` (Go2-specific).

| Reward | Weight | Function | Purpose |
|---|---|---|---|
| `track_linear_velocity` | 1.0 | `track_linear_velocity` | **forward velocity** tracking |
| `track_angular_velocity` | 1.0 | `track_angular_velocity` | yaw tracking |
| `body_orientation_l2` | -0.5 | `body_orientation_l2` | keep upright (relaxed so pitch allowed on stairs) |
| `pose` | 1.0 | `variable_posture` | keep default pose per speed regime |
| `body_ang_vel` | -0.05 | `body_angular_velocity_penalty` | smooth base rotation |
| `angular_momentum` | -0.025 | `angular_momentum_penalty` | natural arm/leg swing |
| `is_terminated` | -200.0 | `is_terminated` | **fall penalty** (via termination) |
| `joint_acc_l2` | -2.5e-7 | `joint_acc_l2` | smooth joints |
| `joint_pos_limits` | -10.0 | `joint_pos_limits` | joint limit avoidance |
| `action_rate_l2` | -0.05 | `action_rate_l2` | smooth actions |
| `foot_gait` | 0.5 | `feet_gait` | **gait reward** (phase-locked trot) |
| `foot_clearance` | -1.0 | `feet_clearance` | step clearance (0.12 m target) |
| `foot_slip` | -0.25 | `feet_slip` | anti-slip (important on stairs) |
| `soft_landing` | -1e-3 | `soft_landing` | gentle footfalls |
| `stand_still` | -1.0 | `stand_still` | posture when commanded to stop |
| `stair_incline` | -1.0 | `stair_incline` | **NEW** — keep base pitch within ±10° of level (pitch control) |
| `forward_progress` | -1.0 | `forward_progress` | **NEW** — anti-stall: penalize not reaching commanded forward speed |

### New reward functions (in `mdp/rewards.py`)

- **`stair_incline`** — computes base pitch from projected gravity
  (`pitch = atan2(-g_x, -g_z)`), then penalizes the squared pitch error *outside*
  a tolerance band `[target - tol, target + tol]`. Parameters:
  `target_angle_deg=0.0`, `tolerance_deg=10.0`. This rewards the robot for not
  pitching over on the stairs while tolerating the small pitch that climbing needs.
- **`forward_progress`** — when a forward command is active (`lin_vel_x >
  command_threshold`), penalizes `(cmd_vx - actual_vx)²`. This is the "penalty for
  standing still" (anti-stall): it stops the robot from freezing instead of climbing.

### Tuning tips
- **Robot not climbing / falling back**: raise `forward_progress` weight or lower
  `stair_incline` tolerance.
- **Too stiff / not pitching on steps**: increase `tolerance_deg`, or lower
  `body_orientation_l2` weight further.
- **Feet hitting step edges**: raise `foot_clearance` `target_height` or weight.
- **Slipping off steps**: raise `foot_slip` weight or increase foot friction range
  in the `foot_friction` event.

---

## Commands & curriculum

- **Commands** (`UniformVelocityCommandCfg`): `lin_vel_x=(0.0, 0.8)`,
  `lin_vel_y=(-0.2, 0.2)`, `ang_vel_z=(-0.3, 0.3)`, `rel_standing_envs=0.05`,
  resample every 3–8 s.
- **Curriculum**: `terrain_levels` uses the standard distance-based
  `terrain_levels_vel` (walking ~4 m promotes to harder terrain). `command_vel`
  starts at 0.5 m/s and ramps to 0.8 m/s after 5000 iterations.
- Optional: replace with a **height-based** curriculum (promote when the robot has
  climbed a certain height) by adding a `terrain_levels_stairs` function to
  `mdp/curriculums.py` and referencing it in `curriculum`.

---

## PPO / runner

- `config/go2/rl_cfg.py` → `unitree_go2_stairs_climbing_ppo_runner_cfg()`
  (`experiment_name="go2_stairs_climbing"`, MLP 512/256/128, standard RSL-RL PPO).
- `rl/runner.py` → `StairsClimbingOnPolicyRunner` (exports `policy.onnx` on save,
  same as the velocity runner).

---

## Deployment (after training)

The training env builds the MuJoCo scene procedurally. The **deployment sim** is a
separate C++ app in `simulate/` that loads a static XML scene, e.g.
`src/assets/robots/unitree_go2/xmls/scene_go2.xml` (configured in
`simulate/config.yaml`). To show stairs there, add stair geoms to that XML — this is
a separate step from training and is not handled by this task folder.

---

## Troubleshooting

- **"Task '...' is already registered"**: you likely re-registered the same task ID
  in two config packages. Task IDs must be unique across `src/tasks`.
- **Robot spawns on top of the stairs**: the terrain's `origin` is at the top; use
  `inverted=True` (bowl) or adjust the origin/flat patches in a custom terrain.
- **Policy can't "see" the stairs**: keep the `terrain_scan` raycast sensor and the
  `height_scan` observation. For taller steps, raise `max_distance` (5.0) in
  `stairs_climbing_env_cfg.py`.
- **Stepping is jerky / CCD issues**: Go2 config sets `ccd_iterations=500` and
  `contact_sensor_maxmatch=500` — keep these for stairs.
