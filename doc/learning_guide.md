# Unitree RL Mjlab — Complete Learning Guide

This guide explains **everything** about this repository: the system architecture, the physics/simulation layer, the RL task (MDP), the **reward engineering**, the **policy network and PPO mathematics**, motion imitation, and sim-to-real deployment.

---

## 1. What Is This Repo?

`unitree_rl_mjlab` trains legged-robot locomotion policies (Unitree Go2, G1, H1_2, A2, R1, …) with **reinforcement learning**, using:

- **MuJoCo** as the physics engine (GPU-accelerated via `mujoco_warp`)
- **mjlab** — an Isaac-Lab-style manager-based RL framework built on MuJoCo (this is the "glue" that defines environments, observations, rewards, commands, etc.)
- **rsl_rl** — the RL algorithm library (PPO), the same one used in Isaac Lab / legged_gym

The core workflow is:

```
Train  ──▶  Play  ──▶  Sim2Real
 (PPO in MuJoCo)   (replay + verify)   (deploy ONNX on real robot)
```

Two kinds of tasks exist:

| Task | Purpose | Config dir |
|------|---------|-----------|
| **Velocity tracking** | Robot follows commanded `vx, vy, ωz` (locomotion / walking / running) | `src/tasks/velocity/` |
| **Motion tracking** (BeyondMimic-style) | Robot imitates a reference humanoid motion (dance, etc.) | `src/tasks/tracking/` |

---

## 2. The Simulation & Control Loop (Physics Layer)

### 2.1 Physics timestep vs control frequency

From `velocity_env_cfg.py`:

```python
sim = SimulationCfg(mujoco=MujocoCfg(timestep=0.005, ...))
decimation = 4
```

- **Physics (MuJoCo) timestep**: `dt_phys = 0.005 s` (200 Hz)
- **Environment / policy step**: `dt_env = dt_phys × decimation = 0.005 × 4 = 0.02 s` (**50 Hz**)
- **Episode length**: `episode_length_s = 20.0` → `max_episode_length = ceil(20 / 0.02) = 1000` env steps

The `step()` method in `mjlab/envs/manager_based_rl_env.py` does:

```python
self.action_manager.process_action(action)
for _ in range(self.cfg.decimation):
    self.action_manager.apply_action()   # set joint targets
    self.sim.step()                      # 1 MuJoCo substep
termination_manager.compute()
reward_manager.compute(dt=self.step_dt)
```

So the policy outputs **one action every 50 Hz**, and the same target is held constant for 4 physics substeps.

> **Key insight for reward engineering:** all rewards are scaled by `dt = step_dt` (`scale_by_dt=True`). This normalizes cumulative reward across different control frequencies so hyperparameters are roughly portable between setups.

### 2.2 Action space — joint position control

The action term is `JointPositionActionCfg` with `use_default_offset=True`. The mapping is:

```
q_target = a * scale + q_default - encoder_bias
```

- `a` = raw policy output (dimension = number of actuated joints, 29 for G1, 12 for Go2)
- `scale` = per-joint action scale
- `q_default` = default joint pose (home keyframe)
- `encoder_bias` = domain-randomized sensor bias

MuJoCo then uses a **built-in PD (position) actuator** to drive the joint toward `q_target`:

$$
\tau = k_p (q_{target} - q) - k_d \dot{q}
$$

where `k_p` (stiffness) and `k_d` (damping) come from the robot actuator config (Section 3).

**Why scale by `0.25·e_max/k_p`?** In `g1_constants.py`:

```python
G1_ACTION_SCALE[n] = 0.25 * e / s   # for each joint
```

This means `a·scale·k_p = a·0.25·e_max`, so a unit action produces **25% of the actuator's max effort** as torque — a principled way to keep all joints in a comparable, torque-safe range.

---

## 3. Robot Models & Actuator Modeling

Each robot lives in `src/assets/robots/<robot>/` as an **MJCF XML** plus a `*_constants.py` that builds actuator configs.

### 3.1 G1 humanoid (29 DoF) — `unitree_g1/g1_constants.py`

G1 uses 4 motor families. Key engineering here is computing **reflected rotor inertia** through two-stage planetary gearboxes:

```
ARMATURE = reflected_inertia_from_two_stage_planetary(ROTOR_INERTIAS, GEARS)
```

The actuator is then tuned to a **natural frequency** so PD gains are physically meaningful:

```
NATURAL_FREQ  = 10 Hz * 2π        # target closed-loop natural frequency
DAMPING_RATIO = 2.0               # critically-damped-ish
STIFFNESS = armature * NATURAL_FREQ²
DAMPING   = 2 * DAMPING_RATIO * armature * NATURAL_FREQ
```

So `k_p` is chosen so the motor+link system has a ~10 Hz response; this is a standard trick to make the PD gains robust and well-scaled across joints.

**Collision model** (`CollisionCfg`): foot geoms get `condim=3` (full 3D friction cone), all other geoms `condim=1` (self-collision "touching" only) — this gives natural self-collision behavior without unstable stacking.

### 3.2 Go2 quadruped (12 DoF) — `unitree_go2/go2_constants.py`

Go2 uses simpler fixed PD gains (k_p = 20/20/40, k_d = 1/1/2) with `armature` for rotor inertia. Different robots → different actuator tuning, but the RL interface is identical.

---

## 4. The RL Task (MDP) for Velocity Tracking

Everything is manager-based. The environment = a collection of managers:

```
ObservationManager   — builds actor/critic observations
ActionManager        — applies policy actions
CommandManager       — generates velocity commands
EventManager         — domain randomization & resets
RewardManager        — computes weighted reward sum
TerminationManager   — decides episode resets
CurriculumManager    — adaptive difficulty
```

### 4.1 Observations (the "sensor vector" fed to the policy)

**Asymmetric actor–critic** design (standard in legged locomotion):

- **Actor obs** (what the policy sees — must be available on the real robot):
  - `base_ang_vel` (3) — gyro, noisy
  - `projected_gravity` (3) — tilt from IMU, noisy
  - `command` (3) — vx, vy, ωz target
  - `phase` (2) — `[sin(2πφ), cos(2πφ)]` gait clock (see §6.3)
  - `joint_pos_rel` (N) — q − q_default, noisy
  - `joint_vel_rel` (N) — noisy
  - `actions` (N) — last action (history/recurrence)
  - `height_scan` (16×9=144 for rough terrain) — raycast terrain heights
- **Critic obs** (only used in training to estimate value; extra privileged info):
  - everything the actor sees **plus**:
  - `base_lin_vel` (3) — true linear velocity (a velocity estimator would be needed on hardware)
  - `height_scan` without noise
  - `foot_height` (N), `foot_air_time` (N), `foot_contact` (N), `foot_contact_forces` (3N)

Corruption: `enable_corruption=True` for actor — uniform sensor noise is added during training (noise configs per term). In `play` mode corruption is turned off.

> **Why asymmetric?** The critic only exists during training, so it can use "cheated" ground-truth information. The actor stays sim-to-real friendly (only IMU + joint encoders + commands + optional height scan).

### 4.2 Commands — the velocity generator

`UniformVelocityCommand` (in `mdp/velocity_command.py`) samples:

```
vx ∈ [-1.0, 2.0],  vy ∈ [-1.0, 1.0],  ωz ∈ [-1.0, 1.0]
```

- Resampled every `resampling_time_range = (3.0, 8.0)` seconds (random).
- `rel_standing_envs = 0.05` → 5% of envs get **zero** command (stand-still training).
- `heading_command=True`: for `rel_heading_envs` fraction, instead of directly commanding ωz, it commands a **heading target** and computes $\omega_z = K \cdot \text{wrap}(\theta_{target} - \theta_{robot})$ with $K=0.5$ — this teaches the robot to turn toward an absolute heading (more natural steering).
- `init_velocity_prob`: some resets give the robot the commanded velocity (prevents "starting from standstill" bias).

### 4.3 Events — Domain Randomization (crucial for Sim2Real)

- `push_robot` (every 5–6 s): random velocity/angular impulse → robustness to disturbances.
- `foot_friction` (startup): per-robot random friction in `[0.3, 1.6]`.
- `encoder_bias` (startup): random joint-position bias ±0.015 rad.
- `base_com` (startup): random COM offset ±0.05 m (mass modeling error).
- `reset_base` / `reset_robot_joints`: randomized episode starts.

This randomization is what makes policies transfer to real hardware.

### 4.4 Terminations

- `time_out`: episode length reached (this is a *truncation*, not a death → PPO bootstraps value here, see §7.5).
- `fell_over`: base tilt > 70° → episode ends, big penalty.

For Go2 rough there's also `illegal_contact` (body touching ground = death).

---

## 5. Reward Engineering — the Heart of It

This is the most important section. All rewards are defined in:

- `src/tasks/velocity/mdp/rewards.py` (custom)
- `mjlab/envs/mdp/rewards.py` (built-in helpers)

### 5.1 The universal reward kernel

Almost every "tracking" reward uses the **exponential error kernel**:

$$
r = \exp\!\left(-\frac{e^2}{\sigma^2}\right) \in (0, 1]
$$

- Perfect tracking ($e=0$) → reward 1
- Error of one σ → reward $e^{-1} \approx 0.37$
- $\sigma$ controls how "forgiving" the reward is.

This is smoother and better-behaved than a squared error penalty because it is bounded and has a clean gradient that saturates — the agent is never "punished forever" for being slightly off.

### 5.2 Task rewards (what the robot must do)

**`track_linear_velocity`** (weight +1.0, σ=√0.25=0.5):

$$
e_{xy} = \|v_{cmd,xy} - v_{act,xy}\|^2,\qquad
r = \exp\!\left(-\frac{e_{xy} + 2 e_z}{\sigma^2}\right)
$$

Note the **z-velocity term is doubled** — bouncing up/down is penalized twice as hard as horizontal error, to prevent hopping.

**`track_angular_velocity`** (weight +1.0, σ=√0.5):

$$
r = \exp\!\left(-\frac{(\omega_{cmd,z} - \omega_{act,z})^2 + 0.05\|\omega_{act,xy}\|^2}{\sigma^2}\right)
$$

Pitching/rolling while turning is penalized at 5%.

**`body_orientation_l2`** (weight −1.0):

$$
r = -\sum (\hat{g}_{b,xy})^2
$$

where $\hat{g}_b = R_b^T g_w$ is **projected gravity** (Section 9). This keeps the torso upright — the single most important "stay balanced" reward.

**`pose`** = `variable_posture` (weight +1.0) — speed-adaptive posture. Three regimes based on commanded speed $s = \|v_{cmd,xy}\| + |\omega_{cmd,z}|$:

| Regime | Condition | Std (how much deviation allowed) |
|--------|-----------|---------------------------------|
| standing | $s < 0.1$ | tight (e.g. 0.05) |
| walking | $0.1 \le s < 1.5$ | moderate (hip/knee 0.5, others 0.1–0.25) |
| running | $s \ge 1.5$ | loose |

$$
r = \exp\!\left(-\frac{1}{J}\sum_j \frac{(q_j - q_{j,default})^2}{\sigma_j^2}\right)
$$

The per-joint σ map (in `g1/env_cfgs.py`) is hand-tuned:
- **knees / hip_pitch → 0.5** (loose): they must move a lot to stride.
- **ankle_roll → 0.1** (tight): critical for lateral balance.
- **waist → 0.1**: keeps torso stable.
- **arms → 0.1–0.25**: natural arm swing without flailing.

This reward alone defines the *gait style* — tight σ = stiff, precise motion; loose σ = relaxed natural motion.

### 5.3 Style / gait rewards

**`foot_gait`** (weight +0.5, `period=0.6`, `offset=[0.0, 0.5]` for G1; `[0, 0.5, 0.5, 0]` for Go2):

Imposes a **phase-based gait pattern** (trot for Go2, alternating biped gait for G1). The gait clock is:

$$
\phi_i(t) = \left(\frac{t}{T_{gait}} + \text{offset}_i\right) \bmod 1,\qquad
T_{gait} = 0.6\text{ s}
$$

Foot $i$ "should be in stance" when $\phi_i < \text{threshold}=0.56$:

$$
r = \frac{1}{N_{feet}}\sum_i \mathbb{1}[\text{in\_stance}(\phi_i) == \text{in\_contact}_i]
$$

So the reward is the **fraction of feet that match the desired phase**. Only active when commanded speed > 0.1 (not while standing).

**`foot_clearance`** (weight −1.0, `target_height=0.10`):

$$
r = -\sum_i |z_{foot,i} - h_{target}| \cdot \|v_{foot,xy,i}\|
$$

Weighted by foot horizontal velocity → only penalizes clearance **during the swing** (when the foot is moving fast). This produces high-stepping, obstacle-clearing gaits.

**`foot_slip`** (weight −0.25):

$$
r = -\sum_i \|v_{foot,xy,i}\|^2 \cdot \mathbb{1}[\text{foot in contact}]
$$

**`soft_landing`** (weight −1e-3):

$$
r = -\sum_i F_i \cdot \mathbb{1}[\text{first contact at step}]
$$

Penalizes impact force at the instant of touchdown → quieter, gentler footfalls (protects hardware).

**`angular_momentum`** (weight −0.025):

$$
r = -\|L_{total}\|^2
$$

Whole-body angular momentum; encourages natural arm swing (arms swing to cancel leg angular momentum). This is why the G1 learns to pump its arms while walking.

### 5.4 Regularization rewards (smoothness / safety)

- **`action_rate_l2`** (weight −0.05): $-\|a_t - a_{t-1}\|^2$ → smooth commands, no jitter.
- **`joint_acc_l2`** (weight −2.5e-7): $-\sum \ddot{q}_j^2$ → no violent accelerations.
- **`joint_pos_limits`** (weight −10.0): hinge penalty beyond **soft** limits:

$$
r = -\sum_j \left[\max(0, q_j - q_{j,max}^{soft}) + \max(0, q_{j,min}^{soft} - q_j)\right]
$$

- **`is_terminated`** (weight −200.0): big penalty for falling.
- **`self_collisions`** (G1, weight −1.0): counts substeps where any contact force > 10 N.
- **`stand_still`** (weight −1.0): only active when command < 0.1 — penalizes joint deviation from default while standing (a "stand perfectly still" term).

### 5.5 Complete G1 weight table (from `velocity_env_cfg.py`)

| Term | Weight | Type |
|------|--------|------|
| track_linear_velocity | +1.0 | task |
| track_angular_velocity | +1.0 | task |
| pose (variable_posture) | +1.0 | task/style |
| body_orientation_l2 | −1.0 | balance |
| body_ang_vel | −0.05 | balance |
| angular_momentum | −0.025 | style |
| foot_gait | +0.5 | gait |
| foot_clearance | −1.0 | gait |
| foot_slip | −0.25 | gait |
| soft_landing | −1e-3 | hardware |
| action_rate_l2 | −0.05 | smoothness |
| joint_acc_l2 | −2.5e-7 | smoothness |
| joint_pos_limits | −10.0 | safety |
| stand_still | −1.0 | standing |
| self_collisions | −1.0 | safety |
| is_terminated | −200.0 | terminal |

### 5.6 How rewards are aggregated

`RewardManager.compute(dt)`:

```python
value = func(env, **params) * weight * dt
```

Total reward per step:

$$
R_t = \left(\sum_i w_i \, r_i\right) \cdot \Delta t
$$

NaN/Inf from corrupted physics are zeroed (`torch.nan_to_num`). Each term's cumulative sum is logged as `Episode_Reward/<name>` **divided by episode length in seconds** → a per-second average, directly comparable across runs.

---

## 6. Curriculum (Adaptive Difficulty)

Two curricula in `mdp/curriculums.py`:

1. **`terrain_levels_vel`** (rough terrains): if the robot walks farther than half the terrain size → move to a **harder terrain** (higher elevation steps); if it walks less than half of what it *should* have given the command → move to an easier one.

2. **`commands_vel`**: command ranges grow with training step:
   - stage 0: `vx ∈ [−0.5, 1.0]`, `vy ∈ [−0.5, 0.5]`, `ωz ∈ [−1, 1]`
   - stage 1 (after `5000×24 = 120k` steps): `vx ∈ [−1.0, 2.0]`, `vy ∈ [−1.0, 1.0]`

This is classic **curriculum RL**: start easy, gradually increase the difficulty so the agent never faces an impossible task early on.

---

## 7. The Policy Network & PPO Mathematics

### 7.1 Architecture

From `g1/rl_cfg.py`:

```python
actor  = MLPModel(hidden_dims=(512, 256, 128), activation="elu",
                  obs_normalization=True,
                  distribution_cfg={"class_name": "GaussianDistribution",
                                    "init_std": 1.0, "std_type": "scalar"})
critic = MLPModel(hidden_dims=(512, 256, 128), activation="elu",
                  obs_normalization=True)
```

**Actor** (policy): MLP `obs_dim → 512 → 256 → 128 → action_dim`, with ELU activations. The output layer feeds a **Gaussian distribution head**.

**Critic** (value function): same MLP shape, output `1` (scalar value).

> Total parameters: roughly `512·(obs_dim) + 512·256 + 256·128 + 128·action_dim` for the actor; same scale for critic. For G1 (obs_dim ≈ 90ish, action_dim=29) this is a few hundred thousand parameters — a deliberately compact MLP (no recurrence), runnable at 50 Hz on embedded hardware.

### 7.2 Observation normalization

`EmpiricalNormalization` (from rsl_rl) keeps **running mean/var**:

$$
\hat{x} = \frac{x - \mu_{running}}{\sigma_{running} + \epsilon}
$$

Updated incrementally each rollout step with an exponential-average update. Crucially, the normalizer is exported with the ONNX model, so deployment normalizes identically. This is essential because raw obs (e.g. joint positions, velocities) have wildly different scales.

### 7.3 The Gaussian policy (stochastic actions)

The policy is a **diagonal Gaussian**:

$$
\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s),\, \sigma^2 I)
$$

- $\mu_\theta(s)$ = MLP output (the "mean action")
- $\sigma$ = **learnable parameter**, `std_type="scalar"`, `init_std=1.0` (one scalar std shared across all action dims, initialized at 1.0)

During training we **sample** $a \sim \mathcal{N}(\mu,\sigma)$ (exploration). At deployment we use the **mean** (deterministic) $\mu_\theta(s)$.

The action log-probability for a sample (needed by PPO):

$$
\log \pi_\theta(a|s) = -\frac{1}{2}\left(\frac{a-\mu}{\sigma}\right)^2 - \frac{1}{2}\log(2\pi\sigma^2)
$$

Entropy (exploration bonus):

$$
H(\pi) = \frac{1}{2}\log(2\pi e \sigma^2) \quad\text{(per dimension, summed)}
$$

As training progresses, $\sigma$ shrinks → the policy becomes more confident/deterministic. `entropy_coef=0.01` gently encourages exploration.

### 7.4 PPO — the algorithm

Config (`g1/rl_cfg.py`):

```python
clip_param=0.2,  gamma=0.99,  lam=0.95,  entropy_coef=0.01,
num_learning_epochs=5,  num_mini_batches=4,  learning_rate=1e-3,
schedule="adaptive",  desired_kl=0.01,  max_grad_norm=1.0,
value_loss_coef=1.0,  use_clipped_value_loss=True
```

#### (a) Rollout + GAE (advantage estimation)

Each iteration collects `num_steps_per_env = 24` steps from each of `num_envs` parallel envs. Then **Generalized Advantage Estimation** computes advantages:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

$$
A_t^{GAE} = \delta_t + (\gamma\lambda)\,\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
$$

with $\gamma = 0.99$, $\lambda = 0.95$. Returns: $G_t = A_t + V(s_t)$.

**Time-out bootstrapping**: when an episode ends by *timeout* (not death), the reward gets $+\gamma V(s_{t+1})$ added so the value function learns to continue beyond the truncation — standard infinite-horizon handling.

#### (b) The clipped surrogate objective

For each mini-batch, compute the **importance ratio** of new vs old policy:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = e^{\log\pi_\theta - \log\pi_{old}}
$$

The PPO loss (to **maximize**):

$$
L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\right)\right]
$$

- If the new policy is **too different** (ratio outside $[0.8, 1.2]$), the clip cuts the gradient → prevents destructive policy updates.

#### (c) Value loss (clipped)

$$
L^{V} = \mathbb{E}_t\!\left[\max\left((V_\theta - G_t)^2,\; (V_\theta^{clip} - G_t)^2\right)\right],
\qquad V_\theta^{clip} = V_{old} + \text{clip}(V_\theta - V_{old}, -\epsilon, \epsilon)
$$

Clipped value loss makes the critic update stable too.

#### (d) Total loss

$$
L = \underbrace{-L^{CLIP}}_{\text{policy}} + \underbrace{c_v L^V}_{\text{value}} - \underbrace{c_e \mathbb{E}[H]}_{\text{entropy bonus}}
$$

with $c_v = 1.0$, $c_e = 0.01$. Gradients are clipped to norm 1.0 (`max_grad_norm`).

#### (e) Adaptive learning rate (KL schedule)

After each update, measure $\text{KL}(\pi_{old} \| \pi_{new})$ (closed form for Gaussians). If `KL > 2·desired_kl` → halve the LR; if `KL < desired_kl/2` → multiply by 1.5. This keeps updates in a safe "trust region" without the complexity of TRPO.

### 7.5 The training loop (from `OnPolicyRunner.learn`)

```
for iteration:
    # rollout (no grad)
    for step in range(num_steps_per_env=24):
        actions = alg.act(obs)          # sample from Gaussian policy
        obs, rew, dones, extras = env.step(actions)
        alg.process_env_step(...)       # store transitions, update normalizers
    alg.compute_returns(obs)            # GAE
    alg.update()                        # 5 epochs × 4 mini-batches of PPO
```

`num_steps_per_env × num_envs = 24 × 4096 = 98,304` transitions per iteration, and `num_mini_batches=4` → mini-batch size 24,576.

---

## 8. Motion Imitation Task (BeyondMimic-style)

`src/tasks/tracking/` re-implements **BeyondMimic / whole_body_tracking**:

- A **motion file** (`.npz` converted from CSV via `scripts/csv_to_npz.py`) contains per-timestep: joint pos/vel, body positions/orientations/velocities for key bodies.
- `MotionCommand` plays the reference motion like a command: the robot must match the reference **anchor body** (torso) position/orientation, and **relative body poses** for 14 bodies.

### 8.1 Rewards (from `tracking/mdp/rewards.py`)

All are exponential kernels:

| Term | Weight | Tracks |
|------|--------|--------|
| motion_global_root_pos | 0.5 | anchor position (σ=0.3) |
| motion_global_root_ori | 0.5 | anchor orientation (σ=0.4) |
| motion_body_pos | 1.0 | relative body positions (σ=0.3) |
| motion_body_ori | 1.0 | relative body orientations (σ=0.4) |
| motion_body_lin_vel | 1.0 | body linear velocities (σ=1.0) |
| motion_body_ang_vel | 1.0 | body angular velocities (σ=3.14) |
| action_rate_l2 | −0.1 | smoothness |
| joint_limit | −10.0 | safety |
| self_collisions | −10.0 | safety |

Tracking in **relative** space (relative to the anchor) makes the robot track the *shape* of the motion, not its absolute position — key for whole-body imitation.

### 8.2 Adaptive phase sampling

`MotionCommand` also does **adaptive sampling**: if the robot fails near a certain part of the motion (tracked by `bin_failed_count`), that part is sampled more often during training (via a softmax over failure-weighted bins). This focuses training on the hardest segments — the trick that makes the whole-body dance learnable.

---

## 9. Key Mathematics Reference

### 9.1 Projected gravity (the "tilt sensor")

If $R_b$ is the body→world rotation matrix and $g_w = [0,0,-9.81]$:

$$
\hat{g}_b = R_b^{T} g_w
$$

- Upright robot → $\hat{g}_b = [0,0,-9.81]$, i.e. $\|\hat{g}_{b,xy}\| = 0$
- Tilted by angle $\theta$ → $\|\hat{g}_{b,xy}\| = 9.81\sin\theta$

So "flat orientation" reward $-\|\hat{g}_{b,xy}\|^2$ directly penalizes tilt angle (squared). This is exactly what an IMU accelerometer measures at rest, so it's sim-to-real safe.

### 9.2 Quaternions & heading

- `heading_w` is derived from the yaw part of the root quaternion.
- `wrap_to_pi` maps an angle into $(-\pi, \pi]$ — used for heading error so a 359° error becomes −1°.
- `quat_apply(q, v)` rotates vector $v$ by quaternion $q$ (used to set initial velocity in body frame → world frame).
- Quaternion error magnitude (used in tracking): angle between two orientations.

### 9.3 GAE — why λ matters

- $\lambda = 0$: advantage = 1-step TD error (high bias, low variance)
- $\lambda = 1$: advantage = Monte-Carlo return (low bias, high variance)
- $\lambda = 0.95$: a good middle ground. This is the standard trick that makes PPO sample-efficient.

### 9.4 Why exponential rewards work

Compare squared-error penalty $-e^2$ vs exponential $\exp(-e^2/\sigma^2)$:

| | squared | exponential |
|---|---|---|
| range | $(-\infty, 0]$ | $(0, 1]$ |
| gradient far from target | grows linearly (harsh) | → 0 (gentle) |
| sensitivity | uniform | tunable via σ |

The exponential gives the agent a "shaped, bounded" signal and makes reward magnitudes comparable across terms, which hugely simplifies weight tuning.

### 9.5 Reward scaling by dt

Because $R_t = \text{(weights·terms)}·\Delta t$, doubling the control frequency would halve per-step reward values if not normalized — the dt scaling keeps cumulative returns comparable, so PPO hyperparameters transfer.

---

## 10. Training → Play → Deployment

### 10.1 Train (`scripts/train.py`)

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

- Task registry loads env + RL configs (mjlab's registry, populated by `src/tasks`).
- Multi-GPU via `torchrunx` + NCCL (rsl_rl supports distributed PPO).
- Checkpoints saved to `logs/rsl_rl/<experiment>/<timestamp>/model_<iter>.pt`
- **ONNX export on save**: `policy.onnx` (deterministic mean policy) + `policy.onnx.data` metadata (joint names, stiffness/damping, default pose, action scale, observation names) — see `VelocityOnPolicyRunner.save`.

### 10.2 Play (`scripts/play.py`)

Loads a checkpoint, sets `play=True` (no noise, no pushes, infinite episode, fixed terrain), runs the deterministic policy in the viewer (native MuJoCo or Viser).

### 10.3 Sim-to-real (`deploy/`)

The C++ deployment stack:

1. **`policy.onnx`** is executed by **ONNX Runtime** (`OrtRunner`).
2. `unitree_articulation.h` reads the real robot's **IMU** (quaternion, gyro) and **joint encoders** (q, dq) via **unitree_sdk2 / DDS**.
3. `observations.h` reconstructs the exact same observation vector as training (base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action, velocity_commands from joystick, gait_phase).
4. `State_RLBase.cpp` runs the policy thread at 50 Hz and writes joint targets to `motor_cmd`.
5. The **FSM** (`config.yaml`) handles safety: `Passive → FixStand → Velocity/Mimic`, with gain (kp/kd) interpolation during transitions, and fall-back to Passive if the robot tilts too far.

The metadata embedded in the ONNX (`joint_stiffness`, `joint_damping`, `default_joint_pos`, `action_scale`) lets the C++ runtime set the real PD gains to **exactly match** the sim actuator tuning.

### 10.4 Sim-to-sim (`simulate/`)

Before real hardware, `unitree_mujoco` (in `simulate/`) runs the *same* deployment binary against a MuJoCo sim of the real robot over `lo` network — validating the whole deployment stack without hardware.

---

## 11. Where Everything Lives (Cheat Sheet)

```
scripts/train.py, play.py      — entry points
src/tasks/velocity/            — velocity tracking task
  velocity_env_cfg.py          — base env (obs/actions/rewards/commands/curriculum)
  mdp/rewards.py               — reward functions
  mdp/observations.py          — obs functions
  mdp/velocity_command.py      — command generator
  mdp/curriculums.py           — curriculum
  config/<robot>/env_cfgs.py   — per-robot tuning (rewards weights, σ maps)
  config/<robot>/rl_cfg.py     — PPO hyperparameters
src/tasks/tracking/            — motion imitation (BeyondMimic-style)
src/assets/robots/             — MJCF models + actuator constants
deploy/                        — C++ sim-to-real stack (ONNX Runtime + DDS)
simulate/                      — unitree_mujoco sim-to-sim validation
logs/rsl_rl/                   — checkpoints + exported ONNX
```

---

## 12. How to Actually Read This Repo (Suggested Order)

1. **`src/tasks/velocity/config/g1/env_cfgs.py`** — see how a task is assembled (per-robot overrides).
2. **`src/tasks/velocity/velocity_env_cfg.py`** — the base env: observations, rewards, commands, curriculum.
3. **`src/tasks/velocity/mdp/rewards.py`** — the reward math (§5 above).
4. **`rsl_rl/algorithms/ppo.py`** — the PPO implementation (§7).
5. **`rsl_rl/models/mlp_model.py`** — the actor/critic network.
6. **`deploy/robots/g1/src/State_RLBase.cpp`** — how a policy runs on hardware.
