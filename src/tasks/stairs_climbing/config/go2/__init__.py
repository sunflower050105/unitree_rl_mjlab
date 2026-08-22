from mjlab.tasks.registry import register_mjlab_task
from src.tasks.stairs_climbing.rl import StairsClimbingOnPolicyRunner

from .env_cfgs import unitree_go2_stairs_climbing_env_cfg
from .rl_cfg import unitree_go2_stairs_climbing_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-Go2-Stairs-Climbing",
  env_cfg=unitree_go2_stairs_climbing_env_cfg(),
  play_env_cfg=unitree_go2_stairs_climbing_env_cfg(play=True),
  rl_cfg=unitree_go2_stairs_climbing_ppo_runner_cfg(),
  runner_cls=StairsClimbingOnPolicyRunner,
)
