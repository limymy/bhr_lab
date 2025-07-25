import gymnasium as gym

from .noarm_env_cfg import (
    Bhr8Fc2NoArmFlatNSCEnvCfg,
    Bhr8Fc2NoArmFlatEnvCfg, 
    Bhr8Fc2NoArmFlatRandomEnvCfg,
    Bhr8Fc2NoArmRoughEnvCfg,
    Bhr8Fc2NoArmRoughRandomEnvCfg,
    Bhr8Fc2NoArmWarehouseEnvCfg,
)
from .agents.rsl_rl_cfg import (
    Bhr8Fc2BasePPORunnerCfg,
    Bhr8Fc2RecurrentPPORunnerCfg,
    Bhr8Fc2NoArmMirrorPPORunnerCfg,
)

##
# Register Gym environments.
##

gym.register(
    id="bhr8_fc2_noarm_flat_nsc_mirror",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmFlatNSCEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2NoArmMirrorPPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_flat_mirror",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmFlatEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2NoArmMirrorPPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_flat_random_mirror",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmFlatRandomEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2NoArmMirrorPPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_rough_random_mirror",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmRoughRandomEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2NoArmMirrorPPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_flat_nsc",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmFlatNSCEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2BasePPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmFlatEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2BasePPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_flat_random",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmFlatRandomEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2BasePPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_rough_random",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmRoughRandomEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2BasePPORunnerCfg,
    },
)

gym.register(
    id="bhr8_fc2_noarm_warehouse_mirror",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Bhr8Fc2NoArmWarehouseEnvCfg,
        "rsl_rl_cfg_entry_point": Bhr8Fc2NoArmMirrorPPORunnerCfg,
    },
)