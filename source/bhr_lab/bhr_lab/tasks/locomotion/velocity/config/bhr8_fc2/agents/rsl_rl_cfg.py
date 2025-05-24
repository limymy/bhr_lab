from bhr_lab.tasks.locomotion.velocity.config.bhr_base.agents.rsl_rl_cfg import (
    BasePPORunnerCfg, RecurrentPPORunnerCfg, MirrorPPORunnerCfg, configclass
)

@configclass
class Bhr8Fc2BasePPORunnerCfg(BasePPORunnerCfg):
    """Bhr8 Fc2 PPO actor critic config."""
    experiment_name = "bhr8_fc2_base"

@configclass
class Bhr8Fc2RecurrentPPORunnerCfg(RecurrentPPORunnerCfg):
    experiment_name = "bhr8_fc2_recurrent"

@configclass
class Bhr8Fc2NoArmMirrorPPORunnerCfg(MirrorPPORunnerCfg):
    experiment_name = "bhr8_fc2_noarm_mirror"

    def __post_init__(self):
        super().__post_init__()

        self.algorithm.symmetry_cfg.mirror_loss_coeff = 0.1
