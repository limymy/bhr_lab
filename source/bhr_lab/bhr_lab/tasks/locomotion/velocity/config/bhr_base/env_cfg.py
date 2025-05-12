from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg, configclass
)
from .event_cfg import BaseEventCfg
from .observations_cfg import BaseObservationCfg
from .reward_cfg import BaseRewardsCfg


@configclass
class BaseRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    '''Configuration for the BHR base rough environment'''
    rewards: BaseRewardsCfg = BaseRewardsCfg()
    events: BaseEventCfg = BaseEventCfg()
    observations: BaseObservationCfg = BaseObservationCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
