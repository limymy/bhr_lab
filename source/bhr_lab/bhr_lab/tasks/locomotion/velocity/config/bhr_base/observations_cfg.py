from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, configclass

@configclass
class BaseObservationCfg(ObservationsCfg):
    '''Configuration for the base observation'''
    critic: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()

    def __post_init__(self):
        # super().__post_init__()
        self.policy.base_lin_vel = None  # type: ignore
        self.policy.height_scan = None  # type: ignore
