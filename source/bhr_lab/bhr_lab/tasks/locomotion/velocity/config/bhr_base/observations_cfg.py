from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, configclass, ObsTerm

@configclass
class BaseObservationCfg(ObservationsCfg):
    '''Configuration for the base observation'''
    critic: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()

    def __post_init__(self):
        # super().__post_init__()
        self.policy.base_lin_vel = None  # type: ignore
        self.policy.height_scan = None  # type: ignore

        for value in self.policy.__dict__.values():
            if isinstance(value, ObsTerm):
                value.clip = (-100.0, 100.0)
        
        for value in self.critic.__dict__.values():
            if value is not self.critic.height_scan and isinstance(value, ObsTerm):
                value.clip = (-100.0, 100.0)
