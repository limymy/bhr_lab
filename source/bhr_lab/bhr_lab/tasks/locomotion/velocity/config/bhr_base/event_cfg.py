from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg, configclass, EventTerm, SceneEntityCfg, mdp
)
@configclass
class BaseEventCfg(EventCfg):
    '''Configuration for the base event'''
    def __post_init__(self):
        self.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        self.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.base_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.push_robot.params["velocity_range"] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
        }
        self.reset_base.params["velocity_range"] = {
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (-0.0, 0.0),
            "roll": (-0.0, 0.0),
            "pitch": (-0.0, 0.0),
            "yaw": (-0.0, 0.0),
        }
        self.reset_robot_joints.params["position_range"] = (1.0, 1.0)

    def reset_base_name(self, base_name: str):
        if self.add_base_mass is not None:
            self.add_base_mass.params["asset_cfg"].body_names = [base_name]
        if self.base_external_force_torque is not None:
            self.base_external_force_torque.params["asset_cfg"].body_names = [base_name]

@configclass
class RandomizationEventCfg(EventCfg):
    '''Configuration for the randomization event'''
    
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*"]), 
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    add_joint_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    def __post_init__(self):
        self.base_external_force_torque.params["force_range"] = (-20.0, 20.0)
        self.base_external_force_torque.params["torque_range"] = (-10.0, 10.0)
        self.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.physics_material.params["static_friction_range"] = (0.5, 0.95)
        self.physics_material.params["dynamic_friction_range"] = (0.2, 0.8)
        self.physics_material.params["restitution_range"] = (0.0, 0.0)

    def reset_base_name(self, base_name: str):
        if self.add_base_mass is not None:
            self.add_base_mass.params["asset_cfg"].body_names = [base_name]
        if self.base_external_force_torque is not None:
            self.base_external_force_torque.params["asset_cfg"].body_names = [base_name]            
