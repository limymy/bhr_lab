import math
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    configclass, RewTerm, SceneEntityCfg
)
import bhr_lab.tasks.locomotion.velocity.mdp as mdp

@configclass
class BaseRewardsCfg:
    '''Base rewards configuration'''
    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- basic penalties
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_sagittal_l2 = RewTerm(func=mdp.saggital_orientation_l2, weight=-2.0)
    flat_orientation_coronal_l2 = RewTerm(func=mdp.coronal_orientation_l2, weight=-2.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"), 
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot")
        },
    )
    touch_down_slightly = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 700.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot"])
        }
    )
    biped_single_feet_in_air = RewTerm(
        func=mdp.single_feet_in_air_cmd,
        weight=1.0,
        params={"command_name": "base_velocity", "reward_threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot")},
    )
    biped_desired_step_time = RewTerm(
        func=mdp.desired_step_time_cmd,
        weight=-1.0,
        params={
            "command_name": "base_velocity", 
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot"]), 
            "desired_time": 0.4, "reward_value": 0.4},
    )
    biped_no_double_feet_air = RewTerm(
        func=mdp.biped_no_double_feet_air,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot"])},
    )

    # -- joint deviation and limit
    joint_deviation_hipYaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hipYaw"])},
    )    
    joint_deviation_hipRoll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hipRoll"])},
    )
    joint_deviation_hipPitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hipPitch"])},
    )
    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*knee"])},
    ) 
    joint_deviation_ankleRoll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankleRoll"])},
    ) 
    joint_deviation_anklePitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*anklePitch"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder.*", ".*elbow"])},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )

    # -- joint coordination
    joint_coordination_hip = RewTerm(
        func=mdp.joint_coordination,
        weight=-0.2,
        params={
            "joint1_cfg": SceneEntityCfg("robot", joint_names=["lhipPitch"]), 
            "joint2_cfg": SceneEntityCfg("robot", joint_names=["rhipPitch"]),
            "ratio": -1.0,
        },
    )
    joint_coordination_larm_leg = RewTerm(
        func=mdp.joint_coordination,
        weight=-0.2,
        params={
            "joint1_cfg": SceneEntityCfg("robot", joint_names=["lshoulderPitch"]), 
            "joint2_cfg": SceneEntityCfg("robot", joint_names=["rhipPitch"]),
            "ratio": 1.0, 
        },
    )
    joint_coordination_rarm_leg = RewTerm(
        func=mdp.joint_coordination,
        weight=-0.2,
        params={
            "joint1_cfg": SceneEntityCfg("robot", joint_names=["rshoulderPitch"]), 
            "joint2_cfg": SceneEntityCfg("robot", joint_names=["lhipPitch"]),
            "ratio": 1.0, 
        },
    )

    # body distance
    distance_feet = RewTerm(
        func=mdp.lateral_distance,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot"]), 
            "min_dist": 0.30, 
            "max_dist": 0.40},
    )
    distance_knee = RewTerm(
        func=mdp.lateral_distance,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*calf"]),
            "min_dist": 0.05,
            "max_dist": 0.40},
    )
    biped_leg_no_cross = RewTerm(
        func=mdp.biped_leg_no_cross,
        weight=-0.1,
        params={
            "assetl_cfg": SceneEntityCfg("robot", joint_names=["lhipYaw"]),
            "assetr_cfg": SceneEntityCfg("robot", joint_names=["rhipYaw"]),
        }
    )

    # from robot_lab
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    def reset_body_names(self, feet_names: list[str], knee_names: list[str]):
        # Feet related rewards
        self.feet_slide.params["asset_cfg"].body_names = feet_names
        self.feet_slide.params["sensor_cfg"].body_names = feet_names
        self.touch_down_slightly.params["sensor_cfg"].body_names = feet_names
        self.biped_single_feet_in_air.params["sensor_cfg"].body_names = feet_names
        self.biped_desired_step_time.params["sensor_cfg"].body_names = feet_names
        self.biped_no_double_feet_air.params["sensor_cfg"].body_names = feet_names
        self.distance_feet.params["asset_cfg"].body_names = feet_names

        # Knee related rewards
        self.distance_knee.params["asset_cfg"].body_names = knee_names
