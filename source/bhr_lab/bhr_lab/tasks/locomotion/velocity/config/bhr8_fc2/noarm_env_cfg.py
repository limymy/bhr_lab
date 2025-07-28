from bhr_lab.tasks.locomotion.velocity.config.bhr_base.env_cfg import (
    configclass, BaseRoughEnvCfg, BaseRewardsCfg
)
from bhr_lab.tasks.locomotion.velocity.config.bhr_base.event_cfg import RandomizationEventCfg

from bhr_lab.assets.bhr8_fc2 import BHR8_FC2_NOARM_CFG, DEG2RAD
from .noarm_mirror_cfg import Bhr8Fc2NoArmMirrorCfg

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg, ImuCfg, ContactSensorCfg
from bhr_lab.tasks.locomotion.velocity.config.bhr_base.recorder import (
    JointStateRecorderCfg,
    CameraRecorderCfg,
    ImuRecorderCfg,
    ContactRecorderCfg
)

from isaaclab.managers.recorder_manager import (
    RecorderManagerBaseCfg,     # Recorder 管理器配置
    DatasetExportMode,          # 选择导出模式
)

@configclass
class Bhr8Fc2NoArmRewards(BaseRewardsCfg):
    '''Rewards for the BHR8 FC2 no arm environment'''

    def __post_init__(self):
        self.reset_body_names(feet_names=[".*foot"], knee_names=[".*calf"])

        self.track_lin_vel_xy_exp.weight = 1.5
        self.track_ang_vel_z_exp.weight  = 1.0
        self.action_rate_l2.weight  = -0.1
        self.dof_torques_l2.weight  = -1.0e-6
        self.dof_acc_l2.weight      = -1.0e-6
        self.dof_vel_l2.weight      = -1.0e-3
        self.lin_vel_z_l2.weight    = -0.0
        self.ang_vel_xy_l2.weight   = -0.5
        self.flat_orientation_sagittal_l2.weight = -5.0
        self.flat_orientation_coronal_l2.weight  = -5.0

        # Joint deviation
        self.joint_deviation_hipPitch.params  ["asset_cfg"].joint_names = [".*hipPitch"]
        self.joint_deviation_hipYaw.params    ["asset_cfg"].joint_names = [".*hipYaw"  ]
        self.joint_deviation_hipRoll.params   ["asset_cfg"].joint_names = [".*hipRoll" ]
        self.joint_deviation_knee.params      ["asset_cfg"].joint_names = [".*knee"    ]
        self.joint_deviation_anklePitch.params["asset_cfg"].joint_names = [".*ankle1"  ]
        self.joint_deviation_ankleRoll.params ["asset_cfg"].joint_names = [".*ankle2"  ]
        self.joint_deviation_hipYaw.weight     = -0.5
        self.joint_deviation_hipRoll.weight    = -0.5
        self.joint_deviation_hipPitch.weight   = -0.0
        self.joint_deviation_knee.weight       = -0.1
        self.joint_deviation_anklePitch.weight = -0.3
        self.joint_deviation_ankleRoll.weight  = -0.5

        # Joint coordination
        self.joint_coordination_hip.params["joint1_cfg"].joint_names = ["lhipPitch"]
        self.joint_coordination_hip.params["joint2_cfg"].joint_names = ["rhipPitch"]
        self.joint_coordination_hip.weight = -0.5

        # Joint limits
        self.joint_pos_limits.weight = -2.0

        # Gait related rewards
        self.biped_single_feet_in_air.weight =  1.5
        self.biped_no_double_feet_air.weight = -1.0
        self.biped_desired_step_time.weight  = -0.0
        self.biped_leg_no_cross.weight       = -0.5
        self.biped_leg_no_cross.params["assetl_cfg"].joint_names = ["lhipYaw"]
        self.biped_leg_no_cross.params["assetr_cfg"].joint_names = ["rhipYaw"]
        self.distance_feet.weight = -0.5
        self.distance_knee.weight = -0.1
        self.stand_still_without_cmd.weight = -0.1
        self.touch_down_slightly.params["threshold"] = 700.0

        # Cancel rewards related to arms
        self.joint_deviation_arms = None
        self.joint_coordination_larm_leg = None
        self.joint_coordination_rarm_leg = None

@configclass
class Bhr8Fc2NoArmRoughEnvCfg(BaseRoughEnvCfg):
    '''Configuration for the BHR8 FC2 no arm environment  with rough terrain'''
    rewards: Bhr8Fc2NoArmRewards = Bhr8Fc2NoArmRewards()
    mirror_cfg: Bhr8Fc2NoArmMirrorCfg = Bhr8Fc2NoArmMirrorCfg()

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = BHR8_FC2_NOARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso"

        # evnets
        self.events.reset_base_name("torso")

        # actions
        if getattr(self.actions, "joint_pos", None) is not None:
            self.actions.joint_pos.clip = {".*":(-10.0, 10.0)}
        
        # change terrain to little rough
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"      ].step_height_range = (0.0, 0.05)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"  ].step_height_range = (0.0, 0.05)
            self.scene.terrain.terrain_generator.sub_terrains["boxes"               ].grid_height_range = (0.0, 0.05)
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"        ].noise_range       = (0.0, 0.05)
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"    ].slope_range       = (0.0, DEG2RAD * 10.0)
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range       = (0.0, DEG2RAD * 10.0)
        
        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso", ".*shoulder.*", ".*hip.*", ".*arm", ".*thigh"]

        # commands
        if getattr(self.commands, "base_velocity", None) is not None:
            self.commands.base_velocity.heading_command = False
            self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
            self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
            self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

@configclass
class Bhr8Fc2NoArmFlatEnvCfg(Bhr8Fc2NoArmRoughEnvCfg):
    '''Configuration for the BHR8 FC2 no arm environment with flat terrain'''

    def __post_init__(self):
        super().__post_init__()

        # change terrain to flat and remove terrain curriculum
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

@configclass
class Bhr8Fc2NoArmFlatNSCEnvCfg(Bhr8Fc2NoArmFlatEnvCfg):
    '''Configuration for the BHR8 FC2 no arm environment with flat terrain'''

    def __post_init__(self):
        super().__post_init__()

        # disable self collisions
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

@configclass
class Bhr8Fc2RandomizationEventCfg(RandomizationEventCfg):
    '''Randomization event for the BHR8 FC2 no arm environment'''
    def __post_init__(self):
        super().__post_init__()
        self.add_link_mass.params["asset_cfg"].body_names=[".*hip.*", ".*thigh", ".*calf", ".*foot", ".*shoulder.*", ".*bigarm"]

@configclass
class Bhr8Fc2NoArmRoughRandomEnvCfg(Bhr8Fc2NoArmRoughEnvCfg):
    '''Configuration for the BHR8 FC2 no arm environment with rough terrain and randomization'''
    events: Bhr8Fc2RandomizationEventCfg = Bhr8Fc2RandomizationEventCfg()

    def __post_init__(self):
        super().__post_init__()

@configclass
class Bhr8Fc2NoArmFlatRandomEnvCfg(Bhr8Fc2NoArmFlatEnvCfg):
    '''Configuration for the BHR8 FC2 no arm environment with flat terrain and randomization'''
    events: Bhr8Fc2RandomizationEventCfg = Bhr8Fc2RandomizationEventCfg()

    def __post_init__(self):
        super().__post_init__()

@configclass
class Bhr8Fc2NoArmWarehouseEnvCfg(Bhr8Fc2NoArmRoughEnvCfg):
    '''Configuration for the BHR8 FC2 no arm environment with warehouse'''

    def __post_init__(self):
        super().__post_init__()

        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # change terrain to flat and remove terrain curriculum
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        )
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        self.scene.cam_left = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso/cam_left",
            update_period=0.05,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.05, 0.0), convention="world"),
        )

        self.scene.cam_right = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso/cam_right",
            update_period=0.05,
            height=1080,
            width=1920,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, -0.05, 0.0), convention="world"),
        )

        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso",
            update_period=0.005,
        )

        self.scene.contact_force_lfoot = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/lfoot",
            update_period=0.005,
            track_pose = True,
        )

        self.scene.contact_force_rfoot = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/rfoot",
            update_period=0.005,
            track_pose = True,
        )

        self.recorders = RecorderManagerBaseCfg(
            dataset_export_dir_path="./logs",       # 数据导出目录
            dataset_filename="bot_dataset",      # 不含扩展名
            dataset_export_mode=DatasetExportMode.EXPORT_ALL,
            export_in_record_pre_reset=True,
        )

        self.recorders.joint_state = JointStateRecorderCfg()
        self.recorders.imu = ImuRecorderCfg(imu_name="imu")
        self.recorders.contact_force_lfoot = ContactRecorderCfg(contact_sensor_name="contact_force_lfoot")
        self.recorders.contact_force_rfoot = ContactRecorderCfg(contact_sensor_name="contact_force_rfoot")
        self.recorders.cam_left = CameraRecorderCfg(cam_name="cam_left")
        self.recorders.cam_right = CameraRecorderCfg(cam_name="cam_right")