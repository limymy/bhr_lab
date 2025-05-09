from bhr_lab.tasks.locomotion.velocity.config.bhr_base.mirror_cfg import MirrorCfg, configclass

@configclass
class Bhr8Fc2NoArmMirrorCfg(MirrorCfg):
    def __init__(self):
        # Joint IDs
        lhipYaw   = 0
        rhipYaw   = 1
        lhipRoll  = 2
        rhipRoll  = 3
        lhipPitch = 4
        rhipPitch = 5
        lknee     = 6
        rknee     = 7
        lankle1   = 8
        rankle1   = 9
        lankle2   = 10
        rankle2   = 11
        JOINT_NUM = 12

        left_joint_ids  = [
            lhipYaw,
            lhipRoll,
            lhipPitch,
            lknee,
            lankle1,
            lankle2,
        ]
        right_joint_ids = [
            rhipYaw,
            rhipRoll,
            rhipPitch,
            rknee,
            rankle1,
            rankle2,
        ]

        self.action_mirror_id_left  = left_joint_ids.copy()
        self.action_mirror_id_right = right_joint_ids.copy()
        self.action_opposite_id = [
            lhipRoll,
            lhipYaw,
            lankle2,
            rhipRoll,
            rhipYaw,
            rankle2,
        ]

        BASE_ANG_VEL      = 0
        PROJECTED_GRAVITY = BASE_ANG_VEL + 3
        VELOCITY_COMMANDS = PROJECTED_GRAVITY + 3
        JOINT_POS         = VELOCITY_COMMANDS + 3
        JOINT_VEL         = JOINT_POS + JOINT_NUM
        ACTIONS           = JOINT_VEL + JOINT_NUM

        self.policy_obvs_mirror_id_left = [JOINT_POS + joint_id for joint_id in left_joint_ids]
        self.policy_obvs_mirror_id_left.extend([JOINT_VEL + joint_id for joint_id in left_joint_ids])
        self.policy_obvs_mirror_id_left.extend([ACTIONS + joint_id for joint_id in left_joint_ids])
        self.policy_obvs_mirror_id_right = [JOINT_POS + joint_id for joint_id in right_joint_ids]
        self.policy_obvs_mirror_id_right.extend([JOINT_VEL + joint_id for joint_id in right_joint_ids])
        self.policy_obvs_mirror_id_right.extend([ACTIONS + joint_id for joint_id in right_joint_ids])
        
        self.policy_obvs_opposite_id = [
            BASE_ANG_VEL + 0,
            BASE_ANG_VEL + 2,
            PROJECTED_GRAVITY + 1,
            VELOCITY_COMMANDS + 1,
            VELOCITY_COMMANDS + 2,
        ]
        self.policy_obvs_opposite_id.extend([JOINT_POS + joint_id for joint_id in self.action_opposite_id])
        self.policy_obvs_opposite_id.extend([JOINT_VEL + joint_id for joint_id in self.action_opposite_id])
        self.policy_obvs_opposite_id.extend([ACTIONS   + joint_id for joint_id in self.action_opposite_id])

