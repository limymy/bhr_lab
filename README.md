# BHR Lab: BIT Humanoid Robot Isaac Lab
Just in test. 

## Temporary record
### Example: BHR8 FC2 without arm joints
#### Tasks definition
| Task Name                           | Terrain | Randomization | Self Collision | Mirror |
|-------------------------------------|---------|---------------|----------------|--------|
| `bhr8_fc2_noarm_flat_nsc`           |  flat   | off           | off            | off    |
| `bhr8_fc2_noarm_flat`               |  flat   | off           | on             | off    |
| `bhr8_fc2_noarm_flat_random`        |  flat   | on            | on             | off    |
| `bhr8_fc2_noarm_rough_random`       |  rough  | on            | on             | off    |
| `bhr8_fc2_noarm_flat_nsc_mirror`    |  flat   | off           | off            | on     |
| `bhr8_fc2_noarm_flat_mirror`        |  flat   | off           | on             | on     |
| `bhr8_fc2_noarm_flat_random_mirror` |  flat   | on            | on             | on     |
| `bhr8_fc2_noarm_rough_random_mirror`|  rough  | on            | on             | on     |

#### Training procedure (on RTX5080)
First training: 
```
python scripts/rsl_rl/train.py --task=bhr8_fc2_noarm_flat_nsc_mirror --headless --num_envs=10240 --max_iterations=3000
```

Second training:
```
python scripts/rsl_rl/train.py --task=bhr8_fc2_noarm_flat_mirror --headless --num_envs=10240 --max_iterations=3000 --resume
```

Third training:
```
python scripts/rsl_rl/train.py --task=bhr8_fc2_noarm_flat_random_mirror --headless --num_envs=10240 --max_iterations=3000 --resume
```

Fourth training:
```
python scripts/rsl_rl/train.py --task=bhr8_fc2_noarm_rough_random_mirror --headless --num_envs=4096 --max_iterations=6000 --resume
```

### vscode settings:
```
ctrl + shift + p
tasks: run task
setup_python_env
```

### Create Lite Lab
To create a lightweight version of BHR Lab with essential files only:
```bash
python scripts/create_lite_lab.py --path /absolute/path/to/target --name project_name
```

This script will:
1. Create necessary directories (.vscode, scripts/rsl_rl, envs)
2. Copy VSCode settings
3. Copy task initialization files
4. Copy and modify training scripts with correct import paths

### Troubleshooting

#### 2025-05-27
- **Issue**: After updating to the latest version of Isaac Lab, the system prompts missing numba package
- **Solution**: Install numba using pip
```bash
pip install numba
```

#### 2025-06-07
- **Issue**: IsaacLab API breaking change - `quat_rotate_inverse` deprecated (Issue #2129)
- **Problem**: 
  - IsaacLab #2129 changed `quat_rotate_inverse` to `quat_apply_inverse`
  - This causes dimension errors in `lateral_distance` reward function
  - Old code produces warnings and potential calculation errors
- **Root Cause**: API function name and signature changes in latest IsaacLab versions
- **Solution**:
  1. Update IsaacLab to the latest version
  2. Replace `quat_rotate_inverse` with `quat_apply_inverse` in reward functions
  3. Verify dimension compatibility in affected reward calculations
- **Files Affected**: 
  - Reward functions using quaternion rotations
  - Specifically `lateral_distance` reward computation
- **Note**: Older IsaacLab versions may not have `quat_apply_inverse` function, update is required