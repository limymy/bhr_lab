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
- Issue: After updating to the latest version of Isaac Lab: 
  - the system prompts missing numba package. Solution: Install numba using pip
    ```bash
    pip install numba
    ```
  - new base_com randmize will cause body name errors (this has been solved)