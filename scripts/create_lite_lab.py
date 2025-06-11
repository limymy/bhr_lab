#!/usr/bin/env python3
"""
Script to create a lite version of BHR Lab by copying necessary files to a target directory.

This script creates a lightweight BHR Lab project structure containing:
- VSCode settings for development environment
- Training scripts with modified import paths
- Example environment configurations
- Essential configuration files (.gitignore, .flake8, pyproject.toml)

Usage:
    python scripts/create_lite_lab.py --path /absolute/path/to/target --name project_name

Author: BHR Lab Team
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def validate_arguments(args):
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        SystemExit: If validation fails
    """
    if not args.path:
        print("Error: Must provide target path parameter --path")
        sys.exit(1)
    
    if not args.name:
        print("Error: Must provide name parameter --name")
        sys.exit(1)
    
    if not os.path.isabs(args.path):
        print("Error: Path must be an absolute path")
        sys.exit(1)


def create_directories(target_base_path):
    """
    Create necessary directories in the target location.
    
    Args:
        target_base_path (str): Base path for the target project
    """
    directories = [
        ".vscode",
        "scripts/rsl_rl",
        "envs/example_envs"
    ]
    
    for directory in directories:
        dir_path = os.path.join(target_base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def copy_configuration_files(target_base_path):
    """
    Copy essential configuration files to the target directory.
    
    Args:
        target_base_path (str): Base path for the target project
    """
    config_files = ['.gitignore', '.flake8', 'pyproject.toml']
    
    # Copy regular configuration files
    for config_file in config_files:
        # Get the parent directory of the current script (project root)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(script_dir, "..", config_file)
        target_file = os.path.join(target_base_path, config_file)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"Copied configuration file: {config_file}")
        else:
            print(f"Warning: Configuration file not found: {config_file}")
    
    # Handle LICENCE file specially - create lite version based on BHR Lab licence
    create_lite_licence_file(target_base_path)


def create_lite_licence_file(target_base_path):
    """
    Create a LICENCE file for the lite BHR Lab project based on the main BHR Lab licence.
    
    Args:
        target_base_path (str): Base path for the target project
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_licence = os.path.join(script_dir, "..", "LICENCE")
    
    # Read the original BHR Lab licence
    if os.path.exists(source_licence):
        with open(source_licence, 'r', encoding='utf-8') as f:
            original_licence = f.read()
        
        # Update the licence for lite version
        lite_licence = original_licence.replace(
            "Copyright (c) 2025, BHR Lab Project Developers.",
            "Copyright (c) 2025, BHR Lab Lite Project.\nBased on BHR Lab - Copyright (c) 2025, BHR Lab Project Developers."
        )
        
        licence_file = os.path.join(target_base_path, "LICENCE")
        with open(licence_file, 'w', encoding='utf-8') as f:
            f.write(lite_licence)
        print("Created LICENCE file for lite project based on BHR Lab licence")
    else:
        print("Warning: Could not find BHR Lab LICENCE file, creating basic licence")
        # Fallback to creating a basic licence if original not found
        basic_licence = """Copyright (c) 2025, BHR Lab Lite Project.
Based on BHR Lab - Copyright (c) 2025, BHR Lab Project Developers.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
        licence_file = os.path.join(target_base_path, "LICENCE")
        with open(licence_file, 'w', encoding='utf-8') as f:
            f.write(basic_licence)
        print("Created basic LICENCE file for lite project")


def copy_vscode_settings(target_base_path):
    """
    Copy VSCode settings.json to target directory.
    
    Args:
        target_base_path (str): Base path for the target project
    """
    source_file = ".vscode/settings.json"
    target_file = os.path.join(target_base_path, ".vscode", "settings.json")
    
    if os.path.exists(source_file):
        shutil.copy2(source_file, target_file)
        print(f"Copied file: {source_file} -> {target_file}")
    else:
        print(f"Warning: Source file not found {source_file}")


def copy_tasks_init(target_base_path):
    """
    Copy tasks/__init__.py to envs folder for environment registration.
    
    Args:
        target_base_path (str): Base path for the target project
    """
    source_file = "source/bhr_lab/bhr_lab/tasks/__init__.py"
    target_file = os.path.join(target_base_path, "envs", "__init__.py")
    
    if os.path.exists(source_file):
        shutil.copy2(source_file, target_file)
        print(f"Copied file: {source_file} -> {target_file}")
    else:
        print(f"Warning: Source file not found {source_file}")


def create_example_env_files(target_base_path):
    """
    Create example environment configuration files.
    
    This function creates three files in the envs/example_envs directory:
    - env_cfg.py: Environment configuration
    - agent_cfg.py: Agent/algorithm configuration  
    - __init__.py: Gymnasium environment registration
    
    Args:
        target_base_path (str): Base path for the target project
    """
    example_dir = os.path.join(target_base_path, "envs", "example_envs")
    
    # Create env_cfg.py - Environment configuration
    env_cfg_content = """# example_envs/env_cfg.py
from bhr_lab.tasks.locomotion.velocity.config.bhr8_fc2.noarm_env_cfg import *

@configclass
class ExampleEnvCfg(Bhr8Fc2NoArmFlatNSCEnvCfg):
    \"\"\"Example environment configuration based on BHR8 FC2 without arms.\"\"\"
    
    def __post_init__(self):
        super().__post_init__()
        # Add custom configuration here if needed
"""
    
    env_cfg_file = os.path.join(example_dir, "env_cfg.py")
    with open(env_cfg_file, 'w', encoding='utf-8') as f:
        f.write(env_cfg_content)
    print(f"Created file: {env_cfg_file}")
    
    # Create agent_cfg.py - Agent configuration
    agent_cfg_content = """# example_envs/agent_cfg.py
from bhr_lab.tasks.locomotion.velocity.config.bhr_base.agents.rsl_rl_cfg import BasePPORunnerCfg

class ExamplePPORunnerCfg(BasePPORunnerCfg):
    \"\"\"Example PPO configuration for training.\"\"\"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.experiment_name = "example"
        # Add custom training parameters here if needed
"""
    
    agent_cfg_file = os.path.join(example_dir, "agent_cfg.py")
    with open(agent_cfg_file, 'w', encoding='utf-8') as f:
        f.write(agent_cfg_content)
    print(f"Created file: {agent_cfg_file}")
    
    # Create __init__.py - Environment registration
    init_content = """# example_envs/__init__.py
\"\"\"Example environment registration for gymnasium.\"\"\"

import gymnasium as gym

from .agent_cfg import ExamplePPORunnerCfg
from .env_cfg import ExampleEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="example",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ExampleEnvCfg,
        "rsl_rl_cfg_entry_point": ExamplePPORunnerCfg,
    },
)
"""
    
    init_file = os.path.join(example_dir, "__init__.py")
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    print(f"Created file: {init_file}")


def copy_and_modify_scripts(target_base_path):
    """
    Copy and modify training scripts with correct import paths.
    
    This function copies training scripts and modifies the import statements
    to use the new project structure.
    
    Args:
        target_base_path (str): Base path for the target project
    """
    script_files = ["cli_args.py", "train.py", "play.py"]
    source_dir = "scripts/rsl_rl"
    target_dir = os.path.join(target_base_path, "scripts", "rsl_rl")
    
    for script_file in script_files:
        source_file = os.path.join(source_dir, script_file)
        target_file = os.path.join(target_dir, script_file)
        
        if os.path.exists(source_file):
            if script_file in ["train.py", "play.py"]:
                # Read, modify and write the file
                try:
                    with open(source_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Replace the import statement with new path
                    # Convert Windows paths to Unix-style for cross-platform compatibility
                    old_import = "import bhr_lab.tasks  # noqa: F401"
                    new_import = "\nimport envs  # Import local environment configurations"

                    modified_content = content.replace(old_import, new_import)
                    
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    print(f"Copied and modified file: {source_file} -> {target_file}")
                    
                except Exception as e:
                    print(f"Error processing {script_file}: {e}")
                    
            elif script_file == "cli_args.py":
                # Special handling for cli_args.py - add PROJECT_ROOT before TYPE_CHECKING
                try:
                    with open(source_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find the position of "if TYPE_CHECKING:" and insert PROJECT_ROOT code before it
                    type_checking_line = "if TYPE_CHECKING:"
                    if type_checking_line in content:
                        # Add PROJECT_ROOT setup before TYPE_CHECKING
                        project_root_code = "import os\nimport sys\n\nPROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nsys.path.append(PROJECT_ROOT)\n\n"
                        modified_content = content.replace(
                            type_checking_line,
                            project_root_code + type_checking_line
                        )
                    else:
                        # Fallback: if TYPE_CHECKING not found, just copy the file
                        modified_content = content
                        print(f"Warning: 'if TYPE_CHECKING:' not found in {script_file}, copying without modification")
                    
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    print(f"Copied and modified file: {source_file} -> {target_file}")
                    
                except Exception as e:
                    print(f"Error processing {script_file}: {e}")
                    
            else:
                # Just copy the file without modification
                try:
                    shutil.copy2(source_file, target_file)
                    print(f"Copied file: {source_file} -> {target_file}")
                except Exception as e:
                    print(f"Error copying {script_file}: {e}")
        else:
            print(f"Warning: Source file not found {source_file}")


def create_readme(target_base_path):
    """
    Create a README file for the lite lab project.
    
    Args:
        target_base_path (str): Base path for the target project
    """
    readme_content = """# BHR Lab Lite

This is a lightweight version of BHR Lab created for focused development and testing.

## Structure

- `envs/`: Environment configurations and registrations
- `scripts/`: Training and evaluation scripts
- `.vscode/`: VSCode development settings

## Usage

### Training
```bash
python scripts/rsl_rl/train.py --task=example --headless --num_envs=4096
```

### Playing/Evaluation
```bash
python scripts/rsl_rl/play.py --task=example
```

## Development

This project includes example environment configurations that can be modified for your specific needs.
See `envs/example_envs/` for reference implementations.

## License

This project is based on BHR Lab, which is built upon Isaac Lab.
- BHR Lab Lite: MIT License
- BHR Lab: MIT License  
- Isaac Lab: BSD 3-Clause License

See the LICENCE file for full details.
"""
    
    readme_file = os.path.join(target_base_path, "README.md")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("Created README.md")


def main():
    """
    Main function to execute the script.
    
    This function orchestrates the creation of a lite BHR Lab project by:
    1. Validating command line arguments
    2. Creating directory structure
    3. Copying essential configuration files
    4. Creating example environment configurations
    5. Copying and modifying training scripts
    """
    parser = argparse.ArgumentParser(
        description="Create a lightweight version of BHR Lab by copying necessary files to a target directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/create_lite_lab.py --path /home/user/projects --name my_bhr_lab
  python scripts/create_lite_lab.py --path C:/Projects --name bhr_test
        """
    )
    parser.add_argument(
        "--path", 
        type=str, 
        required=True,
        help="Absolute path to the target directory"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        required=True,
        help="Name of the target project"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Create target paths
    target_base_path = os.path.join(args.path, args.name)
    target_absolute_path = os.path.abspath(target_base_path)
    
    print("Creating BHR Lab Lite project...")
    print(f"Target path: {target_absolute_path}")
    print("-" * 50)
    
    try:
        # Create the main target directory
        os.makedirs(target_base_path, exist_ok=True)
        print(f"Created main directory: {target_base_path}")
        
        # Create subdirectories
        create_directories(target_base_path)
        
        # Copy configuration files
        copy_configuration_files(target_base_path)
        
        # Copy VSCode settings
        copy_vscode_settings(target_base_path)
        
        # Copy tasks init file
        copy_tasks_init(target_base_path)
        
        # Create example environment files
        create_example_env_files(target_base_path)
        
        # Copy and modify scripts
        copy_and_modify_scripts(target_base_path)
        
        # Create README
        create_readme(target_base_path)
        
        print("-" * 50)
        print(f"✅ Completed! BHR Lab Lite has been created at: {target_absolute_path}")
        print("📁 Project structure ready for development")
        print("🚀 You can now start training with: python scripts/rsl_rl/train.py --task=example")
        
    except Exception as e:
        print(f"❌ Error creating project: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
