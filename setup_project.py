"""
Project setup script for JAX Inventory Optimizer

This script initializes the complete project structure, creates default configurations,
and sets up the development environment.
"""

import os
import sys
from pathlib import Path
import subprocess


def create_directory_structure():
    """Create the complete project directory structure"""

    print("📁 Creating project directory structure...")

    directories = [
        # Source code directories
        "src/core",
        "src/methods/traditional",
        "src/methods/ml_methods",
        "src/methods/rl_methods",
        "src/comparison",
        "src/data",
        "src/api/routes",
        "src/utils",

        # Experiment and analysis directories
        "experiments",
        "notebooks",
        "tests",

        # Configuration and results
        "configs",
        "data/raw",
        "data/processed",
        "data/synthetic",
        "results/benchmarks",
        "results/comparisons",
        "results/visualizations",

        # Documentation
        "docs",
        "examples"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Create __init__.py files for Python packages
        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.touch()

    print(f"✅ Created {len(directories)} directories")


def create_init_files():
    """Create __init__.py files for proper Python packaging"""

    print("📦 Creating Python package structure...")

    init_files = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/methods/__init__.py",
        "src/methods/traditional/__init__.py",
        "src/methods/ml_methods/__init__.py",
        "src/methods/rl_methods/__init__.py",
        "src/comparison/__init__.py",
        "src/data/__init__.py",
        "src/api/__init__.py",
        "src/utils/__init__.py"
    ]

    for init_file in init_files:
        Path(init_file).touch()

    print(f"✅ Created {len(init_files)} __init__.py files")


def setup_configs():
    """Setup default configuration files"""

    print("⚙️  Setting up configuration files...")

    # Import and run config setup
    sys.path.insert(0, str(Path.cwd() / "src"))

    try:
        from utils.config import config_manager
        config_types = ["inventory", "traditional", "ml", "rl", "experiment"]

        for config_type in config_types:
            config_manager.create_default_config(config_type)

        print(f"✅ Created {len(config_types)} configuration files")

    except ImportError as e:
        print(f"⚠️  Could not create configs: {e}")
        print("You can create them manually later by running: python src/utils/config.py")


def create_example_scripts():
    """Create example scripts for quick start"""

    print("📝 Creating example scripts...")

    # Example experiment script
    example_experiment = """#!/usr/bin/env python3
'''
Example experiment script - Quick start for JAX Inventory Optimizer
'''

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("🚀 JAX Inventory Optimizer - Example Experiment")
    print("=" * 50)

    # TODO: Add example experiment code here
    print("📊 Loading sample data...")
    print("🔧 Setting up methods...")
    print("📈 Running comparison...")
    print("💾 Saving results...")

    print("\\n✅ Example experiment completed!")
    print("Next: Implement specific methods and run real comparisons")

if __name__ == "__main__":
    main()
"""

    with open("examples/quick_start.py", "w") as f:
        f.write(example_experiment)

    # Example notebook
    example_notebook = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# JAX Inventory Optimizer - Quick Start\\n",
    "\\n",
    "This notebook demonstrates the basic usage of the inventory optimization system."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required libraries\\n",
    "import sys\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Add src to path\\n",
    "sys.path.insert(0, str(Path.cwd().parent / 'src'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load Data and Setup Methods\\n",
    "\\n",
    "TODO: Add example code for loading data and setting up different methods"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""

    with open("notebooks/quick_start.ipynb", "w") as f:
        f.write(example_notebook)

    print("✅ Created example scripts and notebook")


def create_dockerfile():
    """Create Dockerfile for containerized deployment"""

    dockerfile_content = """# JAX Inventory Optimizer Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV JAX_PLATFORM_NAME=cpu

# Expose API port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    print("✅ Created Dockerfile")


def create_github_workflow():
    """Create GitHub Actions workflow for CI/CD"""

    Path(".github/workflows").mkdir(parents=True, exist_ok=True)

    workflow_content = """name: JAX Inventory Optimizer CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
"""

    with open(".github/workflows/ci.yml", "w") as f:
        f.write(workflow_content)

    print("✅ Created GitHub Actions workflow")


def update_readme():
    """Update README with project structure information"""

    readme_addition = """

## 🏗️ Project Structure

```
JAX-Inventory-Optimizer/
├── src/                        # Source code
│   ├── core/                   # Core interfaces and utilities
│   ├── methods/                # Algorithm implementations
│   │   ├── traditional/        # Classical methods (EOQ, Safety Stock)
│   │   ├── ml_methods/         # ML approaches (LSTM, XGBoost)
│   │   └── rl_methods/         # RL approaches (DQN, PPO)
│   ├── comparison/             # Comparison and evaluation framework
│   ├── data/                   # Data loading and preprocessing
│   ├── api/                    # FastAPI service
│   └── utils/                  # Shared utilities
│
├── experiments/                # Experimental scripts
├── notebooks/                  # Jupyter analysis notebooks
├── configs/                    # YAML configuration files
├── data/                       # Datasets
├── results/                    # Experimental results
└── examples/                   # Example usage scripts
```

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   python setup_project.py    # Initialize project structure
   python create_sample_data.py  # Generate sample data
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Example Experiment**
   ```bash
   python examples/quick_start.py
   ```

4. **Explore in Jupyter**
   ```bash
   jupyter notebook notebooks/quick_start.ipynb
   ```

## 📋 Configuration

The system uses YAML configuration files in the `configs/` directory:

- `inventory.yaml`: Inventory problem parameters
- `traditional.yaml`: Traditional method configurations
- `ml.yaml`: ML method hyperparameters
- `rl.yaml`: RL algorithm settings
- `experiment.yaml`: Experiment and evaluation settings

"""

    # Read existing README
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            existing_content = f.read()

        # Add new content if not already present
        if "## 🏗️ Project Structure" not in existing_content:
            with open(readme_path, 'a') as f:
                f.write(readme_addition)
            print("✅ Updated README.md with project structure")
        else:
            print("ℹ️  README.md already contains project structure")
    else:
        print("⚠️  README.md not found")


def main():
    """Main setup function"""

    print("🏭 JAX Inventory Optimizer - Project Setup")
    print("=" * 50)

    # Create project structure
    create_directory_structure()
    create_init_files()

    # Setup configurations
    setup_configs()

    # Create examples and templates
    create_example_scripts()
    create_dockerfile()
    create_github_workflow()

    # Update documentation
    update_readme()

    print("\n" + "=" * 50)
    print("🎉 Project setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Create sample data: python create_sample_data.py")
    print("3. Run example: python examples/quick_start.py")
    print("4. Start developing methods in src/methods/")
    print("5. Run experiments in experiments/")

    print("\n📚 Key Files Created:")
    print("- src/core/interfaces.py (Base interfaces)")
    print("- src/utils/config.py (Configuration management)")
    print("- configs/*.yaml (Default configurations)")
    print("- examples/quick_start.py (Example script)")
    print("- .github/workflows/ci.yml (CI/CD pipeline)")

    print("\n🔧 Architecture Documentation: ARCHITECTURE.md")


if __name__ == "__main__":
    main()