# Basic Requirements for Deploying the Project

Follow the steps below to set up the project environment.

---

## 1. Install Basic Utilities

Run the following commands to install essential tools and dependencies:

```bash
# Update and upgrade the system
sudo apt-get update && sudo apt-get upgrade -y

# Install utilities
sudo apt install -y vim git cmake python3-pip python3-tk python3-venv pipx

# Add the DeadSnakes PPA for alternative Python versions
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update && sudo apt upgrade -y
```

## 2. Install Additional System Libraries

Install the required system dependencies for the project:

```bash
# Install Cairo development libraries for graphics-related packages
sudo apt-get install -y libcairo2-dev

# Install Python development headers to build C extensions
sudo apt-get install -y python3.10-dev

# Install tkinter for Python 3.10 (required for GUI applications)
sudo apt-get install -y python3.10-tk
```

## 3. Create a Virtual Environment

Set up a dedicated virtual environment for the project:

```bash
# Install Python 3.10 and the virtual environment package if not already installed
sudo apt install -y python3.10 python3.10-venv

# Create and activate the virtual environment
python3.10 -m venv myvenv --system-site-packages
source myvenv/bin/activate
```

Upgrade pip
```bash
pip install --upgrade pip
```

## 4. Install Dependencies

Install core dependencies:

```bash
pip install gymnasium numpy
```

Install Tensorstore without dependencies:

```bash
pip install tensorstore==0.1.68 --no-deps
```

Install additional project-specific dependencies:

```bash
pip install -r requirements.txt
```

Upgrade pip, setuptools, and wheel to ensure compatibility:

```bash
pip install --upgrade pip setuptools wheel
```

Verify dependencies and check for conflicts:

```bash
pip check
```

## 4. Deactivate the Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

Note: Ensure you are in the project directory before running the commands in steps 3 and 4.
