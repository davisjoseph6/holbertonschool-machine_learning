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

Upgrade pip
```bash
pip install --upgrade pip
```

## 2. Create a Virtual Environment

Set up a dedicated virtual environment for the project:

```bash
# Install Python 3.10 and the virtual environment package if not already installed
sudo apt install -y python3.10 python3.10-venv

# Create and activate the virtual environment
python3.10 -m venv myvenv
source myvenv/bin/activate
```

## 3. Install Dependencies

Install core dependencies:

```bash
pip install gymnasium numpy
```

Install project-specific dependencies:

```bash
pip install -r requirements.txt
```

## 4. Deactivate the Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

Note: Ensure you are in the project directory before running the commands in steps 3 and 4.
