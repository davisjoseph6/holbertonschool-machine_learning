# Project Setup Instructions

This guide provides instructions to set up a Python 3.10 environment using `pyenv`, create a virtual environment for the project (if necessary), and make required modifications to the `callbacks.py` file for compatibility.

## Prerequisites

Before starting, make sure you have the following:

- **Ubuntu/Debian** OS (or a similar Linux distribution)
- Access to a terminal with `sudo` privileges

## Step 1: Install `pyenv`

`pyenv` allows you to install and manage multiple Python versions on your system.

### 1.1 Install Required Dependencies

Run the following commands to install the necessary dependencies for building Python:

```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev
```

### 1.2 Install pyenv

Use the following command to install pyenv:

```bash
curl https://pyenv.run | bash
```

### 1.3 Configure Your Shell for pyenv

Add the following lines to your shell configuration file (~/.bashrc for Bash or ~/.zshrc for Zsh) to make pyenv available in your terminal:

```bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

After adding these lines, reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc for Zsh
```

### 1.4 Verify pyenv Installation

Run the following command to confirm that pyenv is installed:

```bash
pyenv --version
```

## Step 2: Install Python 3.10 (If Needed)

Now, use pyenv to install Python 3.10.12, which is required for this project.

Note: If your current Python version is already 3.10.12 or lower, you can skip creating a virtual environment by directly upgrading to Python 3.10.12 with pyenv. Only continue with the virtual environment creation if you have a Python version higher than 3.10.12 (e.g., Python 3.12).

### 2.1 Install Python 3.10.12

```bash
pyenv install 3.10.12
```

If you encounter any errors, ensure all required libraries are installed and try again.

### 2.2 Set Python 3.10.12 as the Local Version

Navigate to your project directory, then set Python 3.10.12 as the local version:

```bash
pyenv local 3.10.12
```

This creates a .python-version file in the project directory, ensuring Python 3.10.12 is used within this folder.

## Step 3: Create and Activate the Virtual Environment

Note: You only need to create this virtual environment if your current Python version is greater than 3.10.12. If you are on Python 3.10.12 or lower, you may skip this step and continue with Step 4 directly.

### 3.1 Create the Virtual Environment

With Python 3.10 set as the local version, create a virtual environment named myenv:

```bash
python3.10 -m venv myenv
```

### 3.2 Activate the Virtual Environment

Activate the virtual environment with the following command:

```bash
source myenv/bin/activate
```

You should see (myenv) at the beginning of your terminal prompt, indicating that the environment is active.

### 3.3 Install Required Packages

With the virtual environment active, install the required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

## Step 4: Modify callbacks.py for Compatibility

In this project, callbacks.py may need modifications to work correctly with the installed dependencies. A reference copy of the original `callbacks.py` file is kept in this repository for reference. Follow these steps to update the active callbacks.py file:

1. Locate callbacks.py:

The callbacks.py file should be in the site-packages directory of your virtual environment. If your virtual environment is named myenv, locate callbacks.py with:

```bash
find myenv/lib -name "callbacks.py"
```

2. Edit callbacks.py:

Open the callbacks.py file in a text editor. Replace any lines referencing KERAS_VERSION with the following to avoid compatibility issues:

```python
import tensorflow as tf
```

Then, remove all lines that check or use KERAS_VERSION.

3. Save and Exit:

After making the changes, save the file and exit the editor.

## Step 5: Verify the Setup

To confirm that the setup is correct, check the Python version within the virtual environment:

```bash
python --version
```

This should display Python 3.10.12, confirming that the correct Python version is being used.

## Deactivating the Virtual Environment

When you're done working in the environment, you can deactivate it by running:

```bash
deactivate
```

## Additional Notes

If you need to use this environment on a different machine with Python 3.12, follow the same instructions but skip Step 2 (installing Python 3.10 with pyenv). The environment should still work with Python 3.12 for most dependencies.
