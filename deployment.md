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

**Important**: Verify `tkinter` installation:

```bash
python3.10 -m tkinter
```
You should see a small GUI window open. If this fails, ensure `Python 3.10` and `python3.10-tk` are correctly installed.


### Create a Virtual Environment (optional)

Set up a dedicated virtual environment for the project. This is an **optional** step in case you are not on `Ubuntu 22.04` ("Jammy Jellyfish") environment with `Python3.10` version:

```bash
# Install Python 3.10 and the virtual environment package if not already installed
sudo apt install -y python3.10 python3.10-venv

# Create and activate the virtual environment
python3.10 -m venv myvenv --system-site-packages
source myvenv/bin/activate
```

## 3. Upgrade pip

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
Note on `tkinter`: 
`tkinter` is a system-level dependency and cannot be included in `requirements.txt`. Make sure it is installed as described in **Step 2**.

### Deactivate the Virtual Environment

When finished, deactivate the virtual environment (if you have created it previously):

```bash
deactivate
```

Notes: 
1. Ensure you are in the project directory before running the commands in steps 3 and 4.
2. If running in a non-GUI environment, consider using a non-interactive `matplotlib` backend like `Agg` for saving plots.

# Using a GPU to Speed Up Model Training (eg: Document Summarization)

This short README guides you through setting up a **GPU** (e.g., an NVIDIA GeForce MX/RTX) in **WSL2** (or native Linux) to accelerate tasks like summarization, which otherwise run slowly on CPU.

## 1. Confirm You Have an NVIDIA GPU

1. **On Windows**:
   - Open **Device Manager** → **Display Adapters** to confirm you see an NVIDIA card (e.g., MX250, RTX 3050).
2. **On WSL**:  
   - **`lspci | grep -i nvidia`** typically shows nothing in WSL. That’s normal.
   - In WSL2, you can rely on **`nvidia-smi`** (once set up) to confirm GPU details.

## 2. Update NVIDIA Drivers (Windows Side)

To enable GPU compute in WSL:
1. **Download** and **Install** the latest **NVIDIA** driver for your GPU from [nvidia.com](https://www.nvidia.com/Download/index.aspx) or via **GeForce Experience**.
2. **Reboot** Windows to finish installation.
3. **In WSL**, verify your driver is recognized after final steps by running:
   ```bash
   nvidia-smi
   ```
If it shows your GPU model and driver version (≥470 recommended), you’re good.

## 3. Install `nvidia-utils-XXX` Inside WSL

Once your Windows driver is updated:

1. **sudo apt-get update**
2. **Choose** a matching (or newer) version from `apt search nvidia-utils`.
3. For example:
   ```bash
   sudo apt-get install nvidia-utils-535
   ```
4. Re-run `nvidia-smi` inside WSL:
   ```bash
   nvidia-smi
   ```
   You should see GPU info.

## 4. Install a CUDA-Enabled PyTorch Environment (Optional)

If you want to do Summarization (or other ML tasks) with GPU acceleration:

1. **Install Miniconda** (recommended) in WSL:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
2. **Create** a conda environment (e.g., `wsl-gpu`):
   ```bash
   conda create -n wsl-gpu python=3.9
   conda activate wsl-gpu
   ```
3. **Install** GPU-enabled PyTorch:
   ```bash
   # (A) conda approach:
   conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

   # OR (B) pip approach:
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   ```
4. Verify:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
→ Expect True.

## 5. Summarize Documents with GPU

1. **Load** or **fine-tune** a summarization model (e.g., BART) in your GPU environment.
2. **Run** your summarization script 
3. If your code does something like:
   ```python
   model.to("cuda")
   ```
it will push the model to GPU, drastically speeding up inference.

## 6. Troubleshooting

- `torch.cuda.is_available() == False`:
  - Possibly installed a CPU-only build. Reinstall a GPU build (see step 4).
- **“No GPU found”** or `nvidia-smi` not working:
  - Check your driver versions, or see if Windows driver is older than 470.
- **WSL kernel headers**:
  - Not typically needed for GPU pass-through. apt-based `linux-headers` can’t be installed for a Microsoft WSL kernel.

## 

That’s it! Now you can run summarization or other ML tasks in WSL2 using your NVIDIA GPU—reducing time per document from many seconds on CPU to often a fraction of that on GPU.
