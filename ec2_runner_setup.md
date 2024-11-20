**Install docker and docker-compose on Ubuntu 22.04**
__PreRequisites__:

    * Have an aws account with a user that has the necessary permissions
    * Have the access key either on env variables or in the github actions secrets
    * Have an ec2 runner instance running/created in the aws account
    * Have a s3 bucket created in the aws account
    * Have aws container registry created in the aws account 
__Local VM setup__:
    * Install aws configure and setup the access key and secret key and the right zone
        ```bash
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install   
        aws configure
        ```
    

__Install docker__:
```bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
sudo systemctl restart docker
sudo reboot
docker --version
docker ps
```
__Install docker-compose__:
```bash
sudo rm /usr/local/bin/docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.30.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

__Github actions self-hosted runner__:
```bash
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.320.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.320.0/actions-runner-linux-x64-2.320.0.tar.gz
echo "93ac1b7ce743ee85b5d386f5c1787385ef07b3d7c728ff66ce0d3813d5f46900  actions-runner-linux-x64-2.320.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-x64-2.320.0.tar.gz
./config.sh --url https://github.com/soutrik71/pytorch-template-aws --token <Latest>
# cd actions-runner/
./run.sh
./config.sh remove --token <> # To remove the runner
# https://github.com/soutrik71/pytorch-template-aws/settings/actions/runners/new?arch=x64&os=linux
```
__Activate aws cli__:
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
unzip awscliv2.zip
sudo ./aws/install
aws --version
aws configure

```
__S3 bucket operations__:
```bash
aws s3 cp data s3://deep-bucket-s3/data --recursive
aws s3 ls s3://deep-bucket-s3
aws s3 rm s3://deep-bucket-s3/data --recursive
```

__Cuda Update Setup__:
```bash
# if you already have nvidia drivers installed and you have a Tesla T4 GPU
sudo apt update
sudo apt upgrade
sudo reboot

sudo apt --fix-broken install
sudo apt install ubuntu-drivers-common
sudo apt autoremove

nvidia-smi
lsmod | grep nvidia

sudo apt install nvidia-cuda-toolkit
nvcc --version

ls /usr/local/ | grep cuda
ldconfig -p | grep cudnn
lspci | grep -i nvidia

Based on the provided details, here is the breakdown of the information about your GPU, CUDA, and environment setup:

---

### **1. GPU Details**
- **Model**: Tesla T4  
  - A popular NVIDIA GPU for deep learning and AI workloads.  
  - It belongs to the Turing architecture (TU104GL).  

- **Memory**: 16 GB  
  - Only **2 MiB is currently in use**, indicating minimal GPU activity.

- **Temperature**: 25°C  
  - The GPU is operating at a low temperature, suggesting no heavy utilization currently.

- **Power Usage**: 11W / 70W  
  - The GPU is in idle or low-performance mode (P8).

- **MIG Mode**: Not enabled.  
  - MIG (Multi-Instance GPU) mode is specific to NVIDIA A100 and other GPUs, so it is not applicable here.

---

### **2. Driver and CUDA Version**
- **Driver Version**: 535.216.03  
  - Installed NVIDIA driver supports CUDA 12.x.

- **CUDA Runtime Version**: 12.2  
  - This is the active runtime version compatible with the driver.

---

### **3. CUDA Toolkit Versions**
From your `nvcc` and file system checks:
- **Default `nvcc` Version**: CUDA 10.1  
  - The system's default `nvcc` is pointing to an older CUDA 10.1 installation (`nvcc --version` output shows CUDA 10.1).  

- **Installed CUDA Toolkits**:
  - `cuda-12`
  - `cuda-12.2`
  - `cuda` (likely symlinked to `cuda-12.2`)
  
  Multiple CUDA versions are installed. However, the runtime and drivers align with **CUDA 12.2**, while the default compiler (`nvcc`) is still from CUDA 10.1.

---

### **4. cuDNN Version**
From `cudnn_version.h` and `ldconfig`:
- **cuDNN Version**: 9.5.1  
  - This cuDNN version is compatible with **CUDA 12.x**.
- **cuDNN Runtime**: The libraries for cuDNN 9 are present under `/lib/x86_64-linux-gnu`.

---

### **5. NVIDIA Software Packages**
From `dpkg`:
- **NVIDIA Drivers**: Driver version 535 is installed.
- **CUDA Toolkit**: Multiple versions installed (`10.1`, `12`, `12.2`).
- **cuDNN**: Versions for CUDA 12 and CUDA 12.6 are installed (`cudnn9-cuda-12`, `cudnn9-cuda-12-6`).

---

### **6. Other Observations**
- **Graphics Settings Issue**: 
  - `nvidia-settings` failed due to the lack of a display server connection (`Connection refused`). Likely, this is a headless server without a GUI environment.
  
- **OpenGL Tools Missing**: 
  - `glxinfo` command is missing, indicating the `mesa-utils` package needs to be installed.

---

### **Summary of Setup**
- **GPU**: Tesla T4  
- **Driver Version**: 535.216.03  
- **CUDA Runtime Version**: 12.2  
- **CUDA Toolkit Versions**: 10.1 (default `nvcc`), 12, 12.2  
- **cuDNN Version**: 9.5.1 (compatible with CUDA 12.x)  
- **Software Packages**: NVIDIA drivers, CUDA, cuDNN installed
```

__CUDA New Installation__:
```bash
# if you don't have nvidia drivers installed and you have a Tesla T4 GPU 
lspci | grep -i nvidia # Check if the GPU is detected
To set up the T4 GPU from scratch, starting with no drivers or CUDA tools, and replicating the above configurations and drivers, follow these reverse-engineered steps:

---

### **1. Update System**
Ensure the system is updated:
```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

---

### **2. Install NVIDIA Driver**
#### **a. Identify Required Driver**
The T4 GPU requires a compatible NVIDIA driver version. Based on your configurations, we will install **Driver 535**.

#### **b. Add NVIDIA Repository**
Add the official NVIDIA driver repository:
```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:graphics-drivers/ppa
sudo apt update
```

#### **c. Install Driver**
Install the driver for the T4 GPU:
```bash
sudo apt install -y nvidia-driver-535
```

#### **d. Verify Driver Installation**
Reboot the system and check the driver:
```bash
sudo reboot
nvidia-smi
```
This should display the GPU model and driver version.

---

### **3. Install CUDA Toolkit**
#### **a. Add CUDA Repository**
Download and install the CUDA 12.2 repository for Ubuntu 20.04:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
```

#### **b. Install CUDA Toolkit**
Install CUDA 12.2:
```bash
sudo apt install -y cuda
```

#### **c. Set Up Environment Variables**
Add CUDA binaries to the PATH and library paths:
```bash
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### **d. Verify CUDA Installation**
Check CUDA installation:
```bash
nvcc --version
nvidia-smi
```

---

### **4. Install cuDNN**
#### **a. Download cuDNN**
Download cuDNN 9.5.1 (compatible with CUDA 12.x) from the [NVIDIA cuDNN page](https://developer.nvidia.com/cudnn). You’ll need to log in and download the appropriate `.deb` files for Ubuntu 20.04.

#### **b. Install cuDNN**
Install the downloaded `.deb` files:
```bash
sudo dpkg -i libcudnn9*.deb
```

#### **c. Verify cuDNN**
Check the installed version:
```bash
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

---

### **5. Install NCCL and Other Libraries**
Install additional NVIDIA libraries (like NCCL) required for distributed deep learning:
```bash
sudo apt install -y libnccl2 libnccl-dev
```

---

### **6. Install PyTorch**
#### **a. Install Python Environment**
Install Python and `pip` if not already present:
```bash
sudo apt install -y python3 python3-pip
```

#### **b. Install PyTorch with CUDA 12.2**
Install PyTorch with the appropriate CUDA runtime:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
```

#### **c. Test PyTorch**
Run a quick test:
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should return "Tesla T4"
```

---

### **7. Optional: Install Nsight Tools**
For debugging and profiling:
```bash
sudo apt install -y nsight-compute nsight-systems
```

---

### **8. Check for OpenGL**
If you need OpenGL utilities (like `glxinfo`):
```bash
sudo apt install -y mesa-utils
glxinfo | grep "OpenGL version"
```

---

### **9. Validate Entire Setup**
Run the NVIDIA sample tests to confirm the configuration:
```bash
cd /usr/local/cuda-12.2/samples/1_Utilities/deviceQuery
make
./deviceQuery
```
If successful, it should show details of the T4 GPU.

---

### **Summary of Installed Components**
- **GPU**: Tesla T4
- **Driver**: 535
- **CUDA Toolkit**: 12.2
- **cuDNN**: 9.5.1
- **PyTorch**: Installed with CUDA 12.2 support

This setup ensures your system is ready for deep learning workloads with the T4 GPU.

Install conda and create a new environment for the project
Install pytorch and torchvision in the new environment
Install other dependencies like numpy, pandas, matplotlib, etc.
Run the project code in the new environment
>>> import torch
>>> print(torch.cuda.is_available())
>>> print(torch.cuda.get_device_name(0))
>>> print(torch.version.cuda)
```
__CUDA Docker Setup__:
```bash
# If you are using docker and want to run a container with CUDA support
sudo apt install -y nvidia-container-toolkit
nvidia-ctk --version
sudo systemctl restart docker
sudo systemctl status docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvcc --version
```
