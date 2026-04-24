## 🛠️ III. How to Install

All source code in the repository is well-structured, extensively commented, and thoroughly debugged. Here's the overall list of platforms we need to install and run the proposed RadHARAGT:

> (1) MATLAB R2024a or Later.<br>
> (2) Python 3.10.13.<br>
> (3) CUDA 12.1.1 or Later.<br>
> (4) Ollama 0.20.6 or Layer (After Updated in April 2026).

Strictly follow the installation instructions below step by step, and you can use our agent normally:

### 🔹 Part 1: Prepare the MATLAB Environments
1. Install MATLAB R2025b+ with all toolboxes on your computer. Our code can also run on versions R2024a~2025a, but it's not very stable. If you have already installed MATLAB, skip this step.
2. Open MATLAB, click "Home" → "Add-Ons" → "Explore Add-Ons". After the Add-on Explorer is running, search and enter the page of "Large Language Models (LLMs) with MATLAB". Click "Version History". Find version 4.8.0+ (Released after 12 Mar 2026). Click "Add" → "Add to MATLAB". Wait a few minutes until the installation is completed.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d61f6858-eeb8-4c6f-89e8-607ab175b661" />
<br><br>

3. Download the whole GitHub project and unzip to your computer. Then, download the files in "EMDM\" folder of the project and unzip: https://drive.google.com/file/d/1kiz1ZZ60IQRm4IQ9VEb471k5oJsmQEQd/view?usp=sharing. The whole "EMDM\" folder is about 728 MB. After all the files are unzipped, the path of your project working folder should look exactly like this:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d5f005e5-52e3-4d09-815c-be6e5e7186fb" />

### 🔹 Part 2: Prepare the Ollama Environments
1. Download the latest Windows version of Ollama here: https://ollama.com/download. Install Ollama in your computer. After the installation, open Ollama, and sign in.
2. In the model options bar, select the LLM you want. Suggested selection: gemma4:e4b. Of course, you can also choose other models that your computer resources can run smoothly. Please remember the name of the LLM you choose, as you will need to enter it later when using RadHARAGT.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f809c583-efa2-4dd7-ad37-0a61be313a5c" />
<br><br>

3. After selecting your favorite LLM, just prompt any content, and Ollama will automatically download the model. After the model download is complete, **do not exit Ollama and keep it running in the background**.

### 🔹 Part 3: Prepare the CUDA Environments
1. Download the CUDA 12.1 Update 1 version here: https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local. Select the correct Windows system version. It is recommended to choose the "exe (local)" package for installation. If you have already installed CUDA 12.1+ on your computer, skip Part 3.

2. Run the downloaded "cuda_12.1.1_531.14_windows.exe". Just keep selecting the default options to complete the installation process.

### 🔹 Part 4: Prepare the Python Environments
1. Download Anaconda here: https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/. Select the correct Windows system version. It is recommended to choose "Anaconda3-2025.12-2-Windows-x86_64" or later. Similarly, our code can also run on old versions after "Anaconda3-2023.03-0-Windows-x86_64", but it's not very stable. If you have already installed Anaconda, skip this step.
2. Open "Anaconda Prompt". Enter the folder you have downloaded and unzipped the GitHub project.

<img width="979" height="512" alt="image" src="https://github.com/user-attachments/assets/3b34b168-e7de-4f5b-a0b8-3a24b42eb7bf" />
<br><br>

3. Create a new conda environment named "emdm" based on Python 3.10.13. Run the following commands one by one.
```bash
conda create -n emdm python=3.10.13 -y
````

```bash
conda activate emdm
```

4. Install the version of PyTorch compatible with CUDA 12.1.1. Run the following command. Note: If the CUDA version installed on your computer is not 12.1, please also change the version of PyTorch installed by modifying the last three digits of the command.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

5. Install other dependent Python packages. Run the following command one by one.

```bash
cd EMDM\CLIP-main
```

```bash
pip install .
```

```bash
cd ..
```

```bash
pip install scipy==1.15.3 einops==0.8.2 spacy==3.8.13 wandb==0.25.1 smplx==0.1.28 pandas==2.3.3 scikit-learn==1.7.2 chardet==7.4.0.post1
```

```bash
pip install chumpy==0.70 --no-build-isolation
```

```bash
pip install matplotlib==3.5.2 --upgrade
```

```bash
pip install numpy==1.23.5 --upgrade
```

6. After completing all installation tasks, run the test script "test_emdm_prompt.py" and input a random activity description. The program will automatically download the pre-trained model of CLIP. If the program generates activity visualization animations normally, it proves that the environment configuration is successful.

```bash
python test_emdm_prompt.py
```
<img alt="image" src="https://github.com/user-attachments/assets/0a08730c-b138-4c91-97a1-4cbf27bd8cb2" />

### 🔹 Part 5: Run the RadHARAGT

Now get back to the MATLAB. In the unzipped GitHub project directory, run the following command. Now feel free to use the app!

```matlab
run("RadHARAGT_GUI.m");
```

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7534d6da-325e-4622-beab-70b106a65522" />
