<div align="center">
  
<img width="260" height="260" alt="RadHARAGT_Logo" src="https://github.com/user-attachments/assets/10f6a669-a038-4dc7-be1f-53b2e643c97a" />

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=700&size=45&pause=500&color=A8A9B5&center=true&vCenter=true&width=1000&height=100&lines=RadHARAGT;A+Portable+Agent+for+Radar-Based+HAR" alt="Dynamic Title" />
<br>

<kbd>
  <a href="https://www.semanticscholar.org/author/Weicheng-Gao/2051685234">
    <img src="https://img.shields.io/badge/Semantic_Scholar-005A3C?&logo=semanticscholar&logoColor=white" alt="Semantic Scholar" height="30"/>
  </a>
</kbd>
&nbsp;&nbsp;
<kbd>
  <a href="https://joeybgofficial.github.io/">
    <img src="https://img.shields.io/badge/Personal_Homepage-252525?&logo=github&logoColor=white" alt="Personal Homepage" height="30"/>
  </a>
</kbd>
&nbsp;&nbsp;
<kbd>
  <a href="https://ieeexplore.ieee.org/author/37089574449">
    <img src="https://img.shields.io/badge/IEEE-00629B?&logo=ieee&logoColor=white" alt="IEEE" height="30"/>
  </a>
</kbd>
&nbsp;&nbsp;
<kbd>
  <a href="https://radar.bit.edu.cn/index.htm">
    <img src="https://img.shields.io/badge/Team_Website-990F4B?&logo=rss&logoColor=white" alt="Team Website" height="30"/>
  </a>
</kbd>
</div>

---

## 💡 I. Introduction & Overview

**This work truly achieves "text to Doppler", and more importantly, it can be deployed on a laptop.**

The implementation of large language models (LLMs) has become extremely popular. As researchers in radar-based human activity recognition (HAR), we are eager to develop an agent that can directly generate radar images through only textual prompts. Although Nano Banana and GPT Imagen have now developed to the level where it can generate radar range maps and Doppler maps, the features on the generated images are completely inconsistent with physical laws. RadHARAGT was born in response to this need.

https://github.com/user-attachments/assets/3df8bb93-4ab2-4a31-b7ee-87ef8fe9cbfd

Furthermore, we propose a lightweight deep network architecture: RHFNet, which uses a partial convolution module for rapid HAR inference.

The hardware requirements of RadHARAGT:

<div align="center">
  
| Minimum VRAM | Suggested VRAM | Run on NVIDIA RTX 3060 | Run on NVIDIA RTX 4060 Laptop |
| :---: | :---: | :---: | :---: |
| 8 GB | 12 GB | Average 254 s / sample | Average 208 s / sample |

</div>

You can also pay attention to our 1st-generation and 2nd-generation simulator if interested:

| RadHARSimulator V1: Model-Based Simulator | RadHARSimulator V2: Video to Doppler Generator |
| :---: | :---: |
| <img width="1024" alt="486687209-15860457-59a0-4e1a-9331-789ce891b373" src="https://github.com/user-attachments/assets/8b2d0faa-fd27-4d51-91ac-71cb3ff050d7" /> | <img width="1024" alt="507388290-6358d013-4a40-4e42-b2b7-df0b880295aa" src="https://github.com/user-attachments/assets/c1440f9f-d13d-4c94-a03e-16d472c64f6b" /> | 
| https://github.com/JoeyBGOfficial/RadHARSimulatorV1-Model-Based-FMCW-Radar-Human-Activity-Recognition-Simulator | https://github.com/JoeyBGOfficial/RadHARSimulatorV2-Video-to-Doppler-Generator |

### 📚 Paper Information
* **Paper Title:** RadHARAGT: A Portable Agent for Radar-Based Human Activity Simulation.
* **Journal References:**
> Submitted to IEEE Internet of Things Journal.
* **Link:**

---

## ✨ II. Core Highlights

<table>
  <tr align="center">
    <td width="50%">
      <h3>🧠 1. Multiple Human Target Support</h3>
      <p>A local LLM interprets semantic prompts (gemma4:26b a4b for suggested VRAM, gemma4:e4b for minimum VRAM), driving an improved multiple-target efficient Motion Diffusion Model (MDM) to synthesize 3D kinematic trajectories, refined by advanced smoothing and truncation filters.</p>
      <br>
    </td>
    <td width="50%">
      <h3>📡 2. MIMO Radar Support</h3>
      <p>Strictly formulates complex electromagnetic wave phenomena including wall attenuation and multipath effects of a MIMO radar system. Besides, range-time maps (RTMs) and Doppler-time maps (DTMs) that look exactly like real-world measurements are generated.</p>
      <br>
    </td>
  </tr>
  <tr align="center">
    <td>
      <h3>⚡ 3. Highly Portable Deployment</h3>
      <p>The agent requires only laptop-level hardware. Besides, a lightweight deep network architecture "RHFNet" utilizing partial convolution modules. It requires merely 2.65 M parameters while achieving an incredibly rapid inference time of 0.054 s per sample.</p>
      <br>
    </td>
    <td>
      <h3>💻 4. Indoor Layout Support</h3>
      <p>Common indoor static objects are supported. In addition, the proposed agent has established a strict adaptive obstacle avoidance system. There will be no overlap between human targets, human targets and static objects, or human targets and walls.</p>
      <br>
    </td>
  </tr>
</table>

---

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

---

## ⚠️ IV. Important Notes

🖥️ **1. Environment and Version Issues** <br>
The project primarily consists of MATLAB and Python code executed locally. Suggested MATLAB version: R2024a+, Python version: 3.10.13, and CUDA version: 12.1.1+.

| Date | Version Infomation | Details |
| :---: | :---: | :---: |
| 2026.3.27 | RadHARSimulator V3.0 (Main_Smulation_Rough.m) | Complete the basic architecture, including input understanding, activity generation, smoothing, truncation, and radar simulation. |
| 2026.4.18 | RadHARSimulator V3.1 (Main_Smulation_Complete.m, RadHARSimulatorV3.mlapp) | Improve the radar simulation physics engine to make it more realistic. Design a simulation agent for the GUI interface. |
| 2026.4.20 | RadHARSimulator V3.2 (Main_MIMO_Multiperson.m, RadHARAGT.mlapp) | Add MIMO radar system support, multi-person support, and simple indoor scatterer-based object supoort. Design the complete simulation GUI interface for the agent. |
| 2026.4.30 | RadHARSimulator V3.3 (RadHARAGT_Main_Script.m) | Writing RadHARAGT testing report and fix all the bugs. |
| 2026.5.6 | RadHARSimulator V3.4 (RadHARAGT_GUI.m) | Add contextual multi-turn dialogue support and design the GUI of the app. Private release. |

🧬 **2. Algorithm Design Issues** <br>
By comprehensively formulating strict physical laws of wall penetration and multipath reverberation, the proposed agent transcends simplified free-space simulators. However, only single-person simulation without any indoor furniture is supported. We are trying to catch up with these improvements! But we still want to thanks to the remarkable work that inspired us:

> Y. Zhou, M. López-Benítez, L. Yu and Y. Yue, "Text2Doppler: Generating Radar Micro–Doppler Signatures for Human Activity Recognition via Textual Descriptions," in IEEE Sensors Letters, vol. 8, no. 10, pp. 1-4, Oct. 2024.

🔒 **3. Copyright & Usage Rights** <br>
⭐ Considering intellectual property and the hard work of our team members, this work open-sources the simulation agent code and network model structures for **learning and academic purposes only**. Any direct use for paper submissions, patents, or commercialization must receive our explicit consent! ⭐

<div align="center">
  <p><i>If you find this repository helpful, please consider citing our paper and giving this repo a "🌟". Really appreciated!</i></p>
</div>
