<div align="center">

<img width="250" height="250" alt="RadHARAGT_Logo" src="https://github.com/user-attachments/assets/67be9103-d363-4696-a701-c0049a9e373d" />

<img src="https://readme-typing-svg.demolab.com?font=Orbitron&weight=700&size=45&pause=500&color=95949A&center=true&vCenter=true&width=1000&height=100&lines=RadHARAGT;A+Portable+Agent+for+Radar-Based+HAR" alt="Dynamic Title" />
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

The implementation of large language models (LLMs) has become extremely popular. As researchers in radar-based human activity recognition (HAR), We are eager to develop an agent that can directly generate radar images through only textual prompts. Although Nano Banana has now developed to the level where it can generate radar range profiles and Doppler profiles, the features on the generated images are completely inconsistent with physical laws.

As shown in the figure below, a system with only traditional architecture is limited in its intelligence. A system with only AI architecture is limited in its physical interpretability. Therefore, a hybrid architecture design that combines both traditional and AI elements is optimal. RadHARAGT was born in response to this need.

<div align="center">
<img width="4150" height="7305" alt="Overall" src="https://github.com/user-attachments/assets/4947eabf-5cdd-493c-8876-44ad22ed0fda" />
</div>
Fig. 1. The overall design idea and architecture of the proposed RadHARAGT.
<br><br>

Furthermore, we propose a lightweight deep network architecture: RHFNet, which uses a partial convolution module for rapid HAR inference. This work introduces an innovative framework for data synthesis, and uniquely, **the entire agent and the recognition network model can be smoothly deployed on a standard laptop or work station with a single discrete graphics card**.

Table I. Minimum requirements and inference speed on different hardware platforms of the proposed RadHARAGT.
| Minimum VRAM | Suggested VRAM | Run on NVIDIA RTX 3060 | Run on NVIDIA RTX 4070 Laptop |
| :---: | :---: | :---: | :---: |
| 8 GB | 12 GB | 254 s / sample | 208 s / sample |

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
      <h3>🧠 1. Efficient Text-to-Motion</h3>
      <p>A local LLM interprets semantic prompts (gemma4:26b a4b for suggested VRAM, qwen3-vl:4b for minimum VRAM), driving an efficient Motion Diffusion Model (MDM) to synthesize 3D kinematic trajectories, refined by advanced smoothing and truncation filters.</p>
      <br>
    </td>
    <td width="50%">
      <h3>📡 2. Comprehensive Radar Physics</h3>
      <p>Strictly formulates complex electromagnetic wave phenomena including wall attenuation and multipath effects. Besies, range-time maps (RTMs) and Doppler-time maps (DTMs) that look exactly like real-world measurements are generated.</p>
      <br>
    </td>
  </tr>
  <tr align="center">
    <td>
      <h3>⚡ 3. Ultra-Fast Inference</h3>
      <p>A lightweight deep network architecture "RHFNet" utilizing partial convolution modules. It requires merely 2.65 M parameters while achieving an incredibly rapid inference time of 0.054 s per sample.</p>
      <br>
    </td>
    <td>
      <h3>💻 4. Highly Portable Deployment</h3>
      <p>No dependency on expensive server clusters or cloud APIs. The entire agent can be deployed on a standard laptop or workstation equipped with a single discrete graphics card.</p>
      <br>
    </td>
  </tr>
</table>

---

## 🛠️ III. How to Install

All source code in the repository is well-structured, extensively commented, and thoroughly debugged. Here's the overall list of platforms we need to install and run the proposed RadHARAGT:

> (1) MATLAB R2025a or Later.<br>
> (2) Python 3.10.13.<br>
> (3) CUDA 12.1.<br>
> (4) Ollama After Updated in April 2026.

Strictly follow the installation instructions below step by step, and you can use our agent normally:

### 🔹 Part 1: Prepare the MATLAB Environments
1. Install MATLAB R2025a or a later version on your computer, you need to install all the toolboxes. Actually, our code can also run on versions R2022a~2024b, but it's not very stable. If you have already installed MATLAB, skip this step.

### 🔹 Part 2: Prepare the Ollama Environments
1. Download the latest windows version of Ollama here: https://ollama.com/download. Install Ollama in your computer. After the installation, open Ollama, and sign in.
2. In the model options bar, select the LLM you want. It is suggested to select qwen3-vl:4b. Of course, you can also choose other models to ensure that your computer resources can run smoothly:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9c711ae5-ee38-4704-8eb4-a031aeecd8b1" />

3. After selected your favorite model, just prompt some content, and Ollama will automatically download the model. After the model download is complete, **do not exit Ollama and keep it running in the background**.

### 🔹 Part 2: Prepare the Ollama Environments
---

## ⚠️ IV. Important Notes

🖥️ **1. Environment Issues** <br>
The project primarily consists of MATLAB and python code executed locally. Suggested MATLAB version: R2025a+. Required python version: 3.10.13. Required CUDA version: 12.1.

🧬 **2. Algorithm Design Issues** <br>
By comprehensively formulating strict physical laws of wall penetration and multipath reverberation, the proposed agent transcends simplified free-space simulators. However, only single-person simulation without any indoor furnitures is supported. We are trying to catch up with these improvements! But we still want to thank to the remarkable work that inspired us:
> Y. Zhou, M. López-Benítez, L. Yu and Y. Yue, "Text2Doppler: Generating Radar Micro–Doppler Signatures for Human Activity Recognition via Textual Descriptions," in IEEE Sensors Letters, vol. 8, no. 10, pp. 1-4, Oct. 2024.

🔒 **3. Copyright & Usage Rights** <br>
⭐ Considering intellectual property and the hard work of our team members, this work open-sources the simulation agent code and network model structures for **learning and academic purposes only**. Any direct use for paper submissions, patents, or commercialization must receive our explicit consent! ⭐

<div align="center">
  <p><i>If you find this repository helpful, please consider citing our paper and giving this repo a </i> "🌟". Really appreciated!</p>
</div>
