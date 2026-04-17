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

---

## ✨ II. Core Highlights

<table>
  <tr align="center">
    <td width="50%">
      <br>
      <h3>🧠 1. Efficient Text-to-Motion</h3>
      <p>A local Large Language Model (LLM) interprets semantic prompts, driving a deep Motion Diffusion Model (MDM) to synthesize 3D kinematic trajectories, refined by advanced mathematical smoothing filters.</p>
      <br>
    </td>
    <td width="50%">
      <br>
      <h3>📡 2. Comprehensive Radar Physics</h3>
      <p>Strictly formulates complex electromagnetic wave phenomena including ground multipath reflections, multi-order internal wall reverberations, and target-wall room reverberations.</p>
      <br>
    </td>
  </tr>
  <tr align="center">
    <td>
      <br>
      <h3>⚡ 3. Ultra-Fast Inference</h3>
      <p>A lightweight deep CNN architecture (RHFNet) utilizing partial convolution modules. It requires merely <b>2.65M parameters</b> while achieving an incredibly rapid inference time of <b>0.054s</b> per sample.</p>
      <br>
    </td>
    <td>
      <br>
      <h3>💻 4. Highly Portable Deployment</h3>
      <p>No dependency on expensive server clusters or cloud APIs. The entire simulation agent can be deployed and run locally on a standard laptop equipped with a single discrete graphics card.</p>
      <br>
    </td>
  </tr>
</table>

### 📌 Abstract Information
* **Paper Title:** RadHARAGT: A Portable Agent for Radar-Based Human Activity Simulation
* **Journal:** Submitted to *IEEE Internet of Things Journal*
* **Author Email:** [JoeyBG@126.com](mailto:JoeyBG@126.com)

---

## 🛠️ III. How to Install

All source code in the repository is well-structured, extensively commented, and thoroughly debugged. With **MATLAB R2025b** or later installed, the scripts are designed to be executed smoothly.

### 🔹 Part 1: Generating Simulated Radar Data
1. **Clone & Setup:** Download the whole repository and unzip. Add the entire repository to the MATLAB search path.
2. **Environment Configuration:** Ensure your local environment supports the required local LLM (we strictly recommend using `gemma4:26b a4b` for optimal prompt refinement).
3. **Motion Generation:** Enter the `RadHARAGT_Agent` folder. Run the following scripts strictly in sequence:
   - `Text_to_Motion_Pipeline.m`
   - `Kinematic_Smoothing_Truncation.m`
4. **Physics Simulation:** Once the motion tensor source is generated, execute `Comprehensive_Radar_Physics_Engine.m` to compute the multipath effects and generate finalized RTMs and DTMs.

<div align="center">
  <img src="https://github.com/user-attachments/assets/your-matlab-workspace-image1" width="75%" alt="MATLAB Workspace Setup">
  <p><i>Expected MATLAB Workspace after successful execution.</i></p>
</div>

### 🔹 Part 2: Training and Evaluating RHFNet
1. **Toolbox Check:** Open `Home` -> `Add-Ons` -> `Explore Add-Ons` to install any required deep learning toolboxes for CNN training.
2. **Run Inference:** Enter the `RHFNet_Model` folder and run `RHFNet_Training_and_Evaluation.m`. 
3. **Results:** When all the partial convolution-based models are trained and validated, the learning curves and confusion matrices will be generated automatically.

---

## ⚠️ IV. Important Notes

> 🖥️ **1. Environment Issues** <br>
> The project primarily consists of pure MATLAB code executed locally. For neural network training and MDM inference, an NVIDIA GPU (e.g., RTX 3060 OC or better) is recommended. If you have any questions about configuring the local environment, feel free to email me.

> 🧬 **2. Algorithm Design Issues** <br>
> By comprehensively formulating strict physical laws of spreading attenuation, wall penetration, and room reverberation, the proposed agent transcends simplified free-space simulators. Our RHFNet consistently outperforms multiple state-of-the-art architectures in both free-space and through-the-wall scenarios.

> 🔒 **3. Copyright & Usage Rights** <br>
> ⭐ Considering intellectual property and the hard work of our team members, this work open-sources the simulation agent code and network model structures for **learning and academic purposes only**. Any direct use for paper submissions, patents, or commercialization must receive our explicit consent! ⭐

<div align="center">
  <p><i>If you find this repository helpful, please consider citing our paper and giving this repo a </i> 🌟</p>
</div>
