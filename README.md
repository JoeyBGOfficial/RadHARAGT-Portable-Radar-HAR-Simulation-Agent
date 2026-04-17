<div align="center">

<img src="https://via.placeholder.com/1200x250/0f172a/ffffff?text=RadHARAGT:+A+Portable+Agent+for+Radar-Based+HAR" alt="Project Banner" width="100%" />

# RadHARAGT: A Portable Agent for Radar-Based Human Activity Simulation

<p>
  <a href="https://www.semanticscholar.org/author/Weicheng-Gao/2051685234"><img src="https://img.shields.io/badge/Semantic_Scholar-464EB8?style=for-the-badge&logo=semanticscholar&logoColor=white" alt="Semantic Scholar"/></a>
  <a href="https://joeybgofficial.github.io/"><img src="https://img.shields.io/badge/Personal_Homepage-252525?style=for-the-badge&logo=github&logoColor=white" alt="Personal Homepage"/></a>
  <a href="https://ieeexplore.ieee.org/author/37089574449"><img src="https://img.shields.io/badge/IEEE-00629B?style=for-the-badge&logo=ieee&logoColor=white" alt="IEEE"/></a>
  <a href="https://radar.bit.edu.cn/index.htm"><img src="https://img.shields.io/badge/Team_Website-005A3C?style=for-the-badge&logo=rss&logoColor=white" alt="Team Website"/></a>
</p>

*An innovative paradigm mapping natural language directly into physical radar representations for Human Activity Recognition (HAR).*

</div>

---

## 📑 Table of Contents
- [I. Introduction & Overview](#i-introduction--overview)
- [II. Core Highlights](#ii-core-highlights)
- [III. How to Reproduce](#iii-how-to-reproduce)
- [IV. Important Notes](#iv-important-notes)

---

## 💡 I. Introduction & Overview

> **Through-the-Wall Radar (TWR) Human Activity Recognition (HAR) represents a cutting-edge field in pattern recognition research.** However, the generation of high-fidelity datasets is conventionally restricted by labor-intensive physical collection processes. 

To address this data scarcity challenge, we propose **RadHARAGT**, a portable agent that automates the synthesis of realistic human radar echo and image datasets directly from natural language semantic descriptions. By integrating a local large language model (LLM), an efficient motion diffusion model (MDM), and a comprehensive radar point-scatterer physics engine, this field can now achieve text-to-radar mapping that strictly conforms to complex electromagnetic wave propagation phenomena.

Furthermore, we propose a lightweight deep CNN architecture, **RHFNet**, featuring a partial convolution module for rapid HAR inference. **This work introduces an innovative paradigm for dataset synthesis, and uniquely, the entire agent can be smoothly deployed on a standard laptop with a single discrete graphics card.** We fully trust our peer community and welcome the use of our open-source code for one-click verification, ensuring the reproducibility of the reported results in our paper.

<div align="center">
  <img src="https://github.com/user-attachments/assets/your-overview-image-link-here" width="80%" alt="Architecture Overview">
  <br>
  <img src="https://github.com/user-attachments/assets/your-visualization-image-link-here" width="80%" alt="Visualizations">
</div>

---

## ✨ II. Core Highlights

<table>
  <tr>
    <td width="50%">
      <h3>🧠 1. Efficient Text-to-Motion Generation</h3>
      A local Large Language Model (LLM) interprets semantic prompts, driving a deep Motion Diffusion Model (MDM) to synthesize 3D kinematic trajectories, refined by advanced mathematical smoothing filters.
    </td>
    <td width="50%">
      <h3>📡 2. Comprehensive Radar Physics Engine</h3>
      Strictly formulates complex electromagnetic wave phenomena including ground multipath reflections, multi-order internal wall reverberations, and target-wall room reverberations to generate realistic RTMs and DTMs.
    </td>
  </tr>
  <tr>
    <td>
      <h3>⚡ 3. Ultra-Fast HAR Inference (RHFNet)</h3>
      A proposed lightweight deep CNN architecture utilizing partial convolution modules. It requires merely 2.65M parameters while achieving an exceptionally rapid inference time of 0.054s per sample.
    </td>
    <td>
      <h3>💻 4. Highly Portable Deployment</h3>
      No dependency on expensive server clusters or cloud APIs. The entire simulation agent can be deployed and run locally on a standard laptop equipped with a single discrete graphics card.
    </td>
  </tr>
</table>

**Abstract Information:**
* **Paper Title:** RadHARAGT: A Portable Agent for Radar-Based Human Activity Simulation
* **Journal:** Submitted to *IEEE Internet of Things Journal*
* **Author Email:** JoeyBG@126.com

---

## 🛠️ III. How to Reproduce

All source code in the repository is well-structured, extensively commented, and thoroughly debugged. With **MATLAB R2025b** or later installed, the scripts are designed to be executed smoothly.

### 🔹 Part 1: Generating Simulated Radar Data
1. **Clone & Setup:** Download the whole repository and unzip. Add the entire repository to the MATLAB search path.
2. **Environment Configuration:** Ensure your local environment supports the required local LLM (we strictly recommend using `gemma4:26b a4b` for optimal prompt refinement based on our ablation study).
3. **Motion Generation:** Enter the `RadHARAGT_Agent` folder. Run the following scripts strictly in sequence:
   - `Text_to_Motion_Pipeline.m`
   - `Kinematic_Smoothing_Truncation.m`
4. **Physics Simulation:** Once the motion tensor source is generated, execute `Comprehensive_Radar_Physics_Engine.m` to compute the multipath effects and generate finalized RTMs and DTMs.

<div align="center">
  <img src="https://github.com/user-attachments/assets/your-matlab-workspace-image1" width="70%" alt="MATLAB Workspace Setup">
  <p><em>Expected MATLAB Workspace after successful execution.</em></p>
</div>

### 🔹 Part 2: Training and Evaluating RHFNet
1. **Toolbox Check:** Open `Home` -> `Add-Ons` -> `Explore Add-Ons` to install any required deep learning toolboxes for CNN training.
2. **Run Inference:** Enter the `RHFNet_Model` folder and run `RHFNet_Training_and_Evaluation.m`. 
3. **Results:** When all the partial convolution-based models are trained and validated, the learning curves and confusion matrices will be generated automatically.

---

## ⚠️ IV. Important Notes

> **1. Environment Issues:** The project primarily consists of pure MATLAB code executed locally. For neural network training and MDM inference, an NVIDIA GPU (e.g., RTX 3060 OC or better) is recommended. If you have any questions about configuring the local environment, feel free to email me.

> **2. Algorithm Design Issues:** By comprehensively formulating strict physical laws of spreading attenuation, wall penetration, and room reverberation, the proposed agent transcends simplified free-space simulators. Our RHFNet consistently outperforms multiple state-of-the-art architectures in both free-space and through-the-wall scenarios.

> **3. Copyright & Usage Rights:** > ⭐ Considering intellectual property and the hard work of our team members, this work open-sources the simulation agent code and network model structures for **learning and academic purposes only**. Any direct use for paper submissions, patents, or commercialization must receive our explicit consent! ⭐

<div align="center">
  <p><i>If you find this repository helpful, please consider citing our paper and giving this repo a ⭐!</i></p>
</div>
