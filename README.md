<img width="487" height="427" alt="image" src="https://github.com/user-attachments/assets/f5c5ab61-1336-4f77-8b32-c41afa17123c" /># AFF-Net
Adaptive Feature Fusion Network for Efficient Medical Image Segmentation
Abstract
Medical image segmentation faces the inherent accuracy‑efficiency trade‑off challenge: state‑of‑the‑art models achieve high segmentation accuracy but bring heavy computational overhead, while traditional lightweight networks suffer from degraded segmentation performance.
We propose AFF‑Net (Adaptive Feature Fusion Network), a lightweight segmentation framework built upon UniRepLKNet backbone under adapter‑based fine‑tuning for small‑sample medical segmentation.
Two task‑specific plug‑and‑play components are designed:
MSPA (Multi‑Scale Perception Adapter): Embedded into backbone for multi‑scale context aggregation under adapter fine‑tuning constraints.
LMCAB (Lightweight Multi‑scale Convolutional Attention Block): Integrated inside the modified EMCAD decoder to refine local anatomical boundaries.
BEA‑U (Bilinear Edge‑Attention Upsample): Edge‑aware upsampling module for precise organ boundary reconstruction.
Extensive experiments are conducted on Synapse and ACDC datasets. AFF‑Net achieves competitive segmentation performance with low parameters and computational cost, yielding favourable accuracy‑efficiency balance for resource‑constrained clinical scenarios.
A simplified high‑level model schematic diagram is also provided in this repository for quick understanding of network topology.
📌 Environment
bash
python >= 3.8
pytorch >= 1.13.0
torchvision
monai
tqdm
numpy
scipy
SimpleITK
Install dependencies:
bash
pip install -r requirements.txt
📂 Dataset Preparation
We use two public medical CT/MR segmentation datasets:
Synapse (Multi‑Organ CT Segmentation)
ACDC (Cardiac MR Segmentation)
