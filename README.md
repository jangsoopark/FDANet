# FDANet


**Note:** ~~Implementations~~ and pretrained weights are currently being prepared and will be available soon.

--- 

Official PyTorch implementation of  
**"Lightweight Attention Mechanism with Feature Difference for Efficient Change Detection in Remote Sensing,"** 
*IEEE Geoscience and Remote Sensing Letters*, 2025.  
**Authors:** J. Park, E. Lee, J. Lee, S. J. Oh, and D. Sim  
**DOI:** [10.1109/LGRS.2025.3633179](https://doi.org/10.1109/LGRS.2025.3633179)

---

## Overview

**FDANet** introduces a **Feature Difference Attention Module (FDAM)** designed for fast and efficient change detection in remote sensing imagery.  
FDAM achieves its efficiency through two key **dimensionality-reduction strategies**:

1. It operates on the **absolute difference** between bi-temporal features, eliminating redundant information from concatenation-based fusion and **halving the attention input dimensionality**.  
2. It applies **spatial and channel pooling** to obtain **compact yet informative representations**, enabling attention computation in a **reduced feature space**.

With this design, FDANet minimizes computational overhead while maintaining competitive accuracy.

<!-- <p align="center">
  <img src="assets/fdanet_architecture.png" width="640">
  <br>
  <em>Overview of the proposed Feature Difference Attention Module (FDAM).</em>
</p> -->

---

## Table of Contents
- [Overview](#overview)
- [Change Detection](#change-detection)
  - [Datasets](#datasets)
- [Model](#model)
- [Training](#training)
- [Experiments](#experiments)
- [Acknowledgements](#acknowledgements)
- [Reference](#references)
- [Citation](#citation)
- [Contact](#contact)

---

## Change Detection

### Datasets

FDANet is evaluated on three standard change detection benchmarks:

| Dataset | Image Size | #Pairs | Resolution | Description |
|----------|-------------|---------|-------------|--------------|
| **WHU-CD** [1] | 512×512 | 222 | 0.3 m | Urban building changes |
| **LEVIR-CD** [2] | 1024×1024 | 637 | 0.5 m | Large-scale urban changes |
| **SYSU-CD**  [3] | 256×256 | 20,000 | 0.5 m | Rural/urban mixed scenes |

Please refer to the official dataset repositories for download links.  ``

---

## Model

FDANet uses a Siamese backbone (e.g., VGG11BN, ResNet18, DenseNet121) to extract features from bi-temporal images (I₁, I₂).
The proposed Feature Difference Attention Module (FDAM) computes their absolute difference and sequentially applies spatial and channel attention to produce compact, noise-suppressed change representations.

```math
\begin{equation}
\begin{aligned}
f_d &= \left| f_{T_1} - f_{T_2} \right|, \\[3pt]
M_c &= \text{concat}\!\left[\,\mu_s(f_d),\, M_s(f_d)\,\right], \\[3pt]
M_s &= \text{concat}\!\left[\,\mu_c(f_d),\, M_c(f_d)\,\right], \\[3pt]
\tilde{M}_c &= \phi_c(M_c; \theta_c), \quad
\tilde{M}_s = \phi_s(M_s; \theta_s), \\[3pt]
Y &= \left( \tilde{M}_c \otimes \tilde{M}_s \right) \odot f_d
\end{aligned}
\end{equation}



```

FDAM sequentially applies
- Spatial attention: generated from channel-wise representative pooling
- Channel attention: generated from spatial-wise representative pooling

The refinement process in Eq. (1) can be implemented using CBAM-style attention functions,
where channel and spatial refinement follow the same operational form as in CBAM,
but are driven by representative statistics from feature differences rather than raw feature activations.

Main source files:
```
src/model/neck.py
src/model/network.py
```

---

## Training

### 1. Prerequisites
- **NVIDIA GPU** with CUDA support  
- **NVIDIA Container Toolkit (nvidia-docker2)** installed  
  > Follow the official installation guide: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Docker** (recommended ≥ 24.0)  
  - *Experiments in this paper were conducted using:*  
    **Docker version 28.1.1, build 4eba377**

### 2. Environment Setup
```bash
cd docker
docker build . -t change-detection

cd ../scripts
./run-docker.sh
```

### 3. Configuration
- The "dataset-path" field must be specified according to your environment.=
- All configuration files are stored in experiments/config/<backbone>/.

Example configuration file:
`experiments/config/resnet18/ResNet18-FDAM-LEVIR-schedule-T_0-5-T_mult-2.json`

```json
{
  "name": "ResNet18-FDAM-LEVIR-schedule-T_0-5-T_mult-2",
  "device": "cuda:0",
  "backbone": "ResNet18",
  "dataset": "levir-cd",
  "dataset-path": "/workspace/dataset/Levir-CD-baseline/",
  "pretrained-backbone": true,
  "num_classes": 1,
  "epochs": 50,
  "batch-size": 8,
  "learning-rate": 5e-4,
  "learning-rate-scheduler": true,
  "T_0": 5,
  "T_mult": 2,
  "weight-decay": 0.0025
}

```

### 4. Train
```bash
# CWD: /workspace
cd scripts
./train-all.sh

```

### 5. Evaluate
```bash
# CWD: /workspace
cd scripts
./run-jupyter.sh # The external jupyter port is set to 8889

# Jupyter Notebook for evaluation will be available soon

```

### 6. Metrics
- **Precision**, **Recall**, **F1-score**, **IoU**
- **MACs** measured using a **modified version of [`ptflops`](https://github.com/sovrasov/flops-counter.pytorch)** optimized for change detection tasks.  
  Implementations will be available soon.


---

## Experiments

| **Dataset** | **Backbone** | **F1 (%)** | **MACs (G)** | **Params (M)** | **Time (ms)** |
|:------------:|:-------------:|:-----------:|:--------------:|:----------------:|:--------------:|
| **WHU-CD** | ResNet18 | 92.58 | 19.15 | 11.28 | 7.83 |
| **WHU-CD** | VGG11 | 93.24 | 19.82 | 9.38 | 7.83 |
| **WHU-CD** | VGG13 | 93.51 | 29.55 | 9.56 | 10.17 |
| **WHU-CD** | VGG16 | 93.22 | 40.44 | 14.88 | 12.33 |
| **──────────** | **──────────** | **──────────** | **──────────** | **──────────** | **──────────** |
| **LEVIR-CD** | ResNet18 | 91.46 | 19.15 | 11.28 | 7.83 |
| **LEVIR-CD** | VGG11 | 91.45 | 19.82 | 9.38 | 7.83 |
| **LEVIR-CD** | VGG13 | 91.71 | 29.55 | 9.56 | 10.17 |
| **LEVIR-CD** | VGG16 | 91.86 | 40.44 | 14.88 | 12.33 |
| **──────────** | **──────────** | **──────────** | **──────────** | **──────────** | **──────────** |
| **SYSU-CD** | ResNet18 | 82.79 | 19.15 | 11.28 | 7.83 |
| **SYSU-CD** | VGG11 | 81.89 | 19.82 | 9.38 | 7.83 |
| **SYSU-CD** | VGG13 | 82.11 | 29.55 | 9.56 | 10.17 |
| **SYSU-CD** | VGG16 | 82.17 | 40.44 | 14.88 | 12.33 |


---

## Acknowledgements

We gratefully acknowledge the open-source implementations of 
[CGNet[4]](https://github.com/ChengxiHAN/CGNet-CD), 
[Open-CD[5]](https://github.com/likyoo/open-cd), and 
[CBAM[6]](https://github.com/Jongchan/attention-module), 
which have contributed to the development and validation of this work.


---

## References

- [1] S. Ji *et al.*, "Fully convolutional neural networks for multisource building extraction from an open aerial and satellite imagery dataset,"  
  *IEEE Transactions on Geoscience and Remote Sensing*, vol. 57, no. 1, pp. 574–586, 2019.  
  DOI: [10.1109/TGRS.2018.2858817](https://doi.org/10.1109/TGRS.2018.2858817)

- [2] H. Chen and Z. Shi, "A spatial-temporal attention-based method and new dataset for remote sensing image change detection,"  
  *Remote Sensing*, vol. 12, no. 10, p. 1662, 2020.  
  DOI: [10.3390/rs12101662](https://doi.org/10.3390/rs12101662)

- [3] Q. Shi, M. Liu, S. Li, X. Liu, F. Wang, and L. Zhang,  
  "A deeply supervised attention metric-based network and an open aerial image dataset for remote sensing change detection,"  
  *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, no. 1, pp. 1–16, 2021.  
  DOI: [10.1109/TGRS.2021.3085870](https://doi.org/10.1109/TGRS.2021.3085870)

- [4] C. Han *et al.*, "Change guiding network: Incorporating change prior to guide change detection in remote sensing imagery,"  
  *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 16, pp. 8395–8407, 2023.  
  DOI: [10.1109/JSTARS.2023.3310208](https://doi.org/10.1109/JSTARS.2023.3310208)

- [5] S. Fang, K. Li, and Z. Li, "Changer: Feature Interaction is What You Need for Change Detection,"  
  *IEEE Transactions on Geoscience and Remote Sensing*, vol. 61, pp. 1–11, 2023.  
  DOI: [10.1109/TGRS.2023.3277496](https://doi.org/10.1109/TGRS.2023.3277496)

- [6] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon,  
  "CBAM: Convolutional Block Attention Module,"  
  *Proceedings of the European Conference on Computer Vision (ECCV)*, Sept. 2018.  
  [https://github.com/Jongchan/attention-module](https://github.com/Jongchan/attention-module)

---

---

## Citation
If you find FDANet useful for your work please cite:

```bibtex
@ARTICLE{11248878,
  author={Park, Jangsoo and Lee, EunSeong and Lee, Jongseok and Oh, Seoung-Jun and Sim, Donggyu},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Lightweight Attention Mechanism With Feature Differences for Efficient Change Detection in Remote Sensing}, 
  year={2026},
  volume={23},
  number={},
  pages={1-5},
  keywords={Feature extraction;Accuracy;Barium;Attention mechanisms;Computational efficiency;Remote sensing;Computer architecture;Distortion;Computational modeling;Spatial resolution;Attention mechanisms;bitemporal remote sensing (RS) images;change detection algorithm;convolutional neural networks},
  doi={10.1109/LGRS.2025.3633179}}

```

---

## Contact

For questions, comments, or collaboration inquiries, please contact:  
📧 **jangsoopark@kw.ac.kr**

---

> *"Efficient change detection begins with informative differences."*  
> — *FDANet (2025)*
