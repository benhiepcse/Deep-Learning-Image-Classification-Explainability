# H01 — Deep Learning Image Classification & Explainability

**An end-to-end Computer Vision project for image classification and explainable AI using PyTorch, CNN, ResNet, and Grad-CAM.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer_Vision-Deep_Learning-blue)
![Status](https://img.shields.io/badge/Status-In_Development-yellow)

## 1. Project Overview

H01 is an individual Deep Learning and Computer Vision project focused on building a complete image classification and explainability pipeline using the CIFAR-10 dataset.

The project involves implementing, training, evaluating, and comparing Convolutional Neural Networks (CNNs) and Residual Networks (ResNets).

It also integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize image regions that influence model predictions.

The main objective is to develop practical skills in Deep Learning engineering, reproducible experimentation, model evaluation, and Explainable AI (XAI).

## 2. Project Information

| Attribute | Description |
|---|---|
| Project ID | H01 |
| Project Name | Deep Learning Image Classification & Explainability |
| Project Type | Individual Project |
| Developer | Hiệp |
| Institution | Ho Chi Minh City University of Technology (HCMUT) |
| Domain | Computer Vision / Deep Learning |
| Dataset | CIFAR-10 |
| Framework | PyTorch |
| Architectures | CNN, ResNet |
| Explainability | Grad-CAM |
| Development Period | October 19 – November 15, 2026 |
| Duration | 4 Weeks |
| Status | In Development |

## 3. Project Objectives

- Develop a complete CIFAR-10 data preprocessing pipeline.
- Implement and train a CNN baseline from scratch.
- Implement and train a ResNet classifier.
- Investigate the impact of data augmentation.
- Compare CNN and ResNet under consistent experimental conditions.
- Evaluate classification performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Apply Grad-CAM to visualize model predictions.
- Build an image classification inference demo.
- Ensure experiment reproducibility through configurations, checkpoints, and documented results.

## 4. Dataset

The project uses the CIFAR-10 dataset, containing 60,000 RGB images across 10 classes.

| Property | Value |
|---|---|
| Total Images | 60,000 |
| Training Images | 50,000 |
| Test Images | 10,000 |
| Resolution | 32 × 32 |
| Channels | 3 (RGB) |
| Classes | 10 |
| Task | Single-label Multiclass Classification |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The official training set will be divided into training and validation subsets. The official test set will be reserved for final evaluation.

Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## 5. System Architecture

The planned pipeline consists of six main stages:

1. **Data Preparation:** CIFAR-10 loading, preprocessing, normalization, and augmentation.
2. **Model Development:** CNN baseline and ResNet classifier.
3. **Training:** Forward propagation, loss computation, backpropagation, optimization, and checkpoint management.
4. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and error analysis.
5. **Explainability:** Grad-CAM heatmap generation and visualization.
6. **Inference:** Image input, class prediction, and explanation visualization.

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep Learning framework |
| Torchvision | Datasets, transformations, and models |
| NumPy | Numerical computation |
| Scikit-learn | Classification evaluation |
| Matplotlib | Visualization |
| Pillow / OpenCV | Image processing |
| pytest | Automated testing |
| Git & GitHub | Version control |
| TensorBoard | Optional experiment visualization |
| Gradio | Optional interactive demo |

Specific dependency versions will be documented after environment validation.

## 7. Repository Structure

The proposed repository structure is:

```text
H01-image-classification/
├── configs/
├── data/
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── explainability/
├── scripts/
├── tests/
├── notebooks/
├── reports/
├── assets/
├── README.md
└── requirements.txt
```

The structure may evolve during implementation.

Datasets and large model checkpoints should not be committed directly to standard Git history.

## 8. Planned Experiments

| ID | Architecture | Augmentation | Initialization |
|---|---|---|---|
| E01 | CNN | Basic preprocessing | Random |
| E02 | CNN | Random crop + horizontal flip | Random |
| E03 | ResNet | Basic preprocessing | Random |
| E04 | ResNet | Random crop + horizontal flip | Random |
| E05 (Optional) | ResNet | Pretrained-compatible transforms | Pretrained |

The experiments will investigate how model architecture and data augmentation affect classification performance.

## 9. Evaluation Metrics

The project will evaluate models using:

- Classification Accuracy
- Precision and Recall
- Macro-F1 and Per-class F1
- Confusion Matrix
- Training and Validation Loss
- Model Parameter Count
- Inference Latency

Experimental results will be documented after training and evaluation.

## 10. Development Roadmap

| Week | Period | Main Tasks |
|---|---|---|
| Week 1 | Oct 19–25 | Environment setup, CIFAR-10 pipeline, CNN baseline |
| Week 2 | Oct 26 – Nov 1 | ResNet implementation, checkpointing, initial comparison |
| Week 3 | Nov 2–8 | Augmentation experiments, evaluation, confusion matrix |
| Week 4 | Nov 9–15 | Grad-CAM, inference demo, testing, documentation |

## 11. Expected Deliverables

- [ ] Reproducible CIFAR-10 data pipeline
- [ ] Trained CNN baseline
- [ ] Trained ResNet classifier
- [ ] Training and validation pipeline
- [ ] Checkpoint management
- [ ] Data augmentation experiments
- [ ] Classification evaluation report
- [ ] Confusion matrix and learning curves
- [ ] Grad-CAM visualizations
- [ ] Image classification inference demo
- [ ] Technical report
- [ ] Demo video

## 12. Future Improvements

Potential extensions include:

- Transfer Learning with pretrained models
- Multiple-seed experiments
- Model calibration analysis
- Grad-CAM++ and Integrated Gradients
- Evaluation under distribution shift
- Interactive web deployment
- Extension toward object detection and robot perception

These extensions are outside the core four-week implementation scope.

## 13. References

1. [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)
2. [Torchvision Documentation](https://docs.pytorch.org/vision/stable/index.html)
3. [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
4. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
5. [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
6. [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 14. Author

**Hiệp**

Ho Chi Minh City University of Technology (HCMUT)

*Individual project — independently developed and maintained.*

---

*This project is developed for educational, experimental, and AI engineering portfolio purposes.*

## 15. Local Development Environment

See [Environment Setup](docs/setup/SETUP.md) for installation instructions and S01-01 evidence.
