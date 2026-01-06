# Fine-Grained Food Image Classification using Transfer Learning with Uncertainty-aware inference
**Overview**

This project explores fine-grained food image classification using transfer learning on the Food-101 dataset. A pretrained ResNet-18 model is progressively adapted through staged training, followed by uncertainty-aware inference to improve prediction reliability. Rather than focusing solely on maximizing raw accuracy, the project emphasizes confidence-aware decision-making by explicitly handling ambiguous predictions.

**Dataset**

- Food-101 dataset
- 101 food categories with real-world visual variability
- Images were accessed using KaggleHub during experimentation

**Methodology**

The project is structured into three clearly defined stages:

Stage 1 — Frozen Backbone Baseline

- ResNet-18 pretrained on ImageNet
- All backbone layers frozen
- Used as a sanity check to validate the data pipeline and training setup

Stage 2 — Partial Fine-Tuning

- Final residual block (layer4) unfrozen
- Adam optimizer with OneCycleLR scheduling
- Batch normalization layers kept in evaluation mode
- Achieved ~70.6% validation accuracy under forced prediction

Stage 3 — Uncertainty-Aware Inference

- Monte Carlo Dropout applied at inference time
- Predictive uncertainty estimated via multiple stochastic forward passes
- Confidence-based decision rules used to accept or defer predictions
- Focus shifted from raw accuracy to reliability and coverage

**Results Summary**

| Stage | Accuracy | Coverage | Defer Rate |
|------|----------|----------|------------|
| Stage 1 (Frozen) | ~53% | 100% | 0% |
| Stage 2 (Partial Fine-Tuning) | 70.6% | 100% | 0% |
| Stage 3 (Selective Inference) | **98.8%*** | **91.9%** | **8.1%** |


* Stage 3 accuracy is computed only on accepted (non-deferred) predictions and is not directly comparable to forced prediction accuracy.

**Key Insights**

- Fine-grained classification performance improves significantly by adapting higher-level semantic features.
- A large fraction of misclassifications originate from a relatively small set of ambiguous samples.
- Explicitly deferring low-confidence predictions leads to substantially higher reliability without increasing model complexity.
- Uncertainty-aware inference provides a practical framework for deployment-ready machine learning systems.

**Model Weights**

Due to repository restrictions, trained model weights (.pth files) are not included in this repository.
All reported results were obtained using a partially fine-tuned ResNet-18 model trained according to the methodology described above. The full training and evaluation pipeline is reproducible using the provided notebook.

**Tools & Frameworks**

- PyTorch
- Torchvision
- KaggleHub
- Google Colab

**Conclusion**

This project demonstrates that combining transfer learning with uncertainty-aware inference enables not only strong performance but also more trustworthy decision-making. By shifting the focus from forced predictions to selective, confidence-based inference, the system better reflects real-world deployment requirements where abstaining from uncertain predictions is often preferable to being confidently wrong.
