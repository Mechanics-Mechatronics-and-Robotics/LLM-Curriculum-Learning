# Project: Leveraging LLMs for Curriculum Scheduling in Noisy or Multi-Objective Learning

This repository was developed using methodologies suggested by the ChatGPT (OpenAI, 2025) and DeepSeek (DeepSeek, 2024) language models.

## Problem Statement

Large Language Models (LLMs) have demonstrated emergent abilities in understanding data semantics, ranking difficulty, and generating optimization heuristics. This project explores the use of LLMs to dynamically generate **curriculum learning schedules** for training neural networks—either on noisy datasets or on multi-objective losses (e.g., in physics-informed neural networks or vision-language models).

By automating the selection and ordering of training samples, and/or adaptively tuning the importance of multiple loss terms, LLMs can facilitate faster convergence, better generalization, and more robust training under real-world, imperfect conditions.

## Keywords

- Curriculum Learning  
- Meta-Learning  
- Noisy Labels  
- Multimodal Loss Optimization  
- LLMs (e.g., GPT-4, DeepSeek, Grok, Mistral)  
- PINNs (Physics-Informed Neural Networks)  
- Vision-Language Models (VLMs)

## Possible Solution Paths

1. **LLM-Guided Curriculum for hyperparameter tuning**
   - Train the model  (e.g., logistic regression, SVM, shallow neural network) with default hyperparameters.
   - Usa a language model (e.g., GPT-4, Claude, or open-source LLMs like Llama-3) as an **LLM Backend** to analyze metrics and propose hyperparameters.
   - **Orchestrator**: Manages the training loop, feeds data to the LLM, and applies hyperparameters.
   - **Tracking**: Tools like Weights & Biases, MLflow, or TensorBoard to log metrics.

3. **LLM-Guided Curriculum for Noisy Classification**  
   - Use an LLM (e.g., GPT-4) to rank CIFAR-10N samples by label confidence or semantic clarity.
   - Build a simple ResNet and compare training with/without LLM-informed curriculum scheduling.
   - Alternative: use a sample weighting scheme learned via LLM analysis.

4. **Adaptive Loss Weighting in Multi-objective Training**  
   - In a model with multiple loss components (e.g., PINN, reconstruction + classification, vision + text analysis), ask the LLM to suggest a dynamic weighting strategy.
   - Evaluate convergence speed and final performance versus static weights or heuristic schedules.

5. **LLM as Meta-Optimizer Assistant**  
   - Use LLM to suggest learning rate schedules, batch composition strategies, or fine-tuning sequences based on real-time logs (loss curves, validation error).
   - Compare against standard strategies like cosine annealing or ReduceLROnPlateau.

## End Goal

Publication of a short research article or position paper (~4 pages) at a [NeurIPS 2025](https://neurips.cc/) Workshop (TBA).

## Milestones

1. Week 1–2: Literature review + dataset setup (CIFAR-10N, or a simple PINN model, or IMDB dataset)
2. Week 3–4: Implementation of baseline model and curriculum schedule logic
3. Week 5–6: Integration of LLMs and logging analysis
4. Week 7–8: Experiments, visualization, and writing the paper

## Datasets

- [CIFAR-10N (label noise version)](https://paperswithcode.com/dataset/cifar-10n)
- [COCO](https://cocodataset.org/#home)
- PINN models are dataset free (usa a simple pinn model of a pendulum)

## Tools & Libraries

- PyTorch + Lightning
- OpenAI API or HuggingFace Transformers
- ClearML or Weights & Biases (for training monitoring)

## References

- Bengio et al. (2009), *Curriculum Learning*: https://doi.org/10.1145/1553374.1553380  
- Zhou et al. (2021), *Curriculum Learning: A Survey*: https://arxiv.org/abs/2101.10382  
- Swayamdipta et al. (2020), *Dataset Cartography*: https://arxiv.org/abs/2009.10795
- Hadi Pouransari et al. (2024), *Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum*: https://arxiv.org/abs/2405.13226
 
