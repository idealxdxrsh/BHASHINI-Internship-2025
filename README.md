# Bhashini Internship 2025: Speech & Language Technologies for Indian Languages

This repository documents the key projects and contributions made during my Summer 2025 internship with the **Digital India Bhashini Mission**. The primary focus was on advancing Automatic Speech Recognition (ASR) and Language Identification (LID) systems for diverse Indian languages.

## 🚀 Core Contributions

### 1. Multilingual Language Identification (LID)
- **Developed** an end-to-end Language Identification system for 10 Indian languages using a Convolutional Neural Network (CNN) architecture on Mel-Frequency Cepstral Coefficients (MFCCs).
- **Trained and evaluated** the model on large-scale datasets, including Common Voice, to ensure robustness.
- **Deployed** the model via a Gradio-based web interface for real-time inference from user-recorded audio.
- <img width="656" height="792" alt="image" src="https://github.com/user-attachments/assets/bdfbd676-9cb7-46ca-bc4c-21878363401f" />


### 2. Automatic Speech Recognition (ASR) Enhancement
- **Fine-tuned** state-of-the-art ASR models, including OpenAI's Whisper and AI4Bharat's IndicWav2Vec, on custom datasets for Indian languages.
- **Improved pronunciation modeling** by integrating `IndicG2P` for phoneme-level analysis and lexicon generation.
- **Enhanced system performance** for low-resource languages through lexicon augmentation techniques.
- **Implemented** a Voice Activity Detection (VAD) pipeline for preprocessing noisy, real-world audio, improving ASR accuracy.

## 🛠️ Technology Stack

- **Core Technologies:** Python, PyTorch, HuggingFace Transformers, `librosa`
- **Model Training:** `Slurm` for distributed training on High-Performance Computing (HPC) clusters.
- **Deployment & UI:** Gradio, Flask, Jupyter Notebooks.

## 📁 Repository Structure
