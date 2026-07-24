Deepfake Detection with Compression Robustness Using Dual-Stream EfficientNet-B0

Motivated by: Lagsoun et al. 2025, IEEE Access (research question, not architecture — see Methodology below)

Status: PREPROCESSING COMPLETE — MODEL TRAINING IN PROGRESS (RQ3 pending)
Team Responsibilities
Person 1: Data collection and preprocessing ✅ DONE
Person 2: Model development and FFT pipeline — baseline (original-only) trained ✅; compression-aware retraining (RQ3) 🔲 in progress
Person 3: Evaluation and Streamlit deployment — 🔲 pending Person 2's compression-aware weights
Dataset
Primary: Celeb-DF v2
Real videos: 408
Fake videos: 795
Total videos: 1203
Random seed: 42
Split type: Video-level (no leakage)
Split	Videos
Train	841
Validation	180
Test	182
Compression Settings
Level	Resolution	CRF	Codec
original	native	-	-
crf23	1280x720	23	libx264
crf28	854x480	28	libx264
crf35	640x360	35	libx264

Pixel format: yuv420p

Face Detection
Tool: MTCNN (facenet-pytorch)
Output size: 224x224
Total faces: 90,042
Failed detections: 43
Model Architecture
Dual-stream EfficientNet-B0
Stream 1: RGB frames
Stream 2: FFT (frequency domain)
Fusion: torch.cat() → 2560-D
Classifier: Dropout(0.3) → Linear(2560,1) → raw logit
Loss (training): BCEWithLogitsLoss (sigmoid applied internally by loss function, not in model forward())
Inference: sigmoid applied manually inside get_frame_probability(), then mean-aggregated across frames, 0.5 threshold
Methodology — Architecture Justification
Relationship to Prior Work

This project is motivated by Lagsoun et al. (2025, IEEE Access), which investigates deepfake detection robustness under compression. Our implemented architecture is not a direct implementation of their method. Lagsoun et al. use a Db2 wavelet transform, ConvLSTM, Conv3D, and a custom ResNet backbone. Our team independently designed a dual-stream EfficientNet-B0 architecture (RGB stream + FFT-based frequency stream) as an alternative approach to the same underlying research question: does compression-aware training improve deepfake detection robustness against real-world platform re-encoding (e.g., WhatsApp, Instagram)?

Architecture Design Rationale

Our dual-stream design shares structural similarity with SpectraNet (Jayarathne et al., arXiv:2511.19187, 2025), which also combines an EfficientNet backbone with an FFT-derived frequency branch, feature concatenation, dropout, a linear classification head, and BCEWithLogitsLoss. We cite SpectraNet as the closest published precedent to our design, noting two key differences:

Backbone size: SpectraNet uses EfficientNet-B6; we use EfficientNet-B0.
Frequency branch design: our FFT pipeline (grayscale → FFT → fftshift → magnitude → log1p → min-max normalization → 3-channel replication → ImageNet normalization) differs from SpectraNet's frequency branch implementation.
Why EfficientNet-B0 (not B6 or larger)

Deliberate design decision, not a resource limitation:

Dataset size: Celeb-DF v2 (1,203 videos) is small relative to what larger backbones typically require to avoid overfitting, especially with a dual-stream architecture that doubles parameter count.
Dual-stream compute cost: running two EfficientNet backbones concurrently roughly doubles training/inference cost versus a single-stream model; B0 keeps this tractable on Kaggle's T4 GPU constraints.
Lightweight/deployment angle: supports the downstream goal of practical, low-latency deployment (Streamlit demo), aligned with real-world use cases.
Summary Statement (for paper/abstract use)

"While our work is motivated by Lagsoun et al.'s (2025) investigation of compression-robust deepfake detection, our dual-stream EfficientNet-B0 architecture was independently developed by our team. It shares conceptual similarity with SpectraNet (Jayarathne et al., 2025) in its use of EfficientNet backbones, FFT-based frequency features, and feature-level fusion, but differs in backbone scale and frequency branch design. Our contribution focuses on evaluating this architecture's robustness specifically under real-world compression conditions (CRF23/28/35), which is not the primary focus of SpectraNet's original work."

FFT Pipeline (Person 2)

All steps must be identical across training, validation, testing, and Streamlit inference:

Convert to grayscale
Apply fft2
Apply fftshift
Take absolute value
Apply log1p
Min-max normalize
Replicate to 3 channels
Apply ImageNet normalization
Interface Contract (Person 2 → Person 3)

Person 2 must expose:

Function: get_frame_probability(image)
Returns: single float probability (0-1)
Person 3 never accesses model internals
Research Question 3 — Compression-Aware Training (OPEN)
Status: In progress (Person 2)
Baseline result (locked): Original-only training, epoch 7, 95.58% val accuracy on clean test set
Needed for paper (2x2 eval matrix):
Training data	Clean test acc	Compressed test acc
Original only (baseline)	95.58% ✅	pending 🔲
Original + CRF23/28/35 (compression-aware)	pending 🔲	pending 🔲
Why this matters: demonstrates whether compression-aware training closes a generalization gap that plain training doesn't — this is the paper's core contribution, not the raw baseline accuracy number.
Blocked on: Person 2 retraining on mixed-compression data + evaluating both checkpoints (baseline and compression-aware) on both clean and compressed test sets.
Key Decisions (LOCKED)
split_manifest.csv is the single source of truth
Video-level split must happen before any processing
Original variant must be retained alongside compressed variants
Failed face detections go to failed_faces/ not discarded
MTCNN imported from facenet_pytorch (not mtcnn package)
Model outputs raw logits (not sigmoid) during training; sigmoid applied only at inference
Architecture is team's own synthesis, citing SpectraNet as closest precedent — not a direct implementation of Lagsoun et al. (see Methodology section)
Processed Dataset

Kaggle: https://www.kaggle.com/datasets/niranjana1310/celeb-df-v2-faces 
GitHub: https://github.com/1310Niranjana/mini-project Branch: preprocessing
