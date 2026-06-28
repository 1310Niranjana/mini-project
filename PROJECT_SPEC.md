\# Deepfake Detection with Compression Robustness Using Dual-Stream EfficientNet-B0



Based on: Lagsoun et al. 2025, IEEE Access



\## Status: PREPROCESSING COMPLETE



\## Team Responsibilities



\- Person 1: Data collection and preprocessing ✅ DONE

\- Person 2: Model development and FFT pipeline

\- Person 3: Evaluation and Streamlit deployment



\## Dataset



\- Primary: Celeb-DF v2

\- Real videos: 408

\- Fake videos: 795

\- Total videos: 1203

\- Random seed: 42

\- Split type: Video-level (no leakage)



| Split | Videos |

|-------|--------|

| Train | 841 |

| Validation | 180 |

| Test | 182 |



\## Compression Settings



| Level | Resolution | CRF | Codec |

|-------|------------|-----|-------|

| original | native | - | - |

| crf23 | 1280x720 | 23 | libx264 |

| crf28 | 854x480 | 28 | libx264 |

| crf35 | 640x360 | 35 | libx264 |



Pixel format: yuv420p



\## Face Detection



\- Tool: MTCNN (facenet-pytorch)

\- Output size: 224x224

\- Total faces: 90,042

\- Failed detections: 43



\## Model Architecture



\- Dual-stream EfficientNet-B0

\- Stream 1: RGB frames

\- Stream 2: FFT (frequency domain)

\- Loss: BCEWithLogitsLoss

\- Dropout: 0.3

\- Inference: sigmoid



\## FFT Pipeline (Person 2)



All steps must be identical across training, validation, testing, and Streamlit inference:

1\. Convert to grayscale

2\. Apply fft2

3\. Apply fftshift

4\. Take absolute value

5\. Apply log1p

6\. Min-max normalize

7\. Replicate to 3 channels

8\. Apply ImageNet normalization



\## Interface Contract (Person 2 → Person 3)



Person 2 must expose:

\- Function: get\_frame\_probability(image)

\- Returns: single float probability (0-1)

\- Person 3 never accesses model internals



\## Key Decisions (LOCKED)



\- split\_manifest.csv is the single source of truth

\- Video-level split must happen before any processing

\- Original variant must be retained alongside compressed variants

\- Failed face detections go to failed\_faces/ not discarded

\- MTCNN imported from facenet\_pytorch (not mtcnn package)



\## Processed Dataset



Kaggle: https://www.kaggle.com/datasets/niranjana1310/celeb-df-v2-faces

GitHub: https://github.com/1310Niranjana/mini-project

Branch: preprocessing

