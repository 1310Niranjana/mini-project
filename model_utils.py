import cv2
import numpy as np
import torch
import torch.nn as nn
import timm


def fft_preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    log_magnitude = np.log1p(magnitude)
    norm = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())
    norm_3ch = np.stack([norm, norm, norm], axis=-1)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (norm_3ch - mean) / std


class DualStreamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.fft_branch = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(2560, 1)

    def forward(self, rgb_input, fft_input):
        rgb_feat = self.rgb_branch(rgb_input)
        fft_feat = self.fft_branch(fft_input)
        fused = torch.cat([rgb_feat, fft_feat], dim=1)
        return self.classifier(self.dropout(fused))


def get_frame_probability(image, model_path='model_baseline_epoch7.pth'):
    """
    Takes a face image (numpy array, RGB format) and returns
    probability of being REAL as a float between 0 and 1.
    Above 0.5 = REAL, Below 0.5 = FAKE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = DualStreamModel().to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()
    img = cv2.resize(image, (224, 224))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_norm = (img / 255.0 - mean) / std
    rgb_tensor = torch.tensor(rgb_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    fft_img = fft_preprocess(img)
    fft_tensor = torch.tensor(fft_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = loaded_model(rgb_tensor, fft_tensor)
        probability = torch.sigmoid(logit).item()
    return probability
