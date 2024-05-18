import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from utils import *
import torchaudio
import soundfile as sf
import argparse
import librosa    

def enhance_one_track(
    model, audio_path, saved_dir, cut_len, n_fft=200, hop=100, save_tracks=False):
    name = os.path.split(audio_path)[-1]
    nump_noise, sr = librosa.load(audio_path,sr=16000)# Downsample 44.1kHz to 8kHz
    noisy = torch.from_numpy(nump_noise)
    noisy = noisy.unsqueeze(0)
    if noisy.shape[0] == 2:
        noisy=stereo_to_mono(noisy)
    sr == 16000
    noisy = noisy.cuda()

    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True, return_complex=True
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_spec_uncompress = torch.complex(est_spec_uncompress[..., 0], est_spec_uncompress[..., 1])
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).cuda(),
        onesided=True
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].detach().cpu().numpy()

    assert len(est_audio) == length
    if save_tracks:
        saved_path = os.path.join(saved_dir, name)
        sf.write(saved_path, est_audio, sr)

    return est_audio, length

def resample_to_16000(audio, current_sample_rate):
    resampler = T.Resample(orig_freq=current_sample_rate, new_freq=16000)
    resampled_audio = resampler(audio)
    
    # Ajuster la longueur de l'audio pour compenser le changement de fréquence d'échantillonnage
    orig_length = audio.shape[1]
    new_length = resampled_audio.shape[1]
    ratio = orig_length / new_length
    resampled_audio = torch.nn.functional.interpolate(resampled_audio.unsqueeze(0), scale_factor=ratio, mode='linear').squeeze(0)
    return resampled_audio

def stereo_to_mono(audio):
    mono_audio = (audio[0, :] + audio[1, :]) / 2.
    mono_audio = mono_audio.unsqueeze(0)
    return mono_audio

def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def power_compress(x):
    real = x.real
    imag = x.imag
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
