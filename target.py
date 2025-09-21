import torch

x = torch.randn(5)

X = torch.fft.fft(x)

re = X.real
im = X.imag

# log magnitude
log_mag = 0.5 * torch.log(re * re + im * im)

# phase
phase = torch.atan2(im, re)

# squashed phase
eps = 1e-7
squashed_phase = torch.atanh(torch.clamp(phase / torch.pi, -1 + eps, 1 - eps))
