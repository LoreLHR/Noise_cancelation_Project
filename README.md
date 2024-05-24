
# Noise Reduction with Neural Networks

Welcome to the GitHub repository of our Master 1 engineering project at Faculté Polytechnique de Mons. We evaluate state-of-the-art open-source noise reduction techniques using neural networks.

## Algorithms Comparison

In this project, we compare two algorithms:

- CMGAN (2022): [GitHub Repository](https://github.com/ruizhecao96/CMGAN/tree/main/src/tools)
  - Uses Conformer-based metric generative adversarial network (CMGAN).
- Facebook Denoiser (2020): [GitHub Repository](https://github.com/facebookresearch/denoiser?tab=readme-ov-file)
  - Uses U-Net-based encoder-decoder architecture with skip-connections.

## Audio Specifications

Both algorithms work on mono wave audio sampled at 16000 Hz.

## CUDA Usage with CMGAN

CMGAN utilizes CUDA to leverage the dedicated memory of the GPU and its massively parallel computing capabilities to accelerate specific tasks, allowing the CPU to handle more general operations. To easily install PyTorch with CUDA, we recommend following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

## Streamlit Demo for CMGAN

For CMGAN, we have developed a demonstration app using Streamlit. Here is a video demonstration:

[![Demo](https://img.youtube.com/vi/XmWqsCZmdX8/0.jpg)](https://www.youtube.com/watch?v=XmWqsCZmdX8)

### Usage Instructions:

To use this app:

- For CMGAN with Streamlit (same app as the video):
  - Navigate to `src_cmgan` (or src_cmgan_no_cuda) and run the following command:

    ```bash
    streamlit run streamlitCMGAN.py
    ```


## Using Facebook Denoiser

To use Facebook Denoiser with Python:

- Navigate to `src_fb` and run the following command:

  ```bash
  python -m denoiser.enhance --dns48 --noisy_dir=noise --out_dir=clean

For CMGAN and Facebook Denoiser, mono WAV audio sampled at 16,000 Hz is required. The Streamlit app can automatically convert any WAV audio to the required format. If not, you can use online tools like Online Convert.

Additionally, we provide comparisons for the same audio at the same SNR (Signal-to-Noise Ratio) between CMGAN, Facebook Denoiser, and other known denoisers like OpenVino (https://github.com/intel/openvino-plugins-ai-audacity/blob/main/README.md) and Nvidia Broadcast (https://www.nvidia.com/fr-be/geforce/broadcasting/broadcast-app/). It's worth noting that Nvidia Broadcast operates in real-time.


