# Noise Reduction with Neural Networks

Welcome to the GitHub repository of our Master 1 engineering project at Faculté Polytechnique de Mons. We evaluate state-of-the-art open-source noise reduction techniques using neural networks.

## Algorithms Comparison

In this project, we compare two algorithms:

- CMGAN (2022): [GitHub Repository](https://github.com/ruizhecao96/CMGAN/tree/main/src/tools)
  - Using a conformer-based metric generative adversarial network (CMGAN)
- Facebook Denoiser (2020): [GitHub Repository](https://github.com/facebookresearch/denoiser?tab=readme-ov-file)
  - Using a conformer-based metric generative adversarial network (CMGAN)

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

## Écouter des Extraits Audio

### echo_test

1. echo_original.wav
<audio controls>
  <source src="comparaison sample/echo_test/1.echo_original.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

2. echo_cmgan.wav
<audio controls>
  <source src="comparaison sample/echo_test/echo_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

3. echo_fb.wav
<audio controls>
  <source src="comparaison sample/echo_test/echo_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

4. echo_noise&echo_supp_nvidia.m4a
<audio controls>
  <source src="comparaison sample/echo_test/echo_noise&echo_supp_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

5. echo_only_noise_sup_nvidia.m4a
<audio controls>
  <source src="comparaison sample/echo_test/echo_only_noise_sup_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

6. echo_OpenVino.wav
<audio controls>
  <source src="comparaison sample/echo_test/echo_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

### man+backgrnd._discussion_noise

1. dischomme -5(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme -5(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

2. dischomme -5_cmgan.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme -5_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

3. dischomme -5_fb.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme -5_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

4. dischomme -5_nvidia.m4a
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme -5_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

5. dischomme -5_OpenVino.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme -5_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

6. dischomme 0(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 0(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

7. dischomme 0_cmgan.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 0_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

8. dischomme 0_fb.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 0_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

9. dischomme 0_nvidia.m4a
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 0_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

10. dischomme 0_OpenVino.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 0_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

11. dischomme 7(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 7(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

12. dischomme 7_cmgan.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 7_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

13. dischomme 7_fb.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 7_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

14. dischomme 7_nvidia.m4a
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 7_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

15. dischomme 7_OpenVino.wav
<audio controls>
  <source src="comparaison sample/man+backgrnd._discussion_noise/dischomme 7_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

### man+storm_noise

1. stormhomme -7(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme -7(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

2. stormhomme -7_fb.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme -7_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

3. stormhomme -7_nvidia.m4a
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme -7_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

4. stormhomme -7_OpenVino.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme -7_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

5. stormhomme 0(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 0(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

6. stormhomme 0_cmgan.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 0_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

7. stormhomme 0_fb.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 0_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

8. stormhomme 0_nvidia.m4a
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 0_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

9. stormhomme 0_OpenVino.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 0_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

10. stormhomme 7(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 7(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

11. stormhomme 7_fb.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 7_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

12. stormhomme 7_nvidia.m4a
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 7_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

13. stormhomme 7_OpenVino.wav
<audio controls>
  <source src="comparaison sample/man+storm_noise/stormhomme 7_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

### woman+bar_env_noise

1. barfemme -2_cmgan.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme -2_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

2. barfemme -7_(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme -7_(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

3. barfemme -7_fb.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme -7_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

4. barfemme -7_nvidia.m4a
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme -7_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

5. barfemme -7_OpenVino.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme -7_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

6. barfemme 3_(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 3_(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

7. barfemme 3_cmgan.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 3_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

8. barfemme 3_fb.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 3_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

9. barfemme 3_nvidia.m4a
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 3_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

10. barfemme 3_OpenVino.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 3_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

11. barfemme 7_(snr)_noised.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 7_(snr)_noised.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

12. barfemme 7_fb.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 7_fb.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

13. barfemme 7_nvidia.m4a
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 7_nvidia.m4a" type="audio/m4a">
  Votre navigateur ne supporte pas la balise audio.
</audio>

14. barfemme 7_OpenVino.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme 7_OpenVino.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

15. barfemme7_cmgan.wav
<audio controls>
  <source src="comparaison sample/woman+bar_env_noise/barfemme7_cmgan.wav" type="audio/wav">
  Votre navigateur ne supporte pas la balise audio.
</audio>

