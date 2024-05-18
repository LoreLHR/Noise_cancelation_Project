import argparse
import io
import os
import time
import wave

import numpy as np
import pyaudio
import sounddevice as sd
import soundfile as sf
import torchaudio

from pydub import AudioSegment
from natsort import natsorted

from models import generator
from tools.compute_metrics import compute_metrics
from utils import *

import librosa
import streamlit as st
import torch.autograd.profiler as profiler
torch.cuda.empty_cache()

@torch.no_grad()
def play_audio(audio_data, sample_rate=16000):
    st.audio(audio_data, format="audio/wav", start_time=0, sample_rate=sample_rate)
    


def add_noise_to_audio(speech, noise, snr):
    # Calculer le nombre de répétitions nécessaires pour couvrir la longueur du discours
    num_repetitions = int(np.ceil(len(speech) / len(noise)))

    # Répéter le bruit
    repeated_noise = np.tile(noise, num_repetitions)

    # Tronquer le bruit pour correspondre à la longueur du discours
    repeated_noise = repeated_noise[:len(speech)]

    # Calculer la puissance du signal et du bruit
    power_speech = np.sum(speech ** 2) / len(speech)
    power_noise = np.sum(repeated_noise ** 2) / len(repeated_noise)

    # Vérifier si la puissance du bruit est non nulle
    if power_noise > 0:
        # Calculer la puissance du bruit pour atteindre le SNR désiré
        target_power_noise = power_speech / (10 ** (snr / 10))
        scale_factor = np.sqrt(target_power_noise / power_noise)

        if not np.isnan(scale_factor):
            # Effectuer la multiplication
            adjusted_noise_array = repeated_noise * scale_factor
        
            # Additionner le discours et le bruit
            result = speech + adjusted_noise_array

        else:
            st.write('it is Nan')
            # Si scale_factor est NaN, renvoyer le discours seul
            result = speech
    else:
        st.write('power_noise is zero or negative')
        # Si la puissance du bruit est nulle ou négative, renvoyer le discours seul
        result = speech

    return result

# Streamlit app
st.set_page_config(layout="wide")
st.title("Demo CMGAN GPU")
st.write('gpu memory : ',torch.cuda.memory_allocated())
option = st.sidebar.radio("**Choose an option :**", ("Use your audio", "Record sound", "use demo file"))
memory_allocated= []
if option == "Record sound":
    duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=10, value=3)
    duration=4 
    sample_rate = 16000
    st.info(f"Click the button to start recording. Recording duration: {duration} seconds.")
    if st.sidebar.button('**Start recording and use CMGAN**'):

    
        # record_button = st.button("Start Recording")
    
        recorded_audio = None
    
    # if record_button:
        # Record audio using sounddevice
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
        sd.wait()
        
        # st.success("Recording completed!")
        
        recorded_audio = audio_data.flatten()
        
        # if recorded_audio is not None:
        # Save recorded audio to a temporary file
        # temp_file_path = save_audio_to_temp_file(recorded_audio)
        rec_file_path = "rec.wav"
        sf.write(rec_file_path, recorded_audio, samplerate=16000)
        # Play the recorded audio
        st.markdown("**Play Recorded Audio**")
        play_audio(recorded_audio)
        
        start_time = time.time()
        model_path = './best_ckpt/ckpt'
        save_tracks = True
        saved_dir = './audio_enregistré'
        n_fft = 400
        
        model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        model.eval().cuda()
        rec_audio, length = enhance_one_track(model, rec_file_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks)
        # Afficher le résultat de la suppression de bruit
        st.markdown("**Audio cleaned with CMGAN**")
        st.audio(rec_audio, format="audio/wav", start_time=0, sample_rate=16000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Execution time : {elapsed_time:.2f} seconds")
if option=="Use your audio":
    
    audio_file = st.file_uploader("**Choisissez le fichier audio (format WAV)**", type=["wav"])
    
    if audio_file is not None:
        audio, sr_audio = librosa.load(io.BytesIO(audio_file.read()), sr=16000)
        st.markdown("**Play audio**")
        play_audio(audio)       
        result_file_path = "result.wav"
        sf.write(result_file_path, audio, samplerate=16000)

        if st.button('Utiliser CMGAN'):
   
            st.write('gpu memory : ',torch.cuda.memory_allocated())
            start_time = time.time()
            model_path = './best_ckpt/ckpt'
            save_tracks = True
            saved_dir = './audio_généré'
            n_fft = 400

            model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
            model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
            model.eval().cuda()

            # Démarrez le profiler
            
            # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
            est_audio, length = enhance_one_track(model, result_file_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks)
            
            # Affichez le résultat de la suppression de bruit
            st.markdown("**Audio cleaned with CMGAN**")
            st.audio(est_audio, format="audio/wav", start_time=0, sample_rate=16000)
            
            # Calcul du temps d'exécution
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Temps d'exécution : {elapsed_time:.2f} secondes")    
        
                    
                    # Affichez le profilage
                    # st.write('cpu/gpu time and % : ')
                    # st.write(prof.key_averages().table(sort_by="cuda_time_total"))
                    # st.write('gpu memory : ',torch.cuda.memory_allocated())    
    
else:

    
    option_speech = ['speech femme','speech homme','speech homme 4s','echo','5s','6s','7s','9s']
    option_noise = ['applause', 'bar', 'course','discussion', 'forest', 'parc','party', 'storm']
    speech_file = st.sidebar.selectbox('**Choose clean speech**', option_speech)
    noise_file = st.sidebar.selectbox('**Choose an noise**', option_noise)
    
    speech_path='./datawav/'+speech_file+'.wav'
    noise_path='./datawav/'+noise_file+'.wav'
    
    snr = st.sidebar.slider("**Choose the signal noise ration (SNR)**", -20, 20, 7)
    
    if speech_file is not None and noise_file is not None:

            
            speech, sr_speech = librosa.load(speech_path, sr=16000)
            noise, sr_noise = librosa.load(noise_path, sr=16000)
            
            st.markdown("**Play clean speech**")
            play_audio(speech)
            
            st.markdown("**Play noise**")
            play_audio(noise)
            # Ajouter du bruit au discours
            result = add_noise_to_audio(speech, noise, snr)
            
            # Enregistrer le résultat dans un fichier temporaire
            result_file_path = "result.wav"
            sf.write(result_file_path, result, samplerate=16000)
            
            # Afficher le résultat
            st.markdown("**Noised audio**")
            st.audio(result_file_path, format="audio/wav", start_time=0)
           
            if st.button('Utiliser CMGAN'):
       
                st.write('gpu memory : ',torch.cuda.memory_allocated())
                start_time = time.time()
                model_path = './best_ckpt/ckpt'
                save_tracks = True
                saved_dir = './audio_généré'
                n_fft = 400
    
                model = generator.TSCNet(num_channel=64, num_features=n_fft // 2 + 1)
                model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
                model.eval().cuda()
    
                # Démarrez le profiler
                
                # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                est_audio, length = enhance_one_track(model, result_file_path, saved_dir, 16000 * 16, n_fft, n_fft // 4, save_tracks)
                
                # Affichez le résultat de la suppression de bruit
                st.markdown("**Audio cleaned with CMGAN**")
                st.audio(est_audio, format="audio/wav", start_time=0, sample_rate=16000)
                
                # Calcul du temps d'exécution
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Temps d'exécution : {elapsed_time:.2f} secondes")
    
                
                # Affichez le profilage
                # st.write('cpu/gpu time and % : ')
                # st.write(prof.key_averages().table(sort_by="cuda_time_total"))
                # st.write('gpu memory : ',torch.cuda.memory_allocated())
            
torch.cuda.empty_cache()