import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import mne
from autoreject import AutoReject

# Python file because jupyter doesn't work with the interactive plots
#Loads a dataset from a CSV file into an mne raw eeg object

columns = [1,2,3,4,5,6,7,8]

with open(r"c:\Users\epicl\Documents\Extracurriculars\QMIND\eeg\datasets\datasets\datasets\1_MB\1_MB_1.csv") as f:
    rawarr = np.loadtxt(f,usecols = columns,skiprows = 1,delimiter=',')

# Convert from microvolts to volts
rawarr = rawarr / (1_000_000)

print(rawarr)

channel_names = ['Fs2','Fs1','C4','C3','T6','T5','P4','P3']
channel_types = 'eeg'
sample_rate = 250

inf = mne.create_info(ch_names=channel_names,ch_types=channel_types,sfreq=sample_rate)

raw = mne.io.RawArray(np.transpose(rawarr), info=inf,first_samp=100)

# Filter settings
low_cut = 0.1 
ica_low_cut = 1.0 

hi_cut  = 30

# ICA settings
seed = 42
ica_n_components = .999

# EOG Channel names
# Assume channels are close enough to the eyes to act as EOG channels
EOG_ch_names = ['Fs1','Fs2']

# Normalize, filter, and reject artefacts from data
def clean(raw):
    raw.load_data()

    # Normalize the input data

    # Regular filter in the 0.1-30hz range
    raw = raw.filter(low_cut, hi_cut)

    # ICA based artefact rejection
    run_ICA(raw,use_autoreject=False)


def run_ICA(data, use_autoreject=False): 

    # Heavy highpass for ICA training data
    train_data = raw.copy().filter(ica_low_cut, None)

    ica = mne.preprocessing.ICA(n_components=ica_n_components,
                                random_state=seed,
                                )

    # Autoreject epoch extraction
    if use_autoreject:
        train_data = do_autoreject(train_data)
    
    ica.fit(train_data)

    # Use EOG signal to eliminate eye blink
    
    eog_indices, _ = ica.find_bads_eog(raw,EOG_ch_names)
    ica.exclude = eog_indices

    # Try to find muscle artefacts

    muscle_indices, _ = ica.find_bads_muscle(raw)
    ica.exclude.extend(muscle_indices)


    # Reconstruct data
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    raw.plot(clipping=None)
    plt.show()

    reconst_raw.plot(clipping=None)
    plt.show()

# Applies autoreject to prevent large artefacts from affecting the ICA training
def do_autoreject(train_data):
    # Break data into 1 s epochs
        tstep = 1.0
        events_ica = mne.make_fixed_length_events(train_data, duration=tstep)
        epochs_ica = mne.Epochs(train_data, events_ica,
                                tmin=0.0, tmax=tstep,
                                baseline=None,
                                preload=True)
        

        
        ar = AutoReject(n_interpolate=[1, 2, 4],
                    random_state=42,
                    picks=mne.pick_types(epochs_ica.info, 
                                        eeg=True,
                                        eog=False
                                        ),
                    n_jobs=-1, 
                    verbose=False
                    )

        ar.fit(epochs_ica)

        reject_log = ar.get_reject_log(epochs_ica)
        return epochs_ica[~reject_log.bad_epochs]

clean(raw)