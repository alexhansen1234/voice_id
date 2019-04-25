import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot( mfcc, title='MFCC', figsize=(10,4) )
    plt.figure(figsize)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()
