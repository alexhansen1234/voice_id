import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def plot( mfcc, title='MFCC', figsize=(10,4) ):
    plt.figure(figsize=figsize)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.xlim(0, 6.4)
    plt.show()

if __name__ == "__main__":
    n = np.load('data_cache/train_data.npy')
    plot(n[3][:][:])
