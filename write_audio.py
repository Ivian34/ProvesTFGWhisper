import torch
import numpy as np
import scipy

data = torch.load("Audios/data.pt", weights_only=False)
data = data['data']
audio = data[90]['audio_sentence']
rate = data[0]['audio_rate']


scipy.io.wavfile.write("bark_out.wav", rate=rate, data=audio)