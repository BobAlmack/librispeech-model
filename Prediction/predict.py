import sys
import numpy as np
import librosa
from python_speech_features.base import mfcc, logfbank
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

labels = ['G', 'Z', 'CH', 'spn', 'B', 'HH', 'F', 'IH', 'IY', 'T', 'sil', 'ER', 'AY', 'R', 'P', 'AH', 'K', 'L', 'JH', 'V', 'OW', 'AA', 'DH', 'ZH', 'NG', 'M', 'SH', 'AW', 'EY', 'OY', 'S', 'UH', 'TH', 'AE', 'N', 'UW', 'AO', 'W', 'D', 'EH', 'Y', 'ctcblank']

standardize_mean = np.array([-5.142197960524675, -3.0658824937612983, -4.394900549247351, 4.500875536035409, -8.237126621164151, -6.7270542546203185, -7.909034834546234, -5.442603387265692, -1.78064662090358, -2.132269858979562, -2.1090506337831436, -3.671890884142077, -5.886241451390704, 0.0014287884484971282, 0.03762966035724957, 0.00401205124631255, 0.035783838497134526, -0.00016446614933610391, 0.0032843761720164908, -0.010556775606317622, -0.0003950246781448321, 0.0053072178684198616, 0.003727176181386583, 0.0070855193625138846, 0.0016567866517965557, 0.0008203597496883395])
standardize_std = np.array([2.80893491696378, 17.945136199171298, 15.126761978292803, 16.107906918739534, 15.617254113282547, 16.356214125480182, 15.69782869410956, 14.752865305322047, 14.315721324884391, 13.43469283765519, 12.431550259931372, 11.750155079045516, 2.921894088540466, 0.9023553989081563, 3.695689820479721, 3.7627496549938786, 3.9092403275983503, 4.333275740201201, 4.665948810618357, 4.794957269533566, 4.803007501228222, 4.863545456877994, 4.777643610744265, 4.633617497261899, 4.427851717422686, 0.978435860018187])

def ctc_lambda_loss(args):
    y_pred, labels, wavlen, lablen = args
    labels = K.argmax(labels)
    return K.ctc_batch_cost(labels, y_pred, wavlen, lablen)

def create_model():
    input_wav = Input((None, 26))

    x = input_wav

    x = Bidirectional(LSTM(200, return_sequences=True), merge_mode='sum')(x)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(100, return_sequences=True), merge_mode='sum')(x)
    x = LayerNormalization()(x)
    x = Bidirectional(LSTM(100, return_sequences=True), merge_mode='sum')(x)
    x = LayerNormalization()(x)

    x = TimeDistributed(Dense(42, activation="softmax"))(x)

    m = Model(input_wav, x)
    m.summary()
    return m

m = create_model()
m.load_weights("model.h5")

def get_mfcc(x):
    y = np.concatenate([mfcc(x, numcep=12, winlen=0.01, winstep=0.005), logfbank(x, nfilt=1, winlen=0.01, winstep=0.005)], axis=-1)
    derivatives = []
    previousf = np.zeros((13,))
    for i in range(len(y)):
        if (i + 1) == len(y):
            nextf = np.zeros((13,))
        else:
            nextf = y[i + 1]
        derivatives.append(((nextf - previousf) / 2).reshape((1, 13)))
        previousf = y[i]
    derivatives = np.concatenate(derivatives, axis=0)
    y = np.concatenate([y, derivatives], axis=1)
    return y

N, sr = librosa.load(sys.argv[1], sr=16000)
N, _ = librosa.effects.trim(N, top_db=35, frame_length=160, hop_length=80)
mi = (get_mfcc(N).reshape((1, -1, 26)) - standardize_mean) / standardize_std

pred = m.predict(mi)[0]

values = ["BLANK"]
for value in pred:
    p = value.argmax()
    if p == 41:
        values.append("BLANK")
    else:
        if values[-1] != p:
            values.append(p)

print([labels[x] for x in values if x != "BLANK"])