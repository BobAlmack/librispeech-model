# librispeech-model

This repository includes:

`predict.py` which will take an aduio file (.wav or .aifc) as the first argument and return as list of ARPABET phonemes.
The predict script uses the weights defined in the pretrained model `model.h5`

If using your own voice recording ensure: 
- your words aren't too quick
- the words they're slightly drawn out
- you are in a very quiet environment with no noise
- a clear microphone (Headphones like the WF-1000XM3s, for example, won't work. MacBook Pro 16-inch microphones are exceptional for this.)
- each recording can only include one word or pseudo word 

Two sample voice recording has been included in /SampleTestAudio

Reference to convert word to ARPABET phoneme list: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
