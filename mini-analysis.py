#+++import modules
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, filedialog
from nptdms import TdmsFile
plt.rcParams['agg.path.chunksize'] = 1000 #for plotting optimization purposes
from lib import *


#+++define user functions



#+++define parameters
fs = 12.8e3 #sampling frequency
channel = 'AC1' #channel name
hp_freq = None #highpass cutoff frequency



#+++load data
print('select signal file: ')
root = Tk()
root.withdraw()
root.update()
filepath = filedialog.askopenfilename()
root.destroy()

wfm = f_open_tdms(filepath, channel)
time = np.arange(len(wfm))/fs


#+++data processing
if hp_freq != None:
    print('HP Filter ON')
    wfm = butter_highpass(wfm, fs, hp_freq, 3)

print('Max value is = ', amax(wfm))
print('Rms value is = ', rms(wfm))



#+++plot waveform
plt.plot(time, wfm)
plt.xlabel('Time [s]'), plt.ylabel('Amplitude [V]')
plt.show()




#+++Fourier spectrum
mag, freq, df = mag_fft(wfm, fs)
plt.plot(freq, mag, color='red')
plt.xlabel('Frequency [Hz]'), plt.ylabel('Magnitude [V]')
plt.show()