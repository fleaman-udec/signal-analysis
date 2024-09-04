# #+++import modules
from nptdms import TdmsFile
from scipy.io import loadmat
import numpy as np
from scipy import signal
import pickle


def f_open_tdms(filename, channel):
    if filename == 'Input':
        filename = filedialog.askopenfilename()
    print(filename)
    file = TdmsFile.read( filename )
    all_groups = file.groups()
    group = all_groups[0]
    try:
        data_channel = group[channel]
        data = data_channel[:]
    except:
        print('***error channel, try: ')
        print(group.channels())
    return data

def mag_fft(x, fs):
	fftx = np.fft.fft(x)
	fftx = np.abs(fftx)/len(fftx)
	fftx = fftx[0:int(len(fftx)/2)]
	fftx[1:] = 2*fftx[1:]
	tr = len(x)/fs
	df = 1.0/tr
	f = np.array([i*df for i in range(len(fftx))])
	magX = fftx
	return magX, f, df

def butter_highpass(x, fs, freq, order):
	f_nyq = 0.5*fs	
	freq = freq/f_nyq
	b, a = signal.butter(order, freq, btype='highpass')
	x_filt = signal.filtfilt(b, a, x)	
	return x_filt

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def amax(x):
    return np.max(np.absolute(x))

def f_open_mat(filename, channel):
    if filename == 'Input':
        filename = filedialog.askopenfilename()
    file = loadmat(filename)
    data = file[channel]    
    return data

def save_pickle(pickle_name, pickle_data):
    pik = open(pickle_name, 'wb')
    pickle.dump(pickle_data, pik)
    pik.close()

def read_pickle(pickle_name):
    pik = open(pickle_name, 'rb')
    pickle_data = pickle.load(pik)
    return pickle_data