import numpy as np
from scipy.fft import fft, fftfreq
import os
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties.umath import sqrt, atan

def process_npy_array(nparr):
    """Return a list of arrays, where each array is the difference between the current and previous array in nparr. The first array is returned as is."""
    data = []
    xs = np.array(nparr[0][0])
    ys = np.array(nparr[0][1])
    # print(f"Dentro de treat_npy_array xs: \n {xs}")
    # print(f"Dentro de treat_npy_array ys: \n {ys}")
    # xs, ys = [], []
    for idx in range(len(nparr)):
        ys = nparr[idx][1] if idx == 0 else (nparr[idx][1]-nparr[idx-1][1])
        data.append(np.column_stack((xs,ys)))
    return data 

def extract_data_info_from_path(path: str):
  freq = int(path.split("_fg")[1].split("Hz")[0])
  amp = float(path.split("Hz_")[1].split("V_")[0])
  offset = float(path.split("V_")[1].split("offs")[0])
  data = np.loadtxt(path) if ".txt" in path else process_npy_array(np.load(path))

  return {
      "freq": freq,
      "amp": amp,
      "offset": offset,
      "data": data,
  }

def get_mean_and_variance(data):
    """Get a PROCESSED DATA array and calculate the mean and variance of the ys values for each x value. The data array is expected to be a list of arrays, where each array has two columns: the first column contains the x values and the second column contains the y values. The function returns three arrays: the x values, the mean of the y values for each x value, and the variance of the y values for each x value."""
    data = np.array(data)
    xs = data[0,:,0]
    mean = data[:,:,1].mean(axis=0)
    variance = data[:,:,1].var(axis=0)

    return (xs, mean, variance)

def fetch_data_filenames(data_path: str) -> list[str]:
    """Returns a list of all filenames in the given directory and its subdirectories."""
    filenames_all = []

    for root, dirs, files in os.walk(data_path):
        for name in files:
            full_path = os.path.join(root, name)
            filenames_all.append(full_path.replace("\\", "/"))

    return filenames_all

def get_fft(time, curve):
  sample_interval = max(time)/len(curve)/1e6  # seconds per sample

  N = len(curve)
  yf = fft(curve)
  xf = fftfreq(N, sample_interval)
  return xf, yf

def get_fund_freq_and_amp(time, curve):
    """Return the fundamental frequency and its complex amplitude (modulus and phase) for a given time-domain curve. \n
    (time, curve) -> (fund_xf, fund_yf)
    """
    xf, yf = get_fft(time, curve)
    # print(xf.shape, yf.shape)
    mask = xf > 0
    xf = xf[mask]
    yf = yf[mask]
    # print(yf)
    idx_fund_freq = np.where(np.abs(yf) == np.abs(yf).max())[0][0] if len(np.where(np.abs(yf) == np.abs(yf).max())[0]) == 1 else None
    fund_freq = xf[idx_fund_freq]
    
    if False:
        return {
            "fund_freq": fund_freq, 
            "modulus": np.abs(yf[idx_fund_freq]), 
            "phase": np.angle(yf[idx_fund_freq]),
            }
    
    return (xf[idx_fund_freq], yf[idx_fund_freq])

def plot_time_and_freq_domain(time, lum_curve, laser_curve):
    """Plot time and frequency domain of the luminescence and laser curves"""
    xf_laser, yf_laser = get_fft(time, laser_curve)
    xf_lum, yf_lum = get_fft(time, lum_curve)
    
    #Filtering only positive frequencies
    lum_mask = xf_lum > 0
    xf_lum = xf_lum[lum_mask]
    yf_lum = yf_lum[lum_mask]
    laser_mask = xf_laser > 0
    xf_laser = xf_laser[laser_mask]
    yf_laser = yf_laser[laser_mask]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    axs[0].plot(time*(1e-6), laser_curve/np.max(laser_curve), color="tab:red", label="laser")
    axs[0].plot(time*(1e-6), lum_curve/np.max(lum_curve), color="tab:green", label="lum")
    axs[0].legend()
    axs[0].set_xlabel("Time ($\mu$s)")
    axs[0].set_ylabel("Intensity (a.u.)")
    axs[0].set_title("Time Domain")

    axs[1].plot(xf_laser, np.abs(yf_laser)/np.abs(yf_laser).max(), color="tab:red", label="laser")
    axs[1].plot(xf_lum, np.abs(yf_lum)/np.abs(yf_lum).max(), color="tab:green", label="lum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].set_title("Frequency Domain")
    plt.show()

def get_modulus_angle_unc(x, dx, y, dy):
    th = atan(ufloat(y, dy)/ufloat(x, dx))
    modulus = sqrt(ufloat(x, dx)**2 + ufloat(y, dy)**2)
    return modulus, th