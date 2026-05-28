import numpy as np
from scipy.fft import fft, fftfreq
import os
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties.umath import sqrt, atan

def many_consecutive_zeros(data, consecutive_zeros = 10):
    """Find consecutive zeros in an array of values."""
    zeros = 0
    for val in data:
        if val == 0:
            zeros += 1
            if zeros == 10:
                return True # Encontrou a sequência!
        else:
            zeros = 0 # Quebrou a sequência, reseta o contador
            
    return False

# def process_npy_array(nparr):
#     """Return a list of arrays, where each array is the difference between the current and previous array in nparr. The first array is returned as is."""
#     data = []
#     non_zero_idx = 0

#     for idx in range(len(nparr)):
#       if len(nparr[idx][0])>0:
#         non_zero_idx = idx
#         xs = np.array(nparr[idx][0])
#         break

#     #Find minimum length
#     min_len = len(xs)
#     for idx in range(len(nparr)):
#       if idx >= non_zero_idx:
#         min_len = min(min_len, len(nparr[idx][1]))
    
#     xs = xs[:min_len]

#     # ys = np.array(nparr[0][1])
    
#     for idx in range(len(nparr)):
#         if idx>=non_zero_idx:
#           if idx == non_zero_idx:
#             ys = np.array(nparr[idx][1])[:min_len]
#           else:
#             ys = np.array(nparr[idx][1])[:min_len]-np.array(nparr[idx-1][1])[:min_len]
#           data.append(np.column_stack((xs,ys)))
    
#     #Filter array to exclude points with 0 counts
#     filtered_data =[
#       array for array in data 
#       if not many_consecutive_zeros(array[:200, 1])
#       # if np.any(array[:100, 1] != 0)
#       ]

#     #Shorten the list of points to 200 points
#     if len(filtered_data[0])>200:
#       short_arr = [arr[:200,:] for arr in filtered_data]
#       return short_arr
    
#     return filtered_data

# def extract_data_info_from_path(path: str, raw_data = False):
#   freq = int(path.split("_fg")[1].split("Hz")[0])
#   amp = float(path.split("Hz_")[1].split("V_")[0])
#   offset = float(path.split("V_")[1].split("offs")[0])
#   if raw_data:
#     data = np.loadtxt(path) if ".txt" in path else np.load(path, allow_pickle=True)
#   else:
#     data = np.loadtxt(path) if ".txt" in path else process_npy_array(np.load(path, allow_pickle=True))

#   return {
#       "freq": freq,
#       "amp": amp,
#       "offset": offset,
#       "data": data,
#   }

def process_npy_array(nparr):
    """Return a list of arrays, where each array is the difference between the current and previous array in nparr. The first array is returned as is."""
    data = []
    num_points = 200
    
    # print(f"nrep: {len(nparr)}")

    if len(nparr[0][0])>0:
      xs = np.array(nparr[0][0])[:num_points]
    else:
      print("A primeira das medidas possui array de tempo vazio")
      return None

    for idx in range(len(nparr)):
      # if idx > 199: break
      ys = np.array(nparr[idx][1])[:num_points]
      # print(f"(xs, ys) = ({len(xs)},{len(ys)})")
      if len(xs)==len(ys):
        data.append(np.column_stack((xs,ys)))
      else:
        # plt.plot(xs, ys)
        print("O tamanho de xs não é o mesmo de ys")
        return None
        
    return data

def extract_data_info_from_path(path: str, raw_data = False):
  freq = int(path.split("_fg")[1].split("Hz")[0])
  amp = float(path.split("Hz_")[1].split("V_")[0])
  offset = float(path.split("V_")[1].split("offs")[0])
  try:
    if raw_data:
      data = np.loadtxt(path) if ".txt" in path else np.load(path, allow_pickle=True)
    else:
      data = np.loadtxt(path) if ".txt" in path else process_npy_array(np.load(path, allow_pickle=True))
    
    return {
      "freq": freq,
      "amp": amp,
      "offset": offset,
      "data": data,
    }
  except:
    print(f"Error to load freq {freq}Hz file.\nThe file path is {path}")
    # return None


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

def get_xf_yf_fund(time, curve):
  """Return the frequency and complex amplitude for the point of the FFT with largest modulus. \n
  (time, curve) -> (xf_fund, yf_fund)"""
  xf, yf = get_fft(time, curve)
  mask = xf > 0
  xf = xf[mask]
  yf = yf[mask]

  abs_yf_list = np.abs(yf)
  largest_amp_list = np.where(abs_yf_list == abs_yf_list.max())[0]
  idx_freq = largest_amp_list[0] if len(largest_amp_list) == 1 else None

  xf_fund, yf_fund = xf[idx_freq], yf[idx_freq]

  return xf_fund, yf_fund

def get_mean_amp_and_phase(phasors_list: list):
    """Return mean modulus, mean phase and their errors for a list of complex numbers. \n
    phasor_list -> (mean_amp, amp_error, mean_phase, phase_error)
    """
    phasors = np.array(phasors_list)
    mean_phasor = phasors.mean()
    mean_amp = np.abs(mean_phasor)
    amp_error = ((np.abs(phasors)).std())/np.sqrt(len(phasors))
    mean_phasor_phase = np.angle(mean_phasor)
    phase_list = np.angle(phasors)
    phase_deviations = phase_list - mean_phasor_phase
    wrapped_deviations = (phase_deviations + np.pi) % (2 * np.pi) - np.pi
    
    return mean_amp, amp_error, mean_phasor_phase, (wrapped_deviations.std())/np.sqrt(len(wrapped_deviations))

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
