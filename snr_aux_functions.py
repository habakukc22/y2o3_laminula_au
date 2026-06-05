import numpy as np
from scipy.fft import fft, fftfreq
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def process_npy_array(nparr):
  """Return a list of arrays, where each array is the difference between the current and previous array in nparr. The first array is returned as is."""
  data = []
  num_points = 200
  points_off = 2
  
  for idx in range(len(nparr)):
    xs = np.array(nparr[idx][0])[:num_points]
    ys = np.array(nparr[idx][1])[:num_points]
    
    if len(xs) != len(ys):
      raise Exception(f"\nO tamanho de xs não é o mesmo de ys: (len(xs), len(ys)) = {len(xs), len(ys)}")
    
    if len(ys) < int(0.95*num_points) :
      # Remove measures with less than 95% of num_point 
      continue
    
    data.append(np.column_stack((
      xs[points_off:-points_off], ys[points_off:-points_off]
      )))

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
  except Exception as e:
    print(f"Error to load freq {freq}Hz file.\nThe file path is {path}\nException detail: {e}")

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

def get_snr_db(time, curve, freq=1):
    """
    Calculate the Signal-Noise Relation (SNR) in dB.
    
    Parâmeters:
      freq = 0: The 'signal' is the DC component (zero frequency).
      freq = 1: The 'signal' is the fundamentabl frequency (largest AC spike).
      freq = 1: The 'signal' is 2nd harmonic (2*fundamental_frequency).

    The remaining power of the spectrum is considered noise.
    """
    # 1. Obtém a FFT usando a sua função existente
    xf, yf = get_fft(time, curve)
    
    # 2. Mantém o termo DC e as frequências positivas
    mask_positive = xf >= 0
    xf = xf[mask_positive]
    yf = yf[mask_positive]
    
    # 3. Calcula o espectro de potência (módulo ao quadrado)
    power = np.abs(yf)**2

    # 4. Encontra a frequência fundamental (omega)
    # Ignoramos temporariamente o termo DC para garantir que acharemos o maior pico AC
    power_ac = np.copy(power)
    power_ac[xf == 0] = 0 

    idx_fund = np.argmax(power_ac)
    
    freq_fund = xf[idx_fund]

    idx_dc = np.argmin(np.abs(xf))
    idx_2omega = np.argmin(np.abs(xf - 2*freq_fund))

    
    # 5. Seleciona o índice do sinal de interesse com base no parâmetro 'freq'
    if freq == 0:
        # Sinal é o termo DC (frequência mais próxima de 0)
        idx_signal = idx_dc
        
    elif freq == 1:
        # Sinal é a frequência fundamental (omega)
        idx_signal = idx_fund
        
    elif freq == 2:
        # Sinal é o segundo harmônico (2 * omega)
        idx_signal = idx_2omega
        
    else:
        raise ValueError("O parâmetro 'freq' deve ser 0, 1 ou 2.")

    # Extrai a potência do sinal selecionado
    p_signal = power[idx_signal]
    
    # Opcional: print para debug/verificação
    # print(f"Modo: freq={freq} | Freq Sinal: {xf[idx_signal]:.4f} Hz | Idx: {idx_signal}")

    # Remove a contribuição do DC para o cálculo do ruído (se o sinal não for o DC)
    if freq != 0:
        # Encontra o índice da frequência 0 (geralmente é o índice 0)
        power[idx_dc] = 0.0
    
    # 6. Identifica a potência do Ruído (Potência total - Potência do sinal)
    p_noise = (np.sum(power) - power[idx_fund] - power[idx_dc] - power[idx_2omega])/(len(power)-3)
    
    # Tratamento de segurança: se o sinal for uma senoide perfeita simulada
    if p_noise <= 1e-15:
        return float('inf') 
        
    # 7. Calcula a razão em dB
    snr_db = 10 * np.log10(p_signal / p_noise)
    
    return snr_db

def plot_time_and_freq_domain(time, lum_curve, laser_curve):
    """Plot time and frequency domain of the luminescence and laser curves"""
    # Suponho que a função get_fft já esteja definida no seu código
    xf_laser, yf_laser = get_fft(time, laser_curve)
    xf_lum, yf_lum = get_fft(time, lum_curve)
    
    # Filtering only positive frequencies
    lum_mask = xf_lum > 0
    xf_lum = xf_lum[lum_mask]
    yf_lum = yf_lum[lum_mask]
    laser_mask = xf_laser > 0
    xf_laser = xf_laser[laser_mask]
    yf_laser = yf_laser[laser_mask]

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # --- DOMÍNIO DO TEMPO ---
    # Eixo principal (Esquerdo - Verde/Lum)
    axs[0].set_xlabel("Time ($\mu$s)")
    axs[0].set_ylabel("Luminescence (a.u.)", color="tab:green")
    line1 = axs[0].plot(time, lum_curve, color="tab:green", label="lum")
    axs[0].tick_params(axis='y', labelcolor="tab:green")
    axs[0].set_title("Time Domain")

    # Eixo secundário (Direito - Vermelho/Laser)
    ax0_twin = axs[0].twinx()
    ax0_twin.set_ylabel("Laser (a.u.)", color="tab:red")
    line2 = ax0_twin.plot(time, laser_curve, color="tab:red", label="laser")
    ax0_twin.tick_params(axis='y', labelcolor="tab:red")

    # Juntando as legendas dos dois eixos
    lines_time = line1 + line2
    labels_time = [l.get_label() for l in lines_time]
    axs[0].legend(lines_time, labels_time, loc="best")


    # --- DOMÍNIO DA FREQUÊNCIA ---
    # Eixo principal (Esquerdo - Verde/Lum)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Lum Power", color="tab:green")
    line3 = axs[1].plot(xf_lum, np.abs(yf_lum)**2, color="tab:green", label="lum")
    axs[1].tick_params(axis='y', labelcolor="tab:green")
    axs[1].set_title("Frequency Domain")
    axs[1].set_yscale("log")

    # Eixo secundário (Direito - Vermelho/Laser)
    ax1_twin = axs[1].twinx()
    ax1_twin.set_ylabel("Laser Power", color="tab:red")
    ax1_twin.set_yscale("log")
    line4 = ax1_twin.plot(xf_laser, np.abs(yf_laser)**2, color="tab:red", label="laser")
    ax1_twin.tick_params(axis='y', labelcolor="tab:red")

    # Juntando as legendas dos dois eixos
    lines_freq = line3 + line4
    labels_freq = [l.get_label() for l in lines_freq]
    axs[1].legend(lines_freq, labels_freq, loc="best")

    plt.show()

def calculate_harmonic_powers(time, curve, fundamental_freq, num_harmonics=5):
    """
    Calculates Total Harmonic Distortion (THD) for a given signal.
    Time has to be in us.
    """
    # 1. Obtém a FFT usando a sua função existente
    xf, yf = get_fft(time, curve)
    
    # 2. Mantém o termo DC e as frequências positivas
    mask_positive = xf > 0
    xf = xf[mask_positive]
    yf = yf[mask_positive]
        
    # Calculate magnitudes
    magnitudes = np.abs(yf)
    
    # Helper to find the magnitude peak closest to a target frequency
    def get_peak_magnitude(target_f):
        idx = np.argmin(np.abs(xf - target_f))
        return magnitudes[idx]

    # Get fundamental magnitude (V1)
    v1 = get_peak_magnitude(fundamental_freq)
    # plt.scatter(fundamental_freq, v1**2, color="b")
    
    
    p_spectrum = [[fundamental_freq, v1**2]]

    # Collect harmonic magnitudes (V2, V3, ..., Vn)
    
    for h in range(2, num_harmonics + 1):
        v_h = get_peak_magnitude(fundamental_freq * h)
        p_spectrum.append([fundamental_freq*h, v_h ** 2])
    
    return np.array(p_spectrum)

def calculate_noise_powers(time, curve, fundamental_freq, first_noisy_harmonic = 20):
    """
    Calculates Total Harmonic Distortion (THD) for a given signal.
    Time has to be in us.
    """
    # 1. Obtém a FFT usando a sua função existente
    xf, yf = get_fft(time, curve)
    
    # 2. Mantém o termo DC e as frequências positivas
    mask_positive = xf > 0
    xf = xf[mask_positive]
    yf = yf[mask_positive]
        
    # Calculate magnitudes
    powers = np.abs(yf)**2
    
    # Helper to find the magnitude peak closest to a target frequency
    def get_peak_power_and_idx(target_f):
        idx = np.argmin(np.abs(xf - target_f))
        return idx, powers[idx]

    # Get fundamental magnitude (V1)
    fund_idx , _ = get_peak_power_and_idx(fundamental_freq)

    greatest_harmonic = int(np.floor(xf[-1]/xf[fund_idx]))
    harmonic_idx_list = []
    for n in range(1, greatest_harmonic + 1):
        idx_n, _ = get_peak_power_and_idx(n*fundamental_freq)
        harmonic_idx_list.append(idx_n)

    harmonic_power_list = powers[harmonic_idx_list]
    
    mean_noise = harmonic_power_list[first_noisy_harmonic:].mean()
    std_noise = harmonic_power_list[first_noisy_harmonic:].std()
              
    noise_threshold = mean_noise + 3*std_noise
    return noise_threshold 

def plot_harmonic_power(particles, particles_off=[], freqs_off=[], laser_exc=[], power_label=[]):
    
    for p_idx, p in enumerate(particles):
        if p_idx in particles_off: continue

        # 1. Cria a figura e a grade de subplots (6 linhas, 4 colunas)
        # O figsize é ajustado para que os gráficos não fiquem espremidos
        fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(16, 20))
        
        # Define o título principal (suptitle) para toda a figura
        if bool(power_label):
            fig.suptitle(f"Harmonic powers - Luminescence - Laser power {power_label[p_idx]}mA", fontsize=18, fontweight='bold')
            # fig.suptitle(f"Laser ??", fontsize=18, fontweight='bold')
        else:
            fig.suptitle(f"Harmonic powers - Luminescence", fontsize=18, fontweight='bold')
            # fig.suptitle(f"Laser", fontsize=18, fontweight='bold')

        # "Achata" a matriz 6x4 de eixos em uma lista 1D de 24 posições para facilitar o acesso
        axes_flat = axes.flatten()

        cmap = plt.get_cmap("coolwarm")
        # Pequena correção no vmax para evitar divisão por zero se houver apenas 1 partícula
        norm = mcolors.Normalize(vmin=0, vmax=max(1, len(particles)-1))
        c_grad = cmap(norm(len(particles) - p_idx))

        plot_idx = 0 # Contador para sabermos em qual subplot estamos

        for step_idx, step in enumerate(p["p_data"]):
            freq = step["freq"]
            if freq in freqs_off: continue

            # Trava de segurança: se houver mais de 24 passos, evita erro de índice
            # if plot_idx >= 24:
            #     print(f"Aviso: Mais de 24 frequências para a partícula {p_idx}. As extras não serão plotadas.")
            #     break

            # 2. Seleciona o subplot atual
            ax = axes_flat[plot_idx]
            
            # Get the luminescence points and noise 
            time = np.array(step["data"])[0,:,0]
            signal = np.array(step["data"])[:,:,1].mean(axis=0)

            p_spec_harm = calculate_harmonic_powers(
                time, signal, 
                fundamental_freq=freq, 
                num_harmonics=20
            )
            threshold_noise = calculate_noise_powers(
                time, signal,
                fundamental_freq=freq
            )
            
            # Get the laser points and noise
            is_laser_present = len(laser_exc) != 0
            does_laser_freq_match = False if (not is_laser_present) else laser_exc[step_idx]["freq"] == freq

            if is_laser_present and does_laser_freq_match:
                time = np.array(laser_exc[step_idx]["data"])[0,:,0]
                signal = np.array(laser_exc[step_idx]["data"])[:,:,1].mean(axis=0)

                p_spec_harm_exc = calculate_harmonic_powers(
                    time, signal, 
                    fundamental_freq=freq, 
                    num_harmonics=20
                )
                threshold_noise_exc = calculate_noise_powers(
                    time, signal,
                    fundamental_freq=freq
                ) 

            # 3. Usa 'ax.plot' e 'ax.scatter' em vez de 'plt.*'
            
            #Plot luminescence
            ax.plot(p_spec_harm[:,0], (threshold_noise)*np.ones(len(p_spec_harm[:,0])),
                    color="#008D07", linestyle="--")
            ax.scatter(p_spec_harm[:,0], p_spec_harm[:,1],
                       label="lum", color=c_grad)
            
            # Plot laser
            if is_laser_present and does_laser_freq_match:
                ax.plot(p_spec_harm_exc[:,0], (threshold_noise_exc)*np.ones(len(p_spec_harm_exc[:,0])),
                        color="#C5C5C5", linestyle="--")
                ax.scatter(p_spec_harm_exc[:,0], p_spec_harm_exc[:,1],
                           label="exc", color="gray")
                # ax.plot(p_spec_harm[:,0], (p_noise - 3*p_std)*np.ones(len(p_spec_harm[:,0])), "b--")
            
            # 4. Configura títulos e eixos (no formato ax.set_*)
            ax.set_title(f"{freq} Hz", fontsize=12)
            ax.legend()
            ax.set_ylabel("$|A(\omega)|^2$")
            ax.set_xlabel("f (Hz)")
            ax.set_yscale("log")
            ax.set_xscale("log")

            plot_idx += 1

        # 5. Limpeza: Oculta subplots vazios caso a partícula tenha menos de 24 frequências válidas
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

        # 6. Ajusta o layout para que os títulos não se sobreponham aos gráficos de cima
        plt.tight_layout()
        fig.subplots_adjust(top=0.93) # Dá um respiro pro suptitle não esmagar a primeira linha
        
        # Mostra o painel completo da partícula atual antes de ir para a próxima
        plt.show()

def calculate_thd(time, curve, fundamental_freq, num_harmonics):
    """
    Calculates Total Harmonic Distortion (THD) for a given signal.
    Time has to be in us.
    """
    # 1. Obtém a FFT usando a sua função existente
    xf, yf = get_fft(time, curve)
    
    # 2. Mantém o termo DC e as frequências positivas
    mask_positive = xf > 0
    xf = xf[mask_positive]
    yf = yf[mask_positive]
        
    # Calculate magnitudes
    magnitudes = np.abs(yf)
    
    # Helper to find the magnitude peak closest to a target frequency
    def get_peak_magnitude(target_f):
        idx = np.argmin(np.abs(xf - target_f))
        return magnitudes[idx]

    # Get fundamental magnitude (V1)
    v1 = get_peak_magnitude(fundamental_freq)
    
    if v1 == 0:
        return 0.0

    # Collect harmonic magnitudes (V2, V3, ..., Vn)
    harmonic_sq_sum = 0.0
    for h in range(2, num_harmonics + 1):
        v_h = get_peak_magnitude(fundamental_freq * h)
        harmonic_sq_sum += v_h ** 2

    # Compute THD ratio and percentage
    thd_ratio = np.sqrt(harmonic_sq_sum) / v1
    return thd_ratio * 100

def plot_thd(particles, num_harmonics, particles_off = [], freqs_off = [], laser_data = []):

    for p_idx, p in enumerate(particles):
        if p_idx in particles_off: continue

        cmap = plt.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=0, vmax = (len(particles)-1) )

        c_grad = cmap(norm(len(particles) - p_idx))
        put_label = True

        for step_idx, step in enumerate(p["p_data"]):
            
            if step_idx != 0: 
                put_label = False

            if step["freq"] in freqs_off: continue
            # if step["freq"]

            time = np.array(step["data"])[0,:,0]
            signal = np.array(step["data"])[:,:,1].mean(axis=0)
            plt.scatter(
                step["freq"], 
                calculate_thd(time, signal, fundamental_freq=step["freq"], num_harmonics=num_harmonics), 
                color=c_grad,
                label= "lum" if put_label else None 
                )
            
    put_label = True

    if bool(laser_data):
        for step_idx, step in enumerate(laser_data):
            if step["freq"] in freqs_off: continue

            if step_idx != 0: 
                put_label = False

            time = np.array(step["data"])[0,:,0]
            signal = np.array(step["data"])[:,:,1].mean(axis=0)

            plt.scatter(step["freq"], 
                        calculate_thd(time, signal, 
                                      fundamental_freq=step["freq"], 
                                      num_harmonics=num_harmonics),
                        color="purple",
                        label= "exc" if put_label else None
            )

    plt.xscale("log")
    plt.ylabel("THD (%)")
    plt.xlabel("Frequency (Hz)")
    plt.title(f"THD from 2 to {num_harmonics} harmonics")
    plt.legend()
    plt.show()

#===========================================================================================

# def plot_time_and_freq_domain(time, lum_curve, laser_curve):
#     """Plot time and frequency domain of the luminescence and laser curves"""
#     xf_laser, yf_laser = get_fft(time, laser_curve)
#     xf_lum, yf_lum = get_fft(time, lum_curve)
    
#     #Filtering only positive frequencies
#     lum_mask = xf_lum > 0
#     xf_lum = xf_lum[lum_mask]
#     yf_lum = yf_lum[lum_mask]
#     laser_mask = xf_laser > 0
#     xf_laser = xf_laser[laser_mask]
#     yf_laser = yf_laser[laser_mask]

#     fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

#     axs[0].plot(time*(1e-6), laser_curve/np.max(laser_curve), color="tab:red", label="laser")
#     axs[0].plot(time*(1e-6), lum_curve/np.max(lum_curve), color="tab:green", label="lum")
#     axs[0].legend()
#     axs[0].set_xlabel("Time ($\mu$s)")
#     axs[0].set_ylabel("Intensity (a.u.)")
#     axs[0].set_title("Time Domain")

#     axs[1].plot(xf_laser, np.abs(yf_laser)/np.abs(yf_laser).max(), color="tab:red", label="laser")
#     axs[1].plot(xf_lum, np.abs(yf_lum)/np.abs(yf_lum).max(), color="tab:green", label="lum")
#     axs[1].set_xlabel("Frequency (Hz)")
#     axs[1].set_ylabel("Amplitude")
#     axs[1].legend()
#     axs[1].set_title("Frequency Domain")
#     plt.show()