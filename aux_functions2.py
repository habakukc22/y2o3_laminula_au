import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit

def load_and_format(file_path: str):
    """Load the data from the given file path and format it into two lists: xs and ys."""
    data = np.loadtxt(file_path)
    xs = []
    ys = []
    for idx, item in enumerate(data):
        xs.append(item[0])
        ys.append(item[1])
        # print(idx, item)

    return np.array(xs), np.array(ys)

def fit_curve(curve, fitting_func, p0=None):
    """Ajusta `fitting_func` aos dados de `curve` (array Nx2)."""
    xs, ys = curve
    popt, _ = curve_fit(fitting_func, xs, ys, p0=p0)
    return xs, popt

def get_area(xs, ys):
    """Calculate the area under the curve defined by the data."""
    area = integrate.simpson(y=ys, x=xs)
    return area

def remove_bg(data: tuple, points_off: int = 30):
    """Remove the background from the given data taking into account the specified number of points."""
    x_data, y_data = data
    
    points_to_remove = list(y_data)[:points_off] + list(y_data)[-points_off:]
    x_bg = list(x_data)[:points_off] + list(x_data)[-points_off:]
    
    bg_func = lambda x, a, b: a * x + b
    
    _ , popt = fit_curve((x_bg, points_to_remove), bg_func)
    
    y_bg = bg_func(x_data, *popt)

    y_data_bg_removed = y_data - y_bg
    return x_data, y_data_bg_removed

def shorten_points(data, num_points_off=10):
    """Remove os primeiros e últimos `num_points_off` pontos. Aceita shape (2, N)."""
    xs, ys = data[0], data[1]
    return xs[num_points_off:-num_points_off], ys[num_points_off:-num_points_off]

def make_fitting_func(omega):
    """Retorna a função seno parametrizada para a frequência angular `omega`."""
    def fitting_func(t, A, A0, phi):
        return A0 + np.abs(A) * np.sin(omega * t + phi)
    return fitting_func

def adj_omega(x):
    """Converte omega (rad/µs) para frequência em Hz."""
    return x * 1e6 / (2 * np.pi)

def adj_ph(x, correct=False):
    """Ajusta a fase: se `correct=True`, mapeia valores de fase para o intervalo [-180, 180]."""
    # if correct:
        # return np.where(x > 180, x - 360, -x)
    return (360 - x)

def adj_data_shape(data):
    """Garante que os dados tenham shape (2, N) para compatibilidade com shorten_points."""
    # Se os dados tiverem shape (N, 2), transponha para (2, N)
    if data.shape[1] == 2:
        return data.T
    return data