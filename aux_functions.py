import numpy as np
from scipy.optimize import curve_fit

def shorten_points(data, num_points_off=10):
    """Remove os primeiros e últimos `num_points_off` pontos. Aceita shape (2, N)."""
    if len(data) > 200:
        data = data[:200]

    xs, ys = data[:,0], data[:,1]
    return xs[num_points_off:-num_points_off], ys[num_points_off:-num_points_off]

def make_fitting_func(omega):
    """ Return a sinusoidal function parameterized by the angular frequency `omega`. """
    def fitting_func(t, A, A0, phi):
        return A0 + A * np.sin(omega * t - phi)
    return fitting_func

def fit_curve(curve, fitting_func, p0=None, bounds=None):
    """Fit `fitting_func` to the data in `curve` (array Nx2)."""    
    if len(curve) > 200:
        curve = curve[:200]
    
    xs, ys = shorten_points(curve)
    popt, _ = curve_fit(fitting_func, xs, ys, p0=p0, bounds=bounds)
    return xs, popt

def adj_omega(x):
    """ Convert omega (rad/µs) to frequency in Hz. """
    return x * 1e6 / (2 * np.pi)

def adj_ph(x, correct=False):
    """Correct the phase: if `correct=True`, map phase values to the range [-180, 180]."""
    if correct:
        return np.where(x > 180, x - 360, -x)
    return (360 - x)

def inverse_variance(variance_list: list):
    """Calculate the combined variance from a list of variances using the formula for inverse variance weighting."""
    return 1/sum([1/variance for variance in variance_list])

def sort_experiments_by_freq(exp_list: list):
    """Return a new list of experiments sorted by their 'freq' key."""
    return sorted(exp_list, key=lambda exp: exp['freq'])   