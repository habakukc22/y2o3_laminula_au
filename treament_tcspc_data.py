import numpy as np

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

def get_mean_and_variance(data):
    """Get a PROCESSED DATA array and calculate the mean and variance of the ys values for each x value. The data array is expected to be a list of arrays, where each array has two columns: the first column contains the x values and the second column contains the y values. The function returns three arrays: the x values, the mean of the y values for each x value, and the variance of the y values for each x value."""
    data = np.array(data)
    xs = data[0,:,0]
    mean = data[:,:,1].mean(axis=0)
    variance = data[:,:,1].var(axis=0)

    return (xs, mean, variance)