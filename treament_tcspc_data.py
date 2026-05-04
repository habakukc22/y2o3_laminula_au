import numpy as np

def treat_npy_array(nparr):
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