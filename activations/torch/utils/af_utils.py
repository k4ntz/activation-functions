
from modulefinder import Module
from venv import create
def can_use_cupy(device): 
    first_ok = "cuda" in device 
    second_ok = False
    try:
        import cupy as _ 
        second_ok = True
    except ModuleNotFoundError:
        second_ok = False
    return first_ok and second_ok

def create_histograms(inp_histograms, device, want_cupy=False):
    created_histograms = []
    can_cupy = can_use_cupy(device) and want_cupy
    msg_npy = "Loading input distributions on numpy histograms"
    #TODO: is there advantage of using numpy when device is cuda?
    if can_cupy: 
        import cupy as _ 
        from activations.torch.utils.histograms_cupy import Histogram
        for hist in inp_histograms:
                created_histograms.append(__create_single_hist(hist, Histogram))
        msg = "Loading input distributions on cupy histograms"
    else:
        if want_cupy:
            msg = """Loading input distributions on numpy histograms, since 
            cupy histograms couldn't be loaded, either due to the Module not being installed
            or cuda not being available"""
        else: 
            msg = msg_npy
        from activations.torch.utils.histograms_numpy import Histogram
        for hist in inp_histograms:
                created_histograms.append(__create_single_hist(hist, Histogram))


    return created_histograms, msg


def __create_single_hist(single_inp_histogram, histo_function):
    bin_size = single_inp_histogram.bin_size
    weights = single_inp_histogram.weights 
    bins = single_inp_histogram.bins
    created_histo = histo_function(bin_size)
    created_histo.weights = weights
    created_histo.bins = bins
    return created_histo



