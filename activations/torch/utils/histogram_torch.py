import torch
import numpy as np
import scipy.stats as sts

class LayerHistogram: 
    def __init__(self, bin_size = "auto", device = "cpu"): 
        self.bin_size = bin_size
        self.device = device
        self._verbose = False
        if bin_size == "auto": 
            self._auto_bin_size = True
            self._bin_size = 1e-4
            self._rd = 4
        else: 
            self._auto_bin_size = False
            self._bin_size = float(bin_size)
            self._rd = int(np.log10(1./bin_size).item())
        self._fill_iplm = self._first_time_fill

    def __repr__(self):
        if self.is_empty:
            rtrn = "Empty Histogram"
        else:
            rtrn = f"Histogram on range {self.bins[0]}, {self.bins[-1]}, of " + \
                   f"bin_size {self._bin_size}, with {self.weights.sum()}" + \
                   f"elements"
        if self._verbose:
            rtrn += f" {hex(id(self))}"
        return rtrn

    @property
    def total(self):
        return self.weights.sum()

    @property
    def is_empty(self):
        if self._is_empty is True and len(self.bins) > 0:
            self._is_empty = False
        return self._is_empty

    @property
    def range(self):
        x_min = float(self.bins[0])
        x_max = float(self.bins[-1])
        return np.arange(x_min, x_max, self._bin_size/100)

    @property
    def bin_size(self):
        return self._bin_size

    @bin_size.setter
    def bin_size(self, value):
        self._bin_size = value


    def _first_time_fill(self, new_input):
        if not isinstance(new_input, torch.Tensor): 
            new_input = torch.tensor(new_input, device=self.device, dtype = torch.float32)

        range_ext = round_n(torch.min(new_input) - (self._bin_size / 2), self._rd), \
                    round_n(torch.max(new_input) + (self._bin_size / 2), self._rd)
        
        bins_array = torch.arange(range_ext[0], range_ext[1] + self._bin_size, self.bin_size)
        weights, bins = torch.histogram(new_input, bins_array)
        if self._auto_bin_size:
            self._rd = int(np.log10(1./(range_ext[1] - range_ext[0])).item()) + 2
            self._bin_size = 1./(10**self._rd)
            range_ext = round_n(torch.min(new_input) - (self._bin_size / 2), self._rd), \
                    round_n(torch.max(new_input) + (self._bin_size / 2), self._rd)
            bins_array = torch.arange(range_ext[0], range_ext[1] + self._bin_size, self.bin_size)
            weights, bins = torch.histogram(new_input, bins_array)

        self.weights, self.bins = weights, bins[:-1]
        self._is_empty = False
        self._fill_iplm = self._update_hist


    def fill_n(self, input): 
        self._fill_iplm(input.detach().numpy())

    def _update_hist(self, new_input):
        if not isinstance(new_input, torch.Tensor): 
            new_input = torch.tensor(new_input, device=self.device, dtype = torch.float32)
    

        range_ext = round_n(torch.min(new_input) - (self._bin_size / 2), self._rd), \
                    round_n(torch.max(new_input) + (self._bin_size / 2), self._rd)
        bins_array = torch.arange(range_ext[0], range_ext[1] + self._bin_size, self.bin_size)
        weights, bins = torch.histogram(new_input, bins_array)
        self.weights, self.bins = self.concat_hists(self.weights, self.bins, 
                                                    weights, bins[:-1], 
                                                    self._bin_size, self._rd)
        
    def concat_hists(self, weights1, bins1, weights2, bins2, bin_size, rd):
        min1, max1 = round_n(bins1[0], rd), round_n(bins1[-1], rd)
        min2, max2 = round_n(bins2[0], rd), round_n(bins2[-1], rd)
        
        real_min, real_max = min(min1, min2), max(max1, max2)
        new_bins = torch.arange(real_min, real_max + bin_size * 0.9, bin_size)
        if min1 - real_min != 0 and real_max - max1 != 0:
            ext1 = np.pad(weights1, (np.int(np.around((min1 - real_min) / bin_size)),
                                 np.int(np.around((real_max - max1) / bin_size))),
                      'constant', constant_values=0)
        elif min1 - real_min != 0:
            ext1 = np.pad(weights1, (np.int(np.around((min1 - real_min) / bin_size)),
                                    0), 'constant', constant_values=0)
        elif real_max - max1 != 0:
            ext1 = np.pad(weights1, (0,
                                    np.int(np.around((real_max - max1) / bin_size))),
                        'constant', constant_values=0)
        else:
            ext1 = weights1
        if min2 - real_min != 0 and real_max - max2 != 0:
            ext2 = np.pad(weights2, (np.int(np.around((min2 - real_min) / bin_size)),
                                    np.int(np.around((real_max - max2) / bin_size))),
                        'constant', constant_values=0)
        elif min2 - real_min != 0:
            ext2 = np.pad(weights2, (np.int(np.around((min2 - real_min) / bin_size)),
                                    0), 'constant', constant_values=0)
        elif real_max - max2 != 0:
            ext2 = np.pad(weights2, (0,
                                    np.int(np.around((real_max - max2) / bin_size))),
                        'constant', constant_values=0)
        else:
            ext2 = weights2
        new_ext = ext1 + ext2
        return torch.tensor(new_ext), torch.tensor(new_bins)

    def kde(self):
        kde = sts.gaussian_kde(self.bins, bw_method=0.13797296614612148,
                               weights=self.weights)
        return kde.pdf

def round_n(input, n): 
    return torch.round(input  * 10**n) / (10**n)