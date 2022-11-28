import torch
from random import randint
from torch.utils.data.sampler import BatchSampler
import random



def get_current_label(batch_label): 
    batch_label = batch_label.tolist()
    set_label = set(batch_label)
    assert len(set_label) == 1, "multiple entries in batch labels"
    return batch_label[0]


class CategorySampler(BatchSampler):
    def __init__(self, dataset : torch.utils.data.Dataset, batch_size = 64, shuffle = False, drop_last = False): 
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.len = len(dataset)
        self.unique_labels = torch.unique(dataset.targets).tolist()
        self.labels_list = dataset.targets
        self.indices_per_label = self.init_indices_per_label()

    def get_current_label(self): 
        return self.curr_label

    def init_indices_per_label(self): 
        indices_per_label = []
        for label in self.unique_labels:
            curr_index_list = (self.labels_list == label).nonzero().squeeze().tolist()
            indices_per_label.append(curr_index_list)

        return indices_per_label

    
    def __iter__(self): 
        for label in self.unique_labels:
            batch = [] 
            curr_label_list = self.indices_per_label[label]
            if self.shuffle:
                random.shuffle(curr_label_list) 
            for idx in curr_label_list:
                batch.append(idx)
                if len(batch) == self.batch_size: 
                    yield batch 
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch        


    def __len__(self): 
        if self.drop_last:
            return self.len // self.batch_size
        else: 
            return (self.len + self.batch_size - 1) // self.batch_size

def can_use_cupy(device): 
    first_ok = "cuda" in device 
    second_ok = False
    try:
        import cupy as _ 
        second_ok = True
    except (ModuleNotFoundError, ImportError) as e:
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


def get_toplevel_functions(network, param_class):
        dict_afs = _get_activations(network, param_class)
        functions = []
        all_keys = []
        for key in dict_afs:
            if "." not in key:
                all_keys.append(key)
        for top_key in all_keys:
            curr_obj = dict_afs[top_key]
            if type(curr_obj) is not list:
                functions.append(curr_obj)
            else:
                functions.extend(curr_obj)
        return functions

# TODO: there should be a way of isinstance(object, Container) instead,
#       but I couldn't find it.
def is_hierarchical(object):
    container_list = [torch.nn.modules.container.ModuleDict,
                    torch.nn.modules.container.ModuleList,
                    torch.nn.modules.container.Sequential,
                    torch.nn.modules.container.ParameterDict,
                    torch.nn.modules.container.ParameterList]

    for class_type in container_list:
        if isinstance(object, class_type):
            return True
    return False


def _get_activations(network, param_class):
    """
    Retrieves a dictionary of all ActivationModule AFs present in the network

    Arguments: 
        network (torch.nn.Module):
            The network from which to retrieve all the ActivationModule AFs
    Returns:
        af_dict (dictionary):
            A dictionary containing as keys the names of the layer
            and as value a list of all AFs contained in the specific layer / object.\n 
            Duplicates will be in the dictionary, as hierarchical AFs are contained 
            in both the top-level hierarchy and lower-level hierarchies
    """
    found_instances = {}
    for name, object in network.named_children():
        if isinstance(object, param_class):
            found_instances[name] = object
        elif (object):
            found_instances[name] = _process_recursive(
                found_instances, name, object, param_class)
    return found_instances


def _process_recursive(original_dict, recName, recObject, param_class):
    af_list = []
    for name, object in recObject.named_children():
        if isinstance(object, param_class):
            af_list.append(object)
        elif is_hierarchical(object):
            wholeName = recName + "." + name
            original_dict[wholeName] = _process_recursive(
                original_dict, name, object)
            af_list.extend(original_dict[wholeName])
    return af_list

def create_colors(n):
    colors = []
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors



