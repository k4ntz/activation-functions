import torch
from activations.utils.utils import _get_auto_axis_layout, _cleared_arrays
from activations.utils.warnings import RationalImportScipyWarning
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import activations.torch.utils.af_utils as af_utils
from activations.torch.utils.af_utils import create_colors, get_toplevel_functions
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch.utils.data.sampler import Sampler
import random

_LINED = dict()


class CategorySampler(Sampler):
    def __init__(self, dataset): 
        self.len = len(dataset)
        self.dataset_labels = torch.unique(dataset.targets)
        self.label_per_data = dataset.targets
        self.len_data_per_label = self.__init_len()
        self.curr_num = 0

    def __init_len(self): 
        pass
    
    def __iter__(self): 
        pass
        


    def __len__(self): 
        return self.len




def _save_inputs(self, input, output):
    curr_dist = self.get_current_distribution()
    if curr_dist is None:
        raise ValueError("Selected distribution is none")
    self.inputs_saved += 1
    curr_dist(input[0])
    #TODO: dunno why I used this here.
    """ if self.inputs_saved > 0 and self.inputs_saved % self.update_interval_dist == 0:
        self.save_histo() """


def _save_gradients(self, in_grad, out_grad):
    self._in_grad_dist.fill_n(in_grad[0])
    self._out_grad_dist.fill_n(out_grad[0])


def _save_inputs_auto_stop(self, input, output):
    if self.current_inp_distribution is None:
        raise ValueError("Selected distribution is none")

    self.inputs_saved += 1
    self.current_inp_distribution.fill_n(input[0])
    if self.inputs_saved > self._max_saves:
        self.training_mode()



class ActivationModule(torch.nn.Module):
    # histograms_colors = plt.get_cmap('Pastel1').colors
    instances = {}
    histograms_colors = ["red", "green", "black"]
    distribution_display_mode = "kde"
    logger = logging.getLogger("ActivationModule")
    

    def __init__(self, function, device=None):
        if isinstance(function, str):
            self.type = function
            function = None
        super().__init__()
        if self.classname not in self.instances:
            self.instances[self.classname] = []
        self.instances[self.classname].append(self)
        if function is not None:
            self.activation_function = function
            if "__forward__" in dir(function):
                self.forward = self.activation_function.__forward__
            else:
                self.forward = self.activation_function


        
        self._init_inp_distributions()
        self._init_grad_distributions()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _init_grad_distributions(self):
        self._handle_grads = None
        self._in_grad_dist = None
        self._out_grad_dist = None
        self._grad_bin_width = "auto"
        self._can_show_grad = False
        self.grad_mode = "layer"
        self.update_interval_grad = 0

    def _init_inp_distributions(self):
        self.curr_cat_name = None
        self.distributions = dict() 


        self._saving_input = False
        self.dist_mode = "layer"
        self._handle_inputs = None
        self.inp_bin_width = "auto"
        self.can_show_inp = False
        self.update_interval_dist = 0
        if self.update_interval_dist > 0: 
            self.saved_histos = []
            

    def __add_category(self, name, histo):
        """
        Adds a distribution histo with a name to the input distributions if the category name is not already 
        present in the distributions
        """
        for cat in self.distributions.keys():
            if name == cat:
                self.logger.warn(f"Will not add a category for {name}, since it is already present in the distributions")
                return

        self.distributions[name] = histo


    @classmethod 
    def print_categories(cls, input_fcts = None): 
        instances = cls.get_instance_list(input_fcts)
        for inst in instances:
            inst.print_categories_single()

    def print_categories_single(self): 
        msg = f"Activation {self} has distributions: {self.distributions}"
        print(msg)

    @classmethod
    def change_categories(cls, value, input_fcts = None):
        value = str(value)
        instances = cls.get_instance_list(input_fcts)
        for inst in instances:
            inst.change_category(value)

    def change_category(self, value):
        if not value in self.distributions.keys():
            raise ValueError(f"No distribution under the name {value} exists, only {self.distributions.keys()} exist")
        else:
            self.curr_cat_name = value

    def get_current_distribution(self):
        if self.curr_cat_name is None: 
            return None
        else:            
            return self.distributions[self.curr_cat_name]


    @classmethod 
    def get_current_dist_cat(cls, input_fcts = None): 
        instances = cls.get_instance_list(input_fcts)

        curr_dists = []
        for inst in instances:
            cat_dist_pair = inst.get_current_dist_cat()
            curr_dists.append(cat_dist_pair)
        
        return curr_dists



    def get_current_dist_cat(self):
        return (self.curr_cat_name, self.get_current_distribution())




    #TODO Category stuff: 
    #1. Want to register categories -> there is n distributions {"name": distribution}
    #2. Want to have a current category -> need a value "self_curr_category"
    #3. Want to set current_category -> just set the value of self_curr_category 




    def get_mode_func(self, value, device, for_dists):
        value = str(value)
        #allowed_modes = ["categories", "neurons", "neurons_categories"]
        allowed_modes = ["layer", "neurons"]
        if value.lower() in allowed_modes: 
            if for_dists: 
                self.dist_mode = value
            else: 
                self.grad_mode = value
            can_cupy = af_utils.can_use_cupy(device)
            if can_cupy:
                if "neurons" in value:
                    from activations.torch.utils.histograms_cupy import NeuronsHistogram as hist   
                else: 
                    from activations.torch.utils.histograms_cupy import Histogram as hist
                histo_func = hist
            else: 
                if "neurons" in value: 
                    from activations.torch.utils.histograms_numpy import NeuronsHistogram as hist
                else:
                    from activations.torch.utils.histograms_numpy import Histogram as hist
                histo_func = hist
            return histo_func
        else: 
            self.logger.critical(f"Mode {value} is currently not supported")
            raise ValueError()


    @property
    def classname(self):
        clsn = str(self.__class__)
        if "activations.torch" in clsn:
            return clsn.split("'")[1].split(".")[-1]
        else:
            return "Unknown"  # TODO, implement

    def save_inputs(self, saving=True, auto_stop=False, max_saves=1000,
                    bin_width=0.1, mode="layer", category_name=None, save_interval = 0):
        """
        Will retrieve the distribution of the input in self.distribution. \n
        This will slow down the function, as it has to retrieve the input \
        dist.\n

        Arguments:
                saving (bool):
                    If True, inputs passing through the activation function 
                    are saved for distribution visualisation. If set to false,
                    the inputs are not retrieved anymore when data flows 
                    through the activation function
                auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``False``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
                bin_width (float):
                    Default bin width for the histogram.\n
                    Default ``0.1``
                mode (str):
                    The mode for the input retrieve.\n
                    Currently only ``categories`` is supported.
                category_name (str):
                    The name of the category
                    Default ``0``
                save_interval (int): 
                    The interval for which input distributions is saved as data flow in.
                    Useful when trying to track input distributions for a number of epochs
                    Default ``0``
        """

        self.save_interval_dist = int(save_interval)
        self.can_show_inp = True

        if not saving:
            if self.can_show_inp:
                self.logger.warn("Not retrieving input anymore")
                self._handle_inputs.remove()
            self._handle_inputs = None
            return
        if self._handle_inputs is not None:
            # print("Already in retrieve mode")
            return
        
        #get function that creates histogram
        self.histo_func = self.get_mode_func(mode, self.device, True)
        
        inp_distr = self.histo_func(bin_width)
        if category_name is None: 
            name = "default"
            self.curr_cat_name = name
        
        self.__add_category(name, inp_distr)
        

        self.inp_bin_width = bin_width

        self.inputs_saved = 0    
        if auto_stop:
            self._handle_inputs = self.register_forward_hook(_save_inputs_auto_stop)
            self._max_saves = max_saves
        else:
            self._handle_inputs = self.register_forward_hook(_save_inputs)

    def save_gradients(self, saving=True, auto_stop=False, max_saves=1000,
                       bin_width="auto", mode="categories"):
        """
        Will retrieve the distribution of the input in self.distribution. \n
        This will slow down the function, as it has to retrieve the input \
        dist.\n

        Arguments:
                auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``False``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
                bin_width (float):
                    Default bin width for the histogram.\n
                    Default ``0.1``
                mode (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``all``
                category_name (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``0``
        """
        if not saving:
            if self._can_show_grad:
                self.logger.warn("Not retrieving gradients anymore")
                self._handle_grads.remove()
            self._handle_grads = None
            return
        if self._handle_grads is not None:
            # print("Already in retrieve mode")
            return
        
        histo_func = self.get_mode_func(mode, self.device, True)
        self._in_grad_dist = histo_func(bin_width)
        self._out_grad_dist = histo_func(bin_width)
        self._grad_bin_width = bin_width

        if auto_stop:
            self.inputs_saved = 0
            raise NotImplementedError
            # self._handle_grads = self.register_full_backward_hook(_save_gradients_auto_stop)
            self._max_saves = max_saves
        else:
            self._handle_grads = self.register_full_backward_hook(_save_gradients)


    @classmethod
    def __track_history_multiple_classes(cls, instances_list = None, want_track = True):
        all_classes = set()
        for curr_cls in instances_list: 
            if curr_cls not in all_classes:
                curr_cls.logger._track_history(want_track)
                all_classes.add(curr_cls)

    @classmethod
    def load_state_dicts(cls, dicts, input_fcts = None, *args, **kwargs):
        instances_list = cls.get_instance_list(input_fcts)
        assert len(instances_list) == len(dicts), "Number of loaded instances must match number of entries in the dict"
        cls.__track_history_multiple_classes(input_fcts, True)
        for i, instance in enumerate(instances_list):
            instance.load_state_dict(dicts[i], *args, **kwargs)
        cls.__track_history_multiple_classes(input_fcts, False)


    @classmethod
    def state_dicts(cls, input_fcts = None, *args, **kwargs):
        """Returns a list of state dicts for the input ActivationModules input_fcts. If it is none, 
        a list of state dicts from the calling class is returned instead.
        """
        instances_list = cls.get_instance_list(input_fcts)
        state_dicts = []
        cls.__track_history_multiple_classes(input_fcts, True)
        for instance in instances_list:
            curr_dict = instance.state_dict(*args, **kwargs)
            state_dicts.append(curr_dict)
        cls.__track_history_multiple_classes(input_fcts, False)
        return state_dicts

    @classmethod
    def save_all_inputs(cls, input_fcts = None, *args, **kwargs):
        """Saves input that the Activation Functions perceive when data flows through them.

        Arguments:
                input_fcts ((Union(List, Dict, torch.nn.Module))): 
                    The Activation Functions for which the inputs shall be saved. Default ``None``, in 
                    which case the inputs are saved for the calling class. 
                saving (bool):
                    If True, inputs passing through the activation function 
                    are saved for distribution visualisation. If set to false,
                    the inputs are not retrieved anymore when data flows 
                    through the activation function
                auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``False``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
                bin_width (float):
                    Default bin width for the histogram.\n
                    Default ``0.1``
                mode (str):
                    The mode for the input retrieve.\n
                    Currently only ``categories`` is supported.
                category_name (str):
                    The name of the category
                    Default ``0`` 
        """
        instances_list = cls.get_instance_list(input_fcts)
        cls.__track_history_multiple_classes(instances_list, True)
        for instance in instances_list:
            instance.save_inputs(*args, **kwargs)
        cls.__track_history_multiple_classes(instances_list, False)

    @classmethod
    def save_all_gradients(cls, input_fcts = None, *args, **kwargs):
        """
        Saves gradients for all instantiates objects of the called class.
        Args:
            input_fcts ((Union(List, Dict, torch.nn.Module))): 
                The Activation Functions for which the gradients shall be saved. Default ``None``, in 
                which case the gradients are saved for the calling class.
            auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``False``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
                bin_width (float):
                    Default bin width for the histogram.\n
                    Default ``0.1``
                mode (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``all``
                category_name (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``0``
        """
        instances_list = cls.get_instance_list(input_fcts)
        cls.__track_history_multiple_classes(input_fcts, True)
        for instance in instances_list:
            instance.save_gradients(*args, **kwargs)
        cls.__track_history_multiple_classes(input_fcts, False)

    def __repr__(self):
        return f"{self.classname}"
        # if "type" in dir(self):
        #     # return  f"{self.type} ActivationModule at {hex(id(self))}"
        #     return  f"{self.type} ActivationModule"
        # if "__name__" in dir(self.activation_function):
        #     # return f"{self.activation_function.__name__} ActivationModule at {hex(id(self))}"
        #     return f"{self.activation_function.__name__} ActivationModule"
        # return f"{self.activation_function} ActivationModule"

    #TODO: code for plotly
    def show_gradients(self, display=True, tolerance=0.001, title=None,
                       axis=None, writer=None, step=None, label=None, colors=None):
        if not self._can_show_grad:
            self.logger.error("Cannot show gradients of ActivationModule, since no inputs were saved")
            return
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False
        if axis is None:
            with sns.axes_style("whitegrid"):
                # fig, axis = plt.subplots(1, 1, figsize=(8, 6))
                fig, axis = plt.subplots(1, 1, figsize=(20, 12))
        if colors is None or len(colors) != 2:
            colors = ["orange", "blue"]
        dists = [self._in_grad_dist, self._out_grad_dist]
        if label is None:
            labels = ['input grads', 'output grads']
        else:
            labels = [f'{label} (inp)', f'{label} (outp)']
        for distribution, col, label in zip(dists, colors, labels):
            weights, x = distribution.weights, distribution.bins
            if self.distribution_display_mode == "kde" and scipy_imported:
                if len(x) > 5:
                    refined_bins = np.linspace(float(x[0]), float(x[-1]), 200)
                    kde_curv = distribution.kde()(refined_bins)
                    # ax.plot(refined_bins, kde_curv, lw=0.1)
                    axis.fill_between(refined_bins, kde_curv, alpha=0.4,
                                      color=col, label=label)
                else:
                    self.logger.warn("The bin size is too big, bins contain too few "
                                     f"elements.\nbins: {x}")
                    axis.bar([], []) # in case of remove needed
            else:
                axis.bar(x, weights/weights.max(), width=x[1] - x[0],
                         linewidth=0, alpha=0.4, color=col, label=label)
            #TODO: why is this here?
            #distribution.empty()
        if writer is not None:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                self.logger.error("Could not use the given SummaryWriter to add the Rational figure")
        elif display:
            plt.legend()
            plt.show()
        else:
            if axis is None:
                return fig


    @classmethod 
    def register_dataset(cls, dataset, is_overwrite = True, input_fcts = None, bin_width = "auto"):
        """
        Register a Dataset, either from a DataLoader object or a Dataset object of pytorch
        This will set n output distribution (for n labels) which will be plotted when data flows through 
        an ActivationModule. The current output distributions can either be overwritten fully (is_overwrite = True) 
        or extended by the new distributions.
        """

        from torch.utils.data.dataloader import DataLoader
        from torch.utils.data.dataset import Dataset
        assert (isinstance(dataset, Dataset)) or (isinstance(dataset, DataLoader))

        if isinstance(dataset, DataLoader):
            dataset = dataset.dataset 
        
        dataset_labels = torch.unique(dataset.targets)
        dataset_labels = dataset_labels.tolist()
        dataset_labels = [str(cat) for cat in dataset_labels]

        instance_list = cls.get_instance_list(input_fcts)
        for instance in instance_list:
            instance.register_dataset_test(dataset, is_overwrite, bin_width, is_for_input_dist = True)
            #instance.register_dataset_test(dataset_labels, is_overwrite, bin_width, is_fo)

    def register_dataset_test(self, dataset, is_overwrite, bin_width, is_for_input_dist = True):
        dataset_labels = torch.unique(dataset.targets)
        dataset_labels = dataset_labels.tolist()
        dataset_labels = [str(cat) for cat in dataset_labels]

        existing_labels = self.distributions.keys()
        existing_labels = [str(cat_name) for cat_name in existing_labels]

        hist_func = self.get_mode_func(self.dist_mode, self.device, is_for_input_dist)
        for new_label in dataset_labels:
            if new_label in existing_labels:
                if is_overwrite:
                    self.logger.info(f"Overwriting current distribution category {new_label} with new distribution")
                    self.distributions[new_label] = hist_func(bin_width)
                else: 
                    self.logger.info(f"Leaving the distribution {new_label} as it currently is, since overwriting mode is off")
            else: 
                self.distributions[new_label] = hist_func(bin_width)


    """ def register_dataset_test(self, new_labels, is_overwrite, bin_width, is_for_input_dist = True):
        existing_labels = self.distributions.keys()
        existing_labels = [str(cat_name) for cat_name in existing_labels]

        hist_func = self.get_mode_func(self.dist_mode, self.device, is_for_input_dist)
        for new_label in new_labels:
            if new_label in existing_labels:
                if is_overwrite:
                    self.logger.info(f"Overwriting current distribution category {new_label} with new distribution")
                    self.distributions[new_label] = hist_func(bin_width)
                else: 
                    self.logger.info(f"Leaving the distribution {new_label} as it currently is, since overwriting mode is off")
            else: 
                self.distributions[new_label] = hist_func(bin_width) """
                    


    @classmethod
    def show_all_gradients(cls, input_fcts = None, display=True, tolerance=0.001, title=None,
                           axes=None, layout="auto", writer=None, step=None,
                           colors=None):
        """
        Shows a graph of the all instanciated activation functions (or returns \
        it if ``returns=True``).

        Arguments:
                input_fcts (Union(List, Dict, torch.nn.Module)):
                    The input ActivationFunctions from which the gradients are to be visualized.
                    Default ``None``, in that case the instances of the calling
                    class are used for visualization.
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                display (bool):
                    If ``True``, displays the plot.
                    Otherwise, returns the figure. \n
                    Default ``False``
                tolerance (float):
                    If the input histogram is used, it will be pruned. \n
                    Every bin containg less than `tolerance` of the total \
                    input is pruned out.
                    (Reduces noise).
                    Default ``0.001``
                title (str):
                    If not None, a title for the figure
                    Default ``None``
                axes (matplotlib.pyplot.axis):
                    On ax or a list of axes to be plotted on. \n
                    If None, creates them automatically (see `layout`). \n
                    Default ``None``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\n
                    Default ``"auto"``
                writer (tensorboardX.SummaryWriter):
                    A tensorboardX writer to give the image to, in case of
                    debugging.
                    Default ``None``
                step (int):
                    A step/epoch for tensorboardX writer.
                    If None, incrementing itself.
                    Default ``None``
        """
        instances_list = cls.get_instance_list(input_fcts)
        cls.__track_history_multiple_classes(input_fcts, True)
        if axes is None:
            if layout == "auto":
                total = len(instances_list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            figs = tuple(np.flip(np.array(layout)* (2, 3)))
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figs)
            except ImportError:
                cls.logger.warn("Could not import seaborn")
                #RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            # if display:
            for ax in axes.flatten()[len(instances_list):]:
                ax.remove()
            axes = axes[:len(instances_list)]
        elif isinstance(axes, plt.Axes):
            axes = np.array([axes for _ in range(len(instances_list))])
            fig = plt.gcf()
        if isinstance(colors, str) or colors is None:
            colors = [colors]*len(axes.flatten())
        for act, ax, color in zip(instances_list, axes.flatten(), colors):
            act.show_gradients(False, tolerance, title, axis=ax,
                               writer=None, step=step, colors=color)
        if title is not None:
            fig.suptitle(title, y=0.95)
        fig = plt.gcf()
        fig.tight_layout()
        if writer is not None:
            if step is None:
                step = cls._step
                cls._step += 1
            writer.add_figure(title, fig, step)
        elif display:
            plt.legend()
            plt.show()
            cls.__track_history_multiple_classes(input_fcts, False)
        else:
            cls.__track_history_multiple_classes(input_fcts, False)
            return fig

    def save_histo(self):
        #TODO: implement, need to take care of different distributions due to categories and how to best save them
        pass


    def give_axis(self, x = None):
        if x is None:
            x = torch.arange(-3., 3, 0.01)
        elif isinstance(x, tuple) and len(x) in (2, 3):
            x = torch.arange(*x).float()
        elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            x = torch.tensor(x.float())
        return x


    def show_plotly(self, x=None, display=True, tolerance=0.001, 
        title=None, fig=None, writer=None, step=None,
        color=None, subplot_index = None):

        #Construct x axis
        if not self.can_show_inp:
            self.logger.error("Cannot show input distribution, since no inputs were saved for it")
            return

        #create axis range depending on input x
        x = self.give_axis(x)
        if fig is None:
            fig = go.Figure()
        if self.distributions:
            if self.distribution_display_mode in ["kde", "bar"]:
                x = self.plot_distributions_plotly(fig, subplot_index, color)
            elif self.distribution_display_mode == "points":
                #TODO: IMPLEMENT
                raise NotImplementedError("need to implement")


        y = self.forward(x.to(self.device)).detach().cpu().numpy()
        if subplot_index is not None: 
            fig.add_trace(go.Scatter(x = x, y = y, mode="lines"), row = subplot_index[0], col=subplot_index[1])
        else: 
            fig.add_trace(go.Scatter(x = x, y = y, mode="lines"))
        if not display: 
            print("TODO WRITE FIGURES TO CALLING FILE IF NO PATH IS GIVEN")
            
        elif display:
            fig.show()


    def set_display_mode(self, value):
        assert value in ["kde", "bar", "points"], f"Display mode: {value} is not allowed"
        self.distribution_display_mode = value

    @property
    def inp_bin_width(self):
        return self.bin_width

    @inp_bin_width.setter
    def inp_bin_width(self, value):
        try:
            if "auto" != value:
                value = float(value)
            self.bin_width = value
        except ValueError:
            self.logger.warn(f'''
                Passed value is not convertable to number, 
                staying with original value, which is {self.inp_bin_width}
            ''')
        


    """ @property
    def current_inp_distribution(self):
        return self.distributions[-1]
                
    @property
    def current_inp_category(self):
        return self.categories[-1]

    @current_inp_category.setter
    def current_inp_category(self, value):
        value = str(value)


        #if the histogram is empty, it means that is was created at the same phase
        #that the current category is created, which means that no input was perceived
        #during this time -> redundant category
    
        for i in range(len(self.distributions)):
            if self.distributions[i].is_empty:
                del self.distributions[i]
                del self.categories[i]
        new_distribution = self.histo_func(self.inp_bin_width)
        self.distributions.append(new_distribution)
        self.categories.append(value) """

    def __is_dist_per_layer(self):
        per_layer = self.dist_mode == "layer"
        return per_layer

 
    def plot_distributions_plotly(self, fig, subplot_index = None, colors=None): 
        """
        Plot the distribution and returns the corresponding x
        """
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False

        if colors is None or not(isinstance(colors, list) or isinstance(colors, tuple)):
            colors = create_colors(len(self.distributions))
        
        plotting_data = dict()
        is_per_layer = self.__is_dist_per_layer()
        for i, (distribution, inp_label, color) in enumerate(zip(self.distributions.items(), self.distributions.keys(), colors)):
            if not distribution.is_empty:
                curr_plot_data, num_dists = self.__get_plotting_data(distribution, per_neuron = not is_per_layer)
                if curr_plot_data is not None: 
                    if is_per_layer:
                        plotting_data[inp_label] = curr_plot_data
                    else: 
                        plotting_data[inp_label] = curr_plot_data

        #TODO: add horizontal slider
        if self.distribution_display_mode == "kde":
            for cat_name in plotting_data.keys():
                layer_data = plotting_data[cat_name]
                for i, (x_plot, y_plot) in enumerate(layer_data):
                    if not is_per_layer:
                        label = f"{cat_name}: Neuron {i}"
                    else: 
                        label = cat_name
                    if subplot_index is not None:
                        fig.add_trace(go.Scatter(name=label, x = x_plot, y = y_plot, mode="lines", fill="tozeroy"), row = subplot_index[0], col = subplot_index[1])
                    else: 
                        fig.add_trace(go.Scatter(name=label, x = x_plot, y = y_plot, mode="lines", fill="tozeroy"), row = subplot_index[0])
        elif self.distribution_display_mode == "bar": 
            for cat_name in plotting_data.keys():
                layer_data = plotting_data[cat_name]
                for i, (x_plot, y_plot) in enumerate(layer_data):
                    if not is_per_layer:
                        label = f"{cat_name}: Neuron {i}"
                    else: 
                        label = cat_name
                    if subplot_index is not None:
                        fig.add_trace(go.Bar(name=label, x = x_plot, y = y_plot), row = subplot_index[0], col = subplot_index[1])
                    else: 
                        fig.add_trace(go.Bar(name=label, x = x_plot, y = y_plot))


        #TODO: when distribution is always empty, size wont be assigned and will throw an error
        return torch.arange(*self.get_distributions_range())


    def __get_plotting_data(self, distribution, per_neuron = False):
        """
        Takes a distribution and whether it has a single distribution per layer or a distribution per layer (per_neuron = True).
        Returns an array of plotting data for each output distribution 
        and the number of output distributions (for layer it should maximally be 1)
        """
        def get_data(x, weights, display_mode, n = None):
            #x := bins, y := histogram function
            x_data = y_data = None
            if display_mode == "kde":
                if len(x) > 5: 
                    x_data = np.linspace(float(x[0]), float(x[-1]), 200)
                    if n is not None: 
                        y_data = distribution.kde(n)(x_data)
                    else: 
                        y_data = distribution.kde()(x_data)
                else:
                    self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
            else:
                #x is originally defined to be x + i * width for i'th category
                #but this is omitted since it doesn't make much of a difference
                x_data = x 
                y_data = weights / weights.max()
            
            if x_data is not None and y_data is not None: 
                return (x_data, y_data)
            else: 
                return None

            
        plotting_data = []
        num_inputs = 0
        if per_neuron:
            for n, (weights, x) in enumerate(zip(distribution.weights, distribution.bins)):
                curr_neuron_data = get_data(x, weights, self.distribution_display_mode, n)
                if curr_neuron_data is not None: 
                    plotting_data.append(curr_neuron_data)
                    num_inputs += 1
        else: 
            weights, x = _cleared_arrays(distribution.weights, distribution.bins, 0.001)
            plot_data = get_data(x, weights, self.distribution_display_mode)
            if plot_data is not None: 
                plotting_data.append(plot_data)
                num_inputs += 1
            
        return plotting_data, num_inputs


    def get_distributions_range(self):
        x_min, x_max = np.inf, -np.inf
        for dist in self.distributions.items():
            if not dist.is_empty:
                x_min, x_max = min(x_min, dist.range[0]), max(x_max, dist.range[-1])
                size = dist.range[1] - dist.range[0]
        if x_min == np.inf or x_max == np.inf:
            return -3, 3, 0.01
        return x_min, x_max, size

    @classmethod
    def get_instance_list(cls, input_fcts = None):
        """ 

        Arguments:
            input_fcts (Union(List, Dict, torch.nn.Module)):
                The input ActivationFunctions which are to be retrieved.
                Default ``None``, in that case the instances of the calling
                class are used for retrieving.

        Returns:
            list: A list of Activation Functions.
        """
        needsModifying = True
        if input_fcts is None:
            needsModifying = False
            instances_list = cls._get_instances()
        elif isinstance(input_fcts, torch.nn.Module):
            instances_list = get_toplevel_functions(input_fcts, cls)
        elif type(input_fcts) is  list:
            instances_list = input_fcts
        elif type(input_fcts) is dict:
            instances_list = []
            for key in input_fcts:
                curr_instc = input_fcts[key]
                if isinstance(curr_instc, ActivationModule):
                    instances_list.append(curr_instc)
                elif type(curr_instc) is list: 
                    instances_list.extend(curr_instc)
        if needsModifying:
            new_list = []
            for inst in instances_list:
                if not issubclass(type(inst), cls):
                    cls.logger.warn(f"Removed {inst} since it's not a Submodule of {cls} class")
                else:
                    new_list.append(inst)
            instances_list = new_list
            cls.logger.info(f"Returning modified list {instances_list}")    
        if len(instances_list) == 0:
            cls.logger.critical("Empty instance list, cannot be used for visualization purposes")
            raise ValueError()
        return instances_list


    @classmethod
    def _get_instances(cls):
        """
        if called from ActivationModule: returning all instanciated functions
        if called from a child-class: returning the instances of this specific class
        """
        if "ActivationModule" in str(cls):
            instances_list = []
            [instances_list.extend(insts) for insts in cls.instances.values()]
        else:
            clsn = str(cls)
            if "activations.torch" in clsn:
                curr_classname = clsn.split("'")[1].split(".")[-1]
                if curr_classname not in cls.instances:
                    print(f"No instanciated function of {curr_classname} found")
                    return []
                instances_list = cls.instances[curr_classname]
            else:
                print(f"Unknown {cls} for show_all")  # shall never happen
                return []
        return instances_list
    
    @classmethod
    def set_display_mode(cls, display_mode, input_fcts = None):
        instances_list = cls.get_instance_list(input_fcts)
        for instance in instances_list:
            instance.set_display_mode(display_mode)
                


    @classmethod
    def show_all(cls, input_fcts=None, x=None, show_method = "display", 
                tolerance=0.001, title=None, axes=None, 
                step=None):

        assert show_method in ["display", "save"], "Figures can be either saved or shown to the user"
        want_display = show_method == "display"
        instances_list = cls.get_instance_list(input_fcts)
        total = len(instances_list)
        layout = _get_auto_axis_layout(total)
        rows = layout[0]
        cols = layout[1]
        colors = create_colors(total)
        fig = make_subplots(rows, cols)
        for curr_row in range(rows):
            for curr_col in range(cols):
                act = instances_list[curr_row * rows + curr_col]
                x_act = instances_list[curr_row * rows + curr_col]
                color = colors[curr_row * rows + curr_col]
                act.show_plotly(x_act, want_display, tolerance, title, fig,
                                step, color, [curr_row + 1, curr_col + 1], display = False)

        fig.show()

        

    def load_state_dict(self, state_dict):
        if "distributions" in state_dict.keys():
            _distributions = state_dict.pop("distributions")
            _inp_category = state_dict.pop("inp_category")
            created_distributions, msg = af_utils.create_histograms(_distributions, self.device)
            self.logger.info(msg)
            self.distributions = created_distributions
            self.current_inp_category = _inp_category
        if "in_grad_dist" in state_dict.keys():
            _in_grad_dist = state_dict.pop("in_grad_dist")
            _out_grad_dist = state_dict.pop("out_grad_dist")
            created_grad_dist, msg = af_utils.create_histograms([_in_grad_dist, _out_grad_dist], self.device)
            self._in_grad_dist = created_grad_dist[0]
            self._out_grad_dist = created_grad_dist[1]
        super().load_state_dict(state_dict)



    def state_dict(self, destination=None, *args, **kwargs):
        _state_dict = super().state_dict(destination, *args, **kwargs)
        if self.distributions is not None:
            saved_distributions = []
            saved_categories = []
            for cat_name in self.distributions:
                curr_dist = self.distributions[cat_name]
                if curr_dist.is_empty:
                    self.logger.warn(f"Not saving distribution for category {cat_name}, since it is empty")
                else: 
                    saved_distributions.append(curr_dist)
                    saved_categories.append(cat_name)

            _state_dict["distributions"] = saved_distributions
            _state_dict["inp_category"] = saved_categories

        #TODO: do for gradients
        if self._in_grad_dist is not None:
            if self._in_grad_dist.is_empty:
                pass
                #""" self.logger.warn("""deleting input and output gradient distribution since it is empty,
                #   at position: """ + i) """
            else: 
                _state_dict["in_grad_dist"] = self._in_grad_dist
                _state_dict["out_grad_dist"] = self._out_grad_dist
        return _state_dict