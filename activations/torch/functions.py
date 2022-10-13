from multiprocessing.sharedctypes import Value
from sklearn import neural_network
import torch
import torch.nn.functional as F
from activations.utils.utils import _get_auto_axis_layout, _cleared_arrays
from activations.utils.warnings import RationalImportScipyWarning
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termcolor import colored
from random import randint
import activations.torch.utils.af_utils as af_utils
from activations.torch.utils.af_utils import create_colors, get_toplevel_functions
import logging
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

_LINED = dict()

def _save_inputs(self, input, output):
    if self.current_inp_distribution is None:
        raise ValueError("Selected distribution is none")
    self.current_inp_distribution.fill_n(input[0])


def _save_gradients(self, in_grad, out_grad):
    self._in_grad_dist.fill_n(in_grad[0])
    self._out_grad_dist.fill_n(out_grad[0])


def _save_inputs_auto_stop(self, input, output):
    self.inputs_saved += 1
    if self.current_inp_distribution is None:
        raise ValueError("Selected distribution is none")
    self.current_inp_distribution.fill_n(input[0])
    if self.inputs_saved > self._max_saves:
        self.training_mode()


class Metaclass(type):
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            key_str = colored(key, "red")
            self_name_str = colored(self, "red")
            msg = colored(f"Setting new Class attribute {key_str}", "yellow") + \
                  colored(f" of {self_name_str}", "yellow")
            print(msg)
        type.__setattr__(self, key, value)


class ActivationModule(torch.nn.Module):#, metaclass=Metaclass):
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

        """ self._handle_inputs = None
        self._handle_grads = None
        self._saving_input = False
        self.distributions = []
        self.categories = ["distribution"]
        self._selected_distribution = None
        self._selected_distribution_name = "distribution" """

        self.vis_mode = "plotlib"
        if self.vis_mode == "plotlib": 
            self.show_method = self.show
        elif self.vis_mode == "plotly": 
            self.show_method = self.show_plotly
        self._init_inp_distributions()
        self._init_grad_distributions()

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.use_kde = True

    def _init_grad_distributions(self):
        self._handle_grads = None
        self._in_grad_dist = None
        self._out_grad_dist = None
        self._grad_bin_width = "auto"
        self._can_show_grad = False


    def _init_inp_distributions(self):
        self.categories = ["distribution"]
        self.distributions = None
        self._saving_input = False
        self.mode = "categories"
        self._handle_inputs = None
        self.inp_bin_width = "auto"
        self.can_show_inp = False



    """ @property
    def mode(self):
        return self.mode 

    @mode.setter
    def mode(self, value):
        value = str(value)
        if value == self.mode:
            self.logger.info("Mode is already in the specified one")
            return
        allowed_modes = ["categories"]
        if value.lower() in allowed_modes: 
            self.mode = value
            from .utils.histograms_numpy import Histogram
            self.histo_func = Histogram """

    def get_mode_func(self, value, device):
        value = str(value)
        #allowed_modes = ["categories", "neurons", "neurons_categories"]
        allowed_modes = ["categories", "neurons", "neurons_categories"]
        if value.lower() in allowed_modes: 
            can_cupy = af_utils.can_use_cupy(device)
            if can_cupy:
                if "neurons" in value:
                    from activations.torch.utils.histograms_cupy import NeuronsHistogram as hist   
                    if self.vis_mode == "plotlib":
                        self.plotting_func = self.plot_layer_distributions
                    elif self.vis_mode == "plotly":
                        self.plotting_func = self.plot_distributions_plotly
                else: 
                    from activations.torch.utils.histograms_cupy import Histogram as hist
                    if self.vis_mode == "plotlib":
                        self.plotting_func = self.plot_distributions
                    elif self.vis_mode == "plotly": 
                        self.plotting_func = self.plot_distributions_plotly
                histo_func = hist
            else: 
                if "neurons" in value: 
                    from activations.torch.utils.histograms_numpy import NeuronsHistogram as hist
                    if self.vis_mode == "plotlib": 
                        self.plotting_func = self.plot_layer_distributions
                    elif self.vis_mode == "plotly":
                        self.plotting_func = self.plot_distributions_plotly
                else:
                    from activations.torch.utils.histograms_numpy import Histogram as hist
                    if self.vis_mode == "plotlib":
                        self.plotting_func = self.plot_distributions
                    elif self.vis_mode == "plotly": 
                        self.plotting_func = self.plot_distributions_plotly
                histo_func = hist
            return histo_func
        else: 
            self.logger.critical("Mode is currently not supported")
            raise ValueError()


    @property
    def classname(self):
        clsn = str(self.__class__)
        if "activations.torch" in clsn:
            return clsn.split("'")[1].split(".")[-1]
        else:
            return "Unknown"  # TODO, implement

    def save_inputs(self, saving=True, auto_stop=False, max_saves=1000,
                    bin_width=0.1, mode="categories", category_name=None):
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
        """
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
        self.histo_func = self.get_mode_func(mode, self.device)
        
        inp_distr = self.histo_func(bin_width)
        if category_name is not None:
            self.categories = [category_name]
        
        self.distributions = [inp_distr]
        

        self.inp_bin_width = bin_width

        
        if auto_stop:
            self.inputs_saved = 0
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
        
        histo_func = self.get_mode_func(mode, self.device)
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
    def __track_history_multiple_classes(cls, input_fcts = None, want_track = True):
        if input_fcts is None: 
            cls.logger._track_history(want_track)
        else:
            all_classes = set()
            for curr_cls in input_fcts: 
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
        cls.__track_history_multiple_classes(input_fcts, True)
        for instance in instances_list:
            instance.save_inputs(*args, **kwargs)
        cls.__track_history_multiple_classes(input_fcts, False)

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



    def give_axis(self, x = None):
        if x is None:
            x = torch.arange(-3., 3, 0.01)
        elif isinstance(x, tuple) and len(x) in (2, 3):
            x = torch.arange(*x).float()
        elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            x = torch.tensor(x.float())
        return x


    def show_plotly(self, x=None, fitted_function=True, other_func=None, display=True,
             tolerance=0.001, title=None, fig=None, writer=None, step=None, label=None,
             color=None):

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
                x = self.plotting_func(fig)
            elif self.distribution_display_mode == "points":
                #TODO: IMPLEMENT
                raise NotImplementedError("need to implement")


        y = self.forward(x.to(self.device)).detach().cpu().numpy()
        if label:
            fig.add_trace(go.Scatter(x = x, y = y, mode="lines"))
        if writer is not None:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                self.logger.error("Could not use the given SummaryWriter to add the Rational figure")
        elif display:
            fig.show()

    def show(self, x=None, fitted_function=True, other_func=None, display=True,
             tolerance=0.001, title=None, axis=None, writer=None, step=None, label=None,
             color=None):

        #Construct x axis
        if not self.can_show_inp:
            self.logger.error("Cannot show input distribution, since no inputs were saved for it")
            return

        #create axis range depending on input x
        x = self.give_axis(x)

        if axis is None:
            with sns.axes_style("whitegrid"):
                # fig, axis = plt.subplots(1, 1, figsize=(8, 6))
                fig, axis = plt.subplots(1, 1, figsize=(20, 12))
        if self.distributions:
            if self.distribution_display_mode in ["kde", "bar"]:
                ax2 = axis.twinx()
                x = self.plotting_func(ax2)
            elif self.distribution_display_mode == "points":
                x0, x_last, _ = self.get_distributions_range()
                x_edges = torch.tensor([x0, x_last]).float()
                y_edges = self.forward(x_edges.to(self.device)).detach().cpu().numpy()
                axis.scatter(x_edges, y_edges, color=color)


        y = self.forward(x.to(self.device)).detach().cpu().numpy()
        if label:
            # axis.twinx().plot(x, y, label=label, color=color)
            axis.plot(x, y, label=label, color=color)
        else:
            # axis.twinx().plot(x, y, label=label, color=color)
            axis.plot(x, y, label=label, color=color)
        if writer is not None:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                self.logger.error("Could not use the given SummaryWriter to add the Rational figure")
        elif display:
            plt.show()
        else:
            if axis is None:
                return fig

    def change_category(cls, value, input_fcts=None):
        """ Changes the input category of the ActivationFunctions passed
        in input_fcts / calling class depending on the parameters.
        This will create a new distribution on new input when visualizing the respective 
        Activation Functions.

        Arguments:
            value (String): The name of the new category for visualisation
            input_fcts ((Union(List, Dict, torch.nn.Module))): The ActivationFunctions 
            for which the category should be changed. Default ``None``, in which case 
            the instances of the classes get assigned a new category.
        """
        value = str(value)
        instances = cls.get_instance_list(input_fcts)
        for inst in instances:
            inst.current_inp_category = value


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
        

    @property
    def current_inp_distribution(self):
        return self.distributions[-1]
                
    @property
    def current_inp_category(self):
        return self.categories[-1]

    @current_inp_category.setter
    def current_inp_category(self, value):
        value = str(value)
        #TODO: do we want to prevent users from creating two different 
        #distributions under the same name?
        """ if value == self.current_inp_category:
            return """
        #if the histogram is empty, it means that is was created at the same phase
        #that the current category is created, which means that no input was perceived
        #during this time -> redundant category
    
        for i in range(len(self.distributions)):
            if self.distributions[i].is_empty:
                del self.distributions[i]
                del self.categories[i]
        new_distribution = self.histo_func(self.inp_bin_width)
        self.distributions.append(new_distribution)
        self.categories.append(value)

    def plot_distributions(self, ax, colors=None, bin_size=None):
        """
        Plot the distribution and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False

        
        dists_fb = []
        x_min, x_max = np.inf, -np.inf
        #TODO: this is obsolete afaik
        """ if colors is None:
            colors = self.histograms_colors """
        if not(isinstance(colors, list) or isinstance(colors, tuple)):
            colors = create_colors(len(self.distributions))

        for i, (distribution, inp_label, color) in enumerate(zip(self.distributions, self.categories, colors)):
            #if distribution is empty, fill it with empty stuff
            if distribution.is_empty:
                if self.distribution_display_mode == "kde" and scipy_imported:
                    fill = ax.fill_between([], [], label=inp_label,  alpha=0.)
                else:
                    fill = ax.bar([], [], label=inp_label,  alpha=0.)
                dists_fb.append(fill)
            #fill it with values
            else:
                weights, x = _cleared_arrays(distribution.weights, distribution.bins, 0.001)
                # weights, x = distribution.weights, distribution.bins
                if self.distribution_display_mode == "kde" and scipy_imported:
                    if len(x) > 5:
                        refined_bins = np.linspace(x[0], x[-1], 200)
                        kde_curv = distribution.kde()(refined_bins)
                        # ax.plot(refined_bins, kde_curv, lw=0.1)
                        fill = ax.fill_between(refined_bins, kde_curv, alpha=0.45,
                                               color=color, label=inp_label)
                    else:
                        self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
                        fill = ax.bar([], []) # in case of remove needed
                    size = x[1] - x[0]
                else:
                    width = (x[1] - x[0])/len(self.distributions)
                    if len(x) == len(weights):
                        fill = ax.bar(x+i*width, weights/weights.max(), width=width,
                                  linewidth=0, alpha=0.7, label=inp_label)
                    else:
                        fill = ax.bar(x[1:]+i*width, weights/weights.max(), width=width,
                                  linewidth=0, alpha=0.7, label=inp_label)
                    size = (x[1] - x[0])/100 # bar size can be larger
                dists_fb.append(fill)
                x_min, x_max = min(x_min, x[0]), max(x_max, x[-1])

        if self.distribution_display_mode in ["kde", "bar"]:
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(0.4)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                _LINED[legline] = origline
            fig = plt.gcf()
            def toggle_distribution(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg = event.artist
                orig = _LINED[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()[0]
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines
                # have been toggled
                if vis:
                    leg.set_alpha(0.4)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_distribution)
        if x_min == np.inf or x_max == np.inf:
            torch.arange(-3, 3, 0.01)
        #TODO: when distribution is always empty, size wont be assigned and will throw an error

        return torch.arange(x_min, x_max, size)


    def plot_layer_distributions(self, ax):
        """
        Plot the layer distributions and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
        dists_fb = []
        colors = create_colors(len(self.distributions))
        for distribution, inp_label, color in zip(self.distributions, self.categories, colors):
            #TODO: why is there no empty distribution check here?
            for n, (weights, x) in enumerate(zip(distribution.weights, distribution.bins)):
                if self.distribution_display_mode == "kde" and scipy_imported:
                    if len(x) > 5:
                        refined_bins = np.linspace(float(x[0]), float(x[-1]), 200)
                        kde_curv = distribution.kde(n)(refined_bins)
                        # ax.plot(refined_bins, kde_curv, lw=0.1)
                        fill = ax.fill_between(refined_bins, kde_curv, alpha=0.4,
                                                color=color, label=f"{inp_label} ({n})")
                    else:
                        self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
                        fill = ax.bar([], []) # in case of remove needed
                else:
                    fill = ax.bar(x, weights/weights.max(), width=x[1] - x[0],
                                  linewidth=0, alpha=0.4, color=color,
                                  label=f"{inp_label} ({n})")
                dists_fb.append(fill)

        if self.distribution_display_mode in ["kde", "bar"]:
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(0.4)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                _LINED[legline] = origline
            fig = plt.gcf()
            def toggle_distribution(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg = event.artist
                orig = _LINED[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()[0]
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines
                # have been toggled
                if vis:
                    leg.set_alpha(0.4)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_distribution)
        return torch.arange(*self.get_distributions_range())
        

    def plot_distributions_plotly(self, fig, colors=None): 
        """
        Plot the distribution and returns the corresponding x
        """
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False

        
        x_min, x_max = np.inf, -np.inf
        #TODO: this is obsolete afaik
        """ if colors is None:
            colors = self.histograms_colors """
        if colors is None or not(isinstance(colors, list) or isinstance(colors, tuple)):
            colors = create_colors(len(self.distributions))
        
        #dict of data where keys are names of the categories and items are tuples of (x_data, y_data)
        plotting_data = dict()
        for i, (distribution, inp_label, color) in enumerate(zip(self.distributions, self.categories, colors)):
            if not distribution.is_empty:
                weights, x = _cleared_arrays(distribution.weights, distribution.bins, 0.001)
                # weights, x = distribution.weights, distribution.bins
                if self.distribution_display_mode == "kde" and scipy_imported:
                    if len(x) > 5:
                        x_data = np.linspace(x[0], x[-1], 200)
                        y_data = distribution.kde()(x_data)
                    else:
                        self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
                else:
                    width = (x[1] - x[0])/len(self.distributions)
                    if len(x) == len(weights):
                        x_data = x+i*width
                        y_data = weights/weights.max()
                plotting_data[inp_label] = (x_data, y_data)


        #TODO: add horizontal slider
        if self.distribution_display_mode == "kde":
            for cat_name in plotting_data.keys():
                fig.add_trace(go.Scatter(name=cat_name, x = plotting_data[cat_name][0], y = plotting_data[cat_name][1], mode="lines", fill="tozeroy"))
        elif self.distribution_display_mode == "bar": 
            for cat_name in plotting_data.keys():
                fig.add_trace(go.Bar(name=cat_name, x = plotting_data[cat_name][0], y = plotting_data[cat_name][1]))


        #TODO: when distribution is always empty, size wont be assigned and will throw an error
        return torch.arange(*self.get_distributions_range())


    def plot_layer_distributions_plotly(self, fig):
        """
        Plot the layer distributions and returns the corresponding x
        """
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
        colors = create_colors(len(self.distributions))
        plotting_data = dict()
        
        for distribution, inp_label, color in zip(self.distributions, self.categories, colors):
            #for all neurons in current distribution get plotting dict
            plotting_data[inp_label] = self.__get_plotting_data(distribution, per_neuron = True)

        #plotting
        if self.distribution_display_mode == "kde":
            for cat_name in plotting_data.keys():
                layer_data = plotting_data[cat_name]
                for i, (x_plot, y_plot) in enumerate(layer_data):
                    label = f"{cat_name}: Neuron {i}"
                    fig.add_trace(go.Scatter(name=label, x = x_plot, y = y_plot, mode="lines", fill="tozeroy"))
        elif self.distribution_display_mode == "bar": 
            for cat_name in plotting_data.keys():
                layer_data = plotting_data[cat_name]
                for i, (x_plot, y_plot) in enumerate(layer_data):
                    label = f"{cat_name}: Neuron {i}"
                    fig.add_trace(go.Bar(name=label, x = x_plot, y = y_plot))

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
        for dist in self.distributions:
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
            instances_list = get_toplevel_functions(input_fcts)
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
    def show_all(cls, input_fcts=None, x=None, fitted_function=True, other_func=None,
                 display=True, tolerance=0.001, title=None, axes=None,
                 layout="auto", writer=None, step=None, colors="#1f77b4"):
        """
        Shows a graph of the all instanciated activation functions (or returns \
        it if ``returns=True``).

        Arguments:
                input_fcts (Union(List, Dict, torch.nn.Module)):
                    The input ActivationFunctions which are to be visualized.
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
        if isinstance(colors, str):
            colors = [colors]*len(axes.flatten())
        if isinstance(x, list):
            for act, ax, x_act, color in zip(instances_list, axes.flatten(), x, colors):
                act.show_method(x_act, fitted_function, other_func, False, tolerance,
                         title, axis=ax, writer=None, step=step,
                         color=color)
        else:
            for act, ax, color in zip(instances_list, axes.flatten(), colors):
                act.show_method(x, fitted_function, other_func, False, tolerance,
                         title, axis=ax, writer=None, step=step,
                         color=color)
        if title is not None:
            fig.suptitle(title, y=0.95)
        fig = plt.gcf()
        fig.tight_layout()
        if writer is not None:
            if step is None:
                step = cls._step
                cls._step += 1
            writer.add_figure(title, fig, step)
            cls.__track_history_multiple_classes(input_fcts, False)
        elif display:
            # plt.legend()
            plt.show()
            cls.__track_history_multiple_classes(input_fcts, False)
        else:
            cls.__track_history_multiple_classes(input_fcts, False)
            return fig

    # def __setattr__(self, key, value):
    #     if not hasattr(self, key):
    #         key_str = colored(key, "red")
    #         self_name_str = colored(self.__class__, "red")
    #         msg = colored(f"Setting new attribute {key_str}", "yellow") + \
    #               colored(f" of instance of {self_name_str}", "yellow")
    #         print(msg)
    #     object.__setattr__(self, key, value)



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
            for i in range(len(self.distributions)):
                if self.distributions[i].is_empty:
                    self.logger.warn("""Deleting input distribution histogram
                                    since it is empty, at position: """  + i)
                else: 
                    saved_distributions.append(self.distributions[i])
                    saved_categories.append(self.categories[i])

            _state_dict["distributions"] = saved_distributions
            _state_dict["inp_category"] = self.categories

        if self._in_grad_dist is not None:
            if self._in_grad_dist.is_empty:
                self.logger.warn("""deleting input and output gradient distribution since it is empty,
                    at position: """ + i)
            else: 
                _state_dict["in_grad_dist"] = self._in_grad_dist
                _state_dict["out_grad_dist"] = self._out_grad_dist
        return _state_dict


if __name__ == '__main__':
    def plot_gaussian(mode, device):
        _2pi_sqrt = 2.5066
        tanh = torch.tanh
        relu = F.relu

        nb_neurons_in_layer = 5

        leaky_relu = F.leaky_relu
        gaussian = lambda x: torch.exp(-0.5*x**2) / _2pi_sqrt
        gaussian.__name__ = "gaussian"
        gau = ActivationModule(gaussian, device=device)
        gau.save_inputs(mode=mode, category_name="neg") # Wrong
        inp = torch.stack([(torch.rand(10000)-(i+1))*2 for i in range(nb_neurons_in_layer)], 1)
        gau(inp.to(device))
        if "categories" in mode:
            gau.current_inp_category = "pos"
            inp = torch.stack([(torch.rand(10000)+(i+1))*2 for i in range(nb_neurons_in_layer)], 1)
            gau(inp.to(device))
            # gau(inp.cuda())
        gau.show_method()

    ActivationModule.distribution_display_mode = "bar"
    # for device in ["cuda:0", "cpu"]:
    for device in ["cpu"]:
        for mode in ["neurons"]:
            plot_gaussian(mode, device)
