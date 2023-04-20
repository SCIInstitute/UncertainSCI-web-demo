r"""
Define your classes and create the instances that you need to expose
"""
import logging
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, plotly

import plotly.graph_objects as go
import plotly.express as px

from . import ui, state_manager



import numpy as np
from UncertainSCI.distributions import BetaDistribution, UniformDistribution, ExponentialDistribution, NormalDistribution
from UncertainSCI.model_examples import laplace_ode_1d, sine_modulation
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot

import jsonpickle


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------




def on_event(type, e):
    print(type, e)
    


class PCE_Model:
    def __init__(self):
        self.pce = None
        self.x = np.empty(1)
        
    def clear(self):
        self.__init__()
        
    def set_pce(self, pce):
        self.pce = pce
    
    def set_x(self, x):
        self.x = x
        
    def set(self, attr, value):
        eval(attr+" = "+value)
        
pce_model = PCE_Model()
    
        
    

class Engine: # same as controller?
    def __init__(self, server=None):
        if server is None:
            server = get_server()
            
        Nparams = 3

        self._server = server
        

        # initialize state + controller
        state, ctrl = server.state, server.controller
        
        state_manager.initialize(server)
        
        state.x = np.empty(1)
        state.pce = None

        # Bind instance methods to controller
#        ctrl.reset_params = self.reset_params
#        ctrl.reset_model = self.reset_model
#        ctrl.reset_all = self.reset_all
        ctrl.on_server_reload = ui.initialize
                
        ctrl.widget_click = self.widget_click
        ctrl.widget_change = self.widget_change
        ctrl.build_model = self.build_model
        ctrl.update_plot = self.update_plot
        ctrl.change_model = self.change_model
        ctrl.change_distribution = self.change_distribution

        # Bind instance methods to state change
#        state.change("active_plot")(self.update_plot)
        state.change("model_name")(self.change_model)
        state.change("dist")(self.change_distribution)

        

    @property
    def server(self):
        return self._server

    @property
    def state(self):
        return self.server.state

    @property
    def ctrl(self):
        return self.server.controller
    
    def show_in_jupyter(self, **kwargs):
        from trame.app import jupyter
        
        logger.setLevel(logging.WARNING)
        jupyter.show(self._server, **kwargs)
        
    def build_model(self):
        
#        pce_model.clear()
    
#        dist_type = [self._server.state.dist_0, self._server.state.dist_1, self._server.state.dist_2 ]
#        alpha = [self._server.state.alpha_0, self._server.state.alpha_1, self._server.state.alpha_2 ]
#        beta = [self._server.state.beta_0, self._server.state.beta_1, self._server.state.beta_2 ]
#        mean = [self._server.state.mean_0, self._server.state.mean_1, self._server.state.mean_2 ]
#        cov = [self._server.state.cov_0, self._server.state.cov_1, self._server.state.cov_2 ]
#        domain = [
#            [ self._server.state.dist_domain_min_0, self._server.state.dist_domain_max_0 ],
#            [ self._server.state.dist_domain_min_1, self._server.state.dist_domain_max_1 ],
#            [ self._server.state.dist_domain_min_2, self._server.state.dist_domain_max_2 ]
#        ]
#        constant = [ self._server.state.constant_0, self._server.state.constant_1, self._server.state.constant_2]
        
        
        
        
#        param0_dict = {
#            "alpha" : self._server.state.alpha_0,
#            "beta" : self._server.state.beta_0,
#            "mean" : self._server.state.dist_mean_0,
#            "cov" : self._server.state.dist_cov_0,
#            "loc" : self._server.state.loc_0,
#            "lbd" : self._server.state.lbd_0,
#            "constant" : self._server.state.constant_0,
#            "domain" : np.array([[ self._server.state.dist_domain_min_0], [self._server.state.dist_domain_max_0 ]])
#        }
#
#        param1_dict = {
#            "alpha" : self._server.state.alpha_1,
#            "beta" : self._server.state.beta_1,
#            "mean" : self._server.state.dist_mean_1,
#            "cov" : self._server.state.dist_cov_1,
#            "loc" : self._server.state.loc_1,
#            "lbd" : self._server.state.lbd_1,
#            "constant" : self._server.state.constant_1,
#            "domain" : np.array([[ self._server.state.dist_domain_min_1], [self._server.state.dist_domain_max_1 ]])
#        }
#
#        param2_dict = {
#            "alpha" : self._server.state.alpha_2,
#            "beta" : self._server.state.beta_2,
#            "mean" : self._server.state.dist_mean_2,
#            "cov" : self._server.state.dist_cov_2,
#            "loc" : self._server.state.loc_2,
#            "lbd" : self._server.state.lbd_2,
#            "constant" : self._server.state.constant_2,
#            "domain" : np.array([[ self._server.state.dist_domain_min_2], [self._server.state.dist_domain_max_2 ]])
#        }
#
        
        state.x = np.empty(1)
        state.pce = None
    
        
        param_dict = {
            "alpha" : [self._server.state.alpha_0, self._server.state.alpha_1, self._server.state.alpha_2],
            "beta" : [self._server.state.beta_0, self._server.state.beta_1, self._server.state.beta_2],
            "mean" : [self._server.state.dist_mean_0, self._server.state.dist_mean_1, self._server.state.dist_mean_2],
            "cov" : self._server.state.dist_cov,
            "loc" : [self._server.state.loc_0, self._server.state.loc_1, self._server.state.loc_2],
            "lbd" : [self._server.state.lbd_0, self._server.state.lbd_1, self._server.state.lbd_2],
            "domain" : [ [self._server.state.dist_domain_min_0, self._server.state.dist_domain_min_1, self._server.state.dist_domain_min_2 ], [self._server.state.dist_domain_max_0, self._server.state.dist_domain_max_1, self._server.state.dist_domain_max_2]]
}
                
        p = make_distribution(self._server.state.dist, param_dict)
        
        order = self._server.state.order
        plabels = self._server.state.plabels

        pce = PolynomialChaosExpansion(distribution=p, order=order, plabels=plabels)
        pce.generate_samples()
        
        print('This queries the model {0:d} times'.format(pce.samples.shape[0]))
        
        

        # evaluate switcher
        x, model_output = make_model(self._server.state.model_name, Nparams, self._server.state.N, pce.samples)
        
        self._server.state.x = x.tolist()
        pce.build(model_output=model_output)
        self._server.state.pce = jsonpickle.encode(pce)
        
        self._server.controller.figure_update
        
#        self.update_plot()

    def change_model(self, **kwargs):
        self.build_model()
        logger.info(f">>> ENGINE(a): model changed to {self._server.state.model}")

           
    def change_distribution(self, **kwargs):
        self.reset_params()
        logger.info(f">>> ENGINE(a):distribution changed to {self._server.state.dist}")
        
    
#    def update_plot(self, **kwargs):
#        self.ctrl.figure_update(PLOTS[self._server.state.active_plot]())
#        logger.info(f">>> ENGINE(a): updating plot to {self._server.state.active_plot}")
        
    def widget_click(self):
        logger.info(">>> ENGINE(a): Widget Click")

    def widget_change(self):
        logger.info(">>> ENGINE(a): Widget Change")

#    def ui(self, *args, **kwargs):
#        # moving everthing to ui.py
#        
            
def make_distribution(dist_type, param_dict, **kwargs):
    func = DISTRIBUTIONS.get(dist_type, lambda: "Invalid distribution")
    print(dist_type)
    print(param_dict)
    print(func)
    return func(param_dict, **kwargs)
    
def make_BetaDistribution(param_dict):
    alpha = param_dict["alpha"]
    beta = param_dict["beta"]
    domain= np.array(param_dict["domain"])
    
    p =[]
    for n in range(3):
        p.append(BetaDistribution(alpha = alpha[n], beta = beta[n], domain = domain[:,n]))
    
    return p
    
def make_UniformDistribution(param_dict):
    return UniformDistribution(domain= np.array(param_dict["domain"]))
    
def make_NormalDistribution(param_dict):
    return NormalDistribution(mean=np.array(param_dict["mean"]), cov= np.array(param_dict["cov"]) )
    
def make_ExponentialDistribution(param_dict):
    print(param_dict["lbd"])
    print(param_dict["loc"])
    return ExponentialDistribution(lbd=np.array(param_dict["lbd"]), loc=np.array(param_dict["loc"]))
    
def make_ConstantParam(param_dict):
    return param_dict["constant"]
    
def make_model(model_name, Nparams, N, samples, **kwargs):
    print(model_name)
    func = MODELS.get(model_name, lambda: "Invalid model")
    return func(Nparams, N, samples, **kwargs)
    
def make_lapace_ode_1d(Nparams, N, samples):

    x, model = laplace_ode_1d(Nparams, N=N)
    
    model_output = np.zeros([samples.shape[0], N])
    for ind in range(samples.shape[0]):
        model_output[ind, :] = model(samples[ind, :])
        
    return x, model_output
    
def make_sine_modulation(Nparams, N, samples):

    left = -1.
    right = 1.
    x = np.linspace(left, right, N)
    model = sine_modulation(N=N)
    
    model_output = np.zeros([samples.shape[0], N])
    for ind in range(samples.shape[0]):
        model_output[ind, :] = model(samples[ind, :])
        
    return x, model_output

MODELS = {
    "1D Lapace Ode" : make_lapace_ode_1d,
    "Sine Modulation" : make_sine_modulation
}

DISTRIBUTIONS = {
    "Beta" : make_BetaDistribution,
    "Normal" : make_NormalDistribution,
    "Exponential" : make_ExponentialDistribution,
    "Uniform" : make_UniformDistribution
#    "Constant" : make_ConstantParam
}

