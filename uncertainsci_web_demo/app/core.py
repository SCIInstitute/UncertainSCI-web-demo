r"""
Define your classes and create the instances that you need to expose
"""
import logging
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, plotly

import plotly.graph_objects as go
import plotly.express as px


import numpy as np
from UncertainSCI.distributions import BetaDistribution, UniformDistribution, ExponentialDistribution, NormalDistribution
from UncertainSCI.model_examples import laplace_ode_1d, sine_modulation
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot

#import jsonpickle


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------


# -------------------------------
# state variables for the simulation
#-----------------------------------

Nparams = 3
def on_event(type, e):
    print(type, e)
    
default_model = "1D Lapace Ode"

default_params_state= [[]]*3

default_params_state[0] = {
    "alpha_0" : 0.5,
    "beta_0" : 1.0,
    "dist_mean_0" : 0.0,
    "loc_0" : 0.0,
    "lbd_0" : 3.0,
    "dist_domain_min_0" : 0.0,
    "dist_domain_max_0" : 1.0,
    "constant_0" : 0.5
}

default_params_state[1] = {
    "alpha_1" : 1.0,
    "beta_1" : 0.5,
    "dist_mean_1" : 1.0,
    "loc_1" : 0.0,
    "lbd_1" : 3.0,
    "dist_domain_min_1" : 0.0,
    "dist_domain_max_1" : 1.0,
    "constant_1" : 0.5
}

default_params_state[2] = {
    "alpha_2" : 1.0,
    "beta_2" : 1.0,
    "dist_mean_2" : 1.0,
    "loc_2" : 0.0,
    "lbd_2" : 3.0,
    "dist_domain_min_2" : 0.0,
    "dist_domain_max_2" : 1.0,
    "constant_2" : 0.5
}

cov = np.random.randn(Nparams, Nparams)
cov = np.matmul(cov.T, cov)/(4*Nparams)  # Some random covariance matrix
# Normalize so that parameters with larger index have smaller importance
D = np.diag(1/np.sqrt(np.diag(cov)) * 1/(np.arange(1, Nparams+1)**2))
cov = D @ cov @ D

default_model_state = {
    "order" : 5,
    "N" : 100,
    "plabels" : ['a', 'b', 'z'],
    "dist" : "Beta",
    "dist_cov" : cov.tolist()
}

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
    
        
    

class Engine:
    def __init__(self, server=None):
        if server is None:
            server = get_server()
            
        Nparams = 3

        self._server = server
        

        # initialize state + controller
        state, ctrl = server.state, server.controller

        # Set state variable
        state.trame__title = "UncertainSCI web demo"
        
        state.active_plot = "Mean and Std"
        
        self.reset_all()

        # Bind instance methods to controller
#        ctrl.reset_params = self.reset_params
#        ctrl.reset_model = self.reset_model
#        ctrl.reset_all = self.reset_all
        ctrl.on_server_reload = self.ui
                
        ctrl.widget_click = self.widget_click
        ctrl.widget_change = self.widget_change
        

        # Bind instance methods to state change
        state.change("active_plot")(self.update_plot)
        state.change("model_name")(self.change_model)
        state.change("dist")(self.change_distribution)

        # Generate UI
        self.ui()
        

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
        
        pce_model.clear()
        
        Nparams = 3
    
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
        
        
        
        
        param0_dict = {
            "alpha" : self._server.state.alpha_0,
            "beta" : self._server.state.beta_0,
            "mean" : self._server.state.dist_mean_0,
            "cov" : self._server.state.dist_cov_0,
            "loc" : self._server.state.loc_0,
            "lbd" : self._server.state.lbd_0,
            "constant" : self._server.state.constant_0,
            "domain" : np.array([[ self._server.state.dist_domain_min_0], [self._server.state.dist_domain_max_0 ]])
        }
        
        param1_dict = {
            "alpha" : self._server.state.alpha_1,
            "beta" : self._server.state.beta_1,
            "mean" : self._server.state.dist_mean_1,
            "cov" : self._server.state.dist_cov_1,
            "loc" : self._server.state.loc_1,
            "lbd" : self._server.state.lbd_1,
            "constant" : self._server.state.constant_1,
            "domain" : np.array([[ self._server.state.dist_domain_min_1], [self._server.state.dist_domain_max_1 ]])
        }
        
        param2_dict = {
            "alpha" : self._server.state.alpha_2,
            "beta" : self._server.state.beta_2,
            "mean" : self._server.state.dist_mean_2,
            "cov" : self._server.state.dist_cov_2,
            "loc" : self._server.state.loc_2,
            "lbd" : self._server.state.lbd_2,
            "constant" : self._server.state.constant_2,
            "domain" : np.array([[ self._server.state.dist_domain_min_2], [self._server.state.dist_domain_max_2 ]])
        }
        
        
    
        
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
        
        pce_model.set_x(x)
        pce.build(model_output=model_output)
        pce_model.set_pce(pce)
        
#        self.update_plot()
        
        
        
    def reset_all(self):
        self._server.state.model_name = default_model
        self.reset_model()
        
    def reset_model(self):
        self._server.state.update(default_model_state)
        self.reset_params()
        

    def reset_params(self):
        for d in range(3):
            self._server.state.update(default_params_state[d])
        self.build_model()
        
    def change_model(self, **kwargs):
        self.build_model()
        logger.info(f">>> ENGINE(a): model changed to {self._server.state.model}")
        
    def change_distribution(self, **kwargs):
        self.reset_params()
        logger.info(f">>> ENGINE(a):distribution changed to {self._server.state.dist}")
        
    
    def update_plot(self, **kwargs):
        self.ctrl.figure_update(PLOTS[self._server.state.active_plot]())
        logger.info(f">>> ENGINE(a): updating plot to {self._server.state.active_plot}")
        
    def widget_click(self):
        logger.info(">>> ENGINE(a): Widget Click")

    def widget_change(self):
        logger.info(">>> ENGINE(a): Widget Change")

    def ui(self, *args, **kwargs):
        with SinglePageLayout(self._server) as layout:
            # Toolbar
            layout.title.set_text("UncertainSCI demo")
            with layout.toolbar:
                vuetify.VSpacer()
                vuetify.VSelect(
                    v_model=("model_name", default_model),
                    items=("models", list(MODELS.keys())),
                    hide_details=True,
                    dense=True,
                )
                vuetify.VSpacer()
                vuetify.VSelect(
                    v_model=("dist", default_model_state["dist"]),
                    items=("distributions", list(DISTRIBUTIONS.keys())),
                    hide_details=True,
                    dense=True,
                )
                vuetify.VSpacer()
                vuetify.VSlider(                    # Add slider
                    v_model=("N", 100),      # bind variable with an initial value of 100
                    min=10, max=1000,                  # slider range
                    dense=True, hide_details=True,  # presentation setup
                )
                vuetify.VSpacer()
                vuetify.VSlider(                    # Add slider
                    v_model=("order", 5),      # bind variable with an initial value of 5
                    min=1, max=7,                  # slider range
                    dense=True, hide_details=True,  # presentation setup
                )
                vuetify.VSpacer()
                vuetify.VSelect(
                    v_model=("active_plot", "Mean and Std"),
                    items=("plots", list(PLOTS.keys())),
                    hide_details=True,
                    dense=True,
                )

            # Main content
            with layout.content:
                with vuetify.VContainer(fluid=True):
                    with vuetify.VRow(dense=True):
                        vuetify.VSpacer()

                        figure = plotly.Figure(
                            display_logo=False,
                            display_mode_bar=("true",),
                            selected=(on_event, "['selected', VuePlotly.safe($event)]"),
                            # hover=(on_event, "['hover', VuePlotly.safe($event)]"),
                            # selecting=(on_event, "['selecting', $event]"),
                            # unhover=(on_event, "['unhover', $event]"),
                        )
                        self.ctrl.figure_update = figure.update
                        vuetify.VSpacer()

            # Footer
            # layout.footer.hide()
            
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

    
def MeanStd():

    mean = pce_model.pce.mean()
    stdev = pce_model.pce.stdev()

    x_rev = pce_model.x[::-1]
    upper = mean+stdev
    lower = mean-stdev
    lower=lower[::- 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.hstack((pce_model.x,x_rev)),
        y=np.hstack((upper,lower)),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Stdev',
    ))
    fig.add_trace(go.Scatter(
        x=pce_model.x, y=mean,
        line_color='rgb(0,100,80)',
        name='mean',
    ))

    fig.update_traces(mode='lines')

    return fig

# -----------------------------------------------------------------------------


def Quantiles():

    bands = 3
    band_mass = 1/(2*(bands+1))
    x_rev = pce_model.x[::-1]

    dq = 0.5/(bands+1)
    q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
    q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)
    quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

    quantiles = pce_model.pce.quantile(quantile_levels, M=int(2e3))
    median = quantiles[-1, :]

    fig = go.Figure()

    for ind in range(bands):
        alpha = (bands-ind) * 1/bands - (1/(2*bands))
        upper = quantiles[ind, :]
        lower = quantiles[bands+ind, ::-1]
        if ind == 0:
            fig.add_trace(go.Scatter(
                x=np.hstack((pce_model.x,x_rev)),
                y=np.hstack((upper,lower)),
                fill='toself',
                fillcolor='rgba(100,0,0,'+str(alpha)+')',
                line_color='rgba(100,0,0,0)',
                showlegend=True,
                name='{0:1.2f} probability mass (each band)'.format(band_mass),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=np.hstack((pce_model.x,x_rev)),
                y=np.hstack((upper,lower)),
                fill='toself',
                fillcolor='rgba(100,0,0,'+str(alpha)+')',
                line_color='rgba(100,0,0,0)',
                showlegend=False,
            ))


    fig.add_trace(go.Scatter(
        x=pce_model.x, y=median,
        line_color='rgb(0,0,0)',
        name='median',
    ))

    fig.update_traces(mode='lines')


    return fig


# -----------------------------------------------------------------------------


def SensitivityPiechart():

    global_sensitivity, variable_interactions = pce_model.pce.global_sensitivity()
    scalarized_GSI = np.mean(global_sensitivity, axis=1)
    print(type(scalarized_GSI))
    labels = [' '.join([pce_model.pce.plabels[v] for v in varlist]) for varlist in variable_interactions]
    
#    print(type(labels[0]))
#    print(variable_interactions)
#    print(labels)
#    print(scalarized_GSI)

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=scalarized_GSI.tolist())]
        )

    #    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    #    values = [4500, 2500, 1053, 500]
    #
    #    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])


    return fig
    



PLOTS = {
    "Mean and Std": MeanStd,
    "Quantiles Plot": Quantiles,
    "Sensitivities": SensitivityPiechart,
}


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

