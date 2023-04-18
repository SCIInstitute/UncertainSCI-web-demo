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
from UncertainSCI.distributions import BetaDistribution
from UncertainSCI.model_examples import laplace_ode_1d
from UncertainSCI.pce import PolynomialChaosExpansion

from UncertainSCI.vis import piechart_sensitivity, quantile_plot, mean_stdev_plot


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------


# -------------------------------
# state variables for the simulation
#-----------------------------------

def on_event(type, e):
    print(type, e)


class PCE_Model:
    def __init__(self):
        self.pce = None
        self.x = np.empty(1)
        
    def set_pce(self, pce):
        self.pce = pce
    
    def set_x(self, x):
        self.x = x
        
#pce = PCE_Model()

class Engine:
    def __init__(self, server=None):
        if server is None:
            server = get_server()
            
        Nparams = 3

        p1 = BetaDistribution(alpha=0.5, beta=1.)
        p2 = BetaDistribution(alpha=1., beta=0.5)
        p3 = BetaDistribution(alpha=1., beta=1.)

        plabels=['a', 'b', 'z']

        # # Polynomial order
        order = 5

        N = 100
        x, model = laplace_ode_1d(Nparams, N=N)

        pce = PolynomialChaosExpansion(distribution=[p1, p2, p3], order=order, plabels=plabels)
        pce.generate_samples()

        print('This queries the model {0:d} times'.format(pce.samples.shape[0]))

        model_output = np.zeros([pce.samples.shape[0], N])
        for ind in range(pce.samples.shape[0]):
            model_output[ind, :] = model(pce.samples[ind, :])
        pce.build(model_output=model_output)
            

        self._server = server

        # initialize state + controller
        state, ctrl = server.state, server.controller

        # Set state variable
        state.trame__title = "UncertainSCI-web-demo"
        state.distribution = "beta"
        state.model = "lapace-ode-1d"
        state.alpha = [ 0.5, 1, 1]
        state.beta = [ 1, 0.5, 1]
        state.order = order
        state.model_output = model_output
        state.Nparams = Nparams
        state.x = x
        state.plabels = plabels
        state.samples = pce.samples
        state.pce = pce

        # Bind instance methods to controller
        ctrl.reset_distribution = self.reset_distribution
        ctrl.on_server_reload = self.ui
                
        ctrl.widget_click = self.widget_click
        ctrl.widget_change = self.widget_change
        

        # Bind instance methods to state change
        state.change("active_plot")(self.update_plot)

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


    def reset_distribution(self):
        
        self._server.state.distribution = "beta"
        self._server.state.model = "lapace-ode-1d"
        self._server.state.alpha = [ 0.5, 1, 1]
        self._server.state.beta = [ 1, 0.5, 1]
    
    def update_plot(active_plot, **kwargs):
        ctrl.figure_update(PLOTS[active_plot]())
        logger.info(f">>> ENGINE(a): updating plot to {active_plot}")
        
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
                    v_model=("active_plot", "MeanStd"),
                    items=("plots", list(PLOTS.keys())),
                    hide_details=True,
                    dense=True,
                )

            # Main content
            with layout.content:
                with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
                    with vuetify.VRow(dense=True):
                        vuetify.VSpacer()
                        html_plot = plotly.Figure(
                            display_logo=False,
                            display_mode_bar=("true",),
                            selected=(on_event, "['selected', VuePlotly.safe($event)]"),
                            # hover=(on_event, "['hover', VuePlotly.safe($event)]"),
                            # selecting=(on_event, "['selecting', $event]"),
                            # unhover=(on_event, "['unhover', $event]"),
                        )
                        self.ctrl.figure_update = html_plot.update
                        vuetify.VSpacer()

            # Footer
            # layout.footer.hide()

    
def MeanStd():

    mean = pce_ctrl.pce.mean()
    stdev = pce_ctrl.pce.stdev()

    x_rev = pce_ctrl.x[::-1]
    upper = mean+stdev
    lower = mean-stdev
    lower=lower[::- 1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.hstack((pce_ctrl.x,x_rev)),
        y=np.hstack((upper,lower)),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=True,
        name='Stdev',
    ))
    fig.add_trace(go.Scatter(
        x=pce_ctrl.x, y=mean,
        line_color='rgb(0,100,80)',
        name='mean',
    ))

    fig.update_traces(mode='lines')
    
    return fig

def protocols_ready():
    print(">>> ENGINE: Server protocols initialized / Client not connected yet")
    update_plot("MeanStd")
    return True

def reset_resolution():
    state.resolution = 6
    
def widget_click():
    print(">>> ENGINE: Widget Click")
    update_plot()

def widget_change():
    print(">>> ENGINE: Widget Change")
    update_plot()

# -----------------------------------------------------------------------------


def Quantiles():
    
    bands = 3
    band_mass = 1/(2*(bands+1))
    x_rev = pce_ctrl.x[::-1]
    
    dq = 0.5/(bands+1)
    q_lower = np.arange(dq, 0.5-1e-7, dq)[::-1]
    q_upper = np.arange(0.5 + dq, 1.0-1e-7, dq)
    quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

    quantiles = pce_ctrl.pce.quantile(quantile_levels, M=int(2e3))
    median = quantiles[-1, :]
    
    fig = go.Figure()
        
    for ind in range(bands):
        alpha = (bands-ind) * 1/bands - (1/(2*bands))
        upper = quantiles[ind, :]
        lower = quantiles[bands+ind, ::-1]
        if ind == 0:
            fig.add_trace(go.Scatter(
                x=np.hstack((pce_ctrl.x,x_rev)),
                y=np.hstack((upper,lower)),
                fill='toself',
                fillcolor='rgba(100,0,0,'+str(alpha)+')',
                line_color='rgba(100,0,0,0)',
                showlegend=True,
                name='{0:1.2f} probability mass (each band)'.format(band_mass),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=np.hstack((pce_ctrl.x,x_rev)),
                y=np.hstack((upper,lower)),
                fill='toself',
                fillcolor='rgba(100,0,0,'+str(alpha)+')',
                line_color='rgba(100,0,0,0)',
                showlegend=False,
            ))

    
    fig.add_trace(go.Scatter(
        x=pce_ctrl.x, y=median,
        line_color='rgb(0,0,0)',
        name='median',
    ))

    fig.update_traces(mode='lines')


    return fig


# -----------------------------------------------------------------------------


def SensitivityPiechart():

    global_sensitivity, variable_interactions = pce_ctrl.pce.global_sensitivity()
    scalarized_GSI = np.mean(global_sensitivity, axis=1)
    print(type(scalarized_GSI))
    labels = [' '.join([pce_ctrl.pce.plabels[v] for v in varlist]) for varlist in variable_interactions]
    print(type(labels[0]))
    print(variable_interactions)
    print(labels)
    print(scalarized_GSI)

    fig = go.Figure(
        data=[go.Pie(labels=labels, values=scalarized_GSI.tolist())]
        )

    #    labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
    #    values = [4500, 2500, 1053, 500]
    #
    #    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])


    return fig



PLOTS = {
    "MeanStd": MeanStd,
    "Quantiles": Quantiles,
    "Sensitivities": SensitivityPiechart,
}
