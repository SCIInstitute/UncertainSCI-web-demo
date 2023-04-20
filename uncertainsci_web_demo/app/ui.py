import logging
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, plotly

import plotly.graph_objects as go
import plotly.express as px

import jsonpickle


import numpy as np
from UncertainSCI.pce import PolynomialChaosExpansion
from .state_defaults import *
from .pce_builder import MODELS, DISTRIBUTIONS

# Create single page layout type
# (FullScreenPage, SinglePage, SinglePageWithDrawer)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize(server):


    state, ctrl = server.state, server.controller
    state, ctrl = server.state, server.controller

    state.trame__title = "UncertainSCI web demo"
    
    # Set state variable
    state.active_plot = default_plot
    
    @state.change("order", "N")
    def change_model_params(**kwargs):
        ctrl.build_model()
    
    @state.change("active_plot")
    def update_plot(**kwargs):
        ctrl.figure_update(PLOTS[state.active_plot]())
        logger.info(f">>> ENGINE(a): updating plot to {state.active_plot}")
        
#    def make_plot(active_plot, **kwargs):
#        func = PLOTS.get(dist_type, lambda: "Invalid plot_type")
#    return func(state, **kwargs)
    
    def MeanStd():

        pce = jsonpickle.decode(state.pce)
        x = np.array(state.x)

        mean = pce.mean()
        stdev = pce.stdev()

        x_rev = x[::-1]
        upper = mean+stdev
        lower = mean-stdev
        lower=lower[::- 1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.hstack((x,x_rev)),
            y=np.hstack((upper,lower)),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            showlegend=True,
            name='Stdev',
        ))
        fig.add_trace(go.Scatter(
            x=x, y=mean,
            line_color='rgb(0,100,80)',
            name='mean',
        ))

        fig.update_traces(mode='lines')

        return fig

    # -----------------------------------------------------------------------------


    def Quantiles(state):
        
        pce = jsonpickle.decode(state.pce)
        x = np.array(state.x)

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


    def SensitivityPiechart(state):

        pce = jsonpickle.decode(state.pce)
        x = np.array(state.x)

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



    with SinglePageLayout(server) as layout:
        # Toolbar
        with layout.icon:
            vuetify.VIcon("uncertainsci")
        layout.title.set_text("UncertainSCI demo")
        with layout.toolbar:
    #        with vuetify.VRow(dense=True):
            vuetify.VSpacer()
            vuetify.VSelect(
                v_model=("model_name", default_model),
                items=("models", list(MODELS.keys())),
                hide_details=True,
                dense=True,
            )
            vuetify.VSelect(
                v_model=("dist", default_model_state["dist"]),
                items=("distributions", list(DISTRIBUTIONS.keys())),
                hide_details=True,
                dense=True,
            )
            vuetify.VSelect(
                v_model=("active_plot", "Mean and Std"),
                items=("plots", list(PLOTS.keys())),
                hide_details=True,
                dense=True,
            )
            vuetify.VProgressLinear(
                indeterminate=True,
                absolute=True,
                bottom=True,
                active=("trame__busy",),
            )

        # Main content
        with layout.content:
            with vuetify.VContainer(fluid=True):
                with vuetify.VRow(dense=True, style = "min-height: 200;"):
                    create_section_parameters()
                    vuetify.VSpacer()
                    figure = plotly.Figure(
                        display_logo=False,
                        display_mode_bar=("true",),
                        selected=(on_event, "['selected', VuePlotly.safe($event)]"),
                        # hover=(on_event, "['hover', VuePlotly.safe($event)]"),
                        # selecting=(on_event, "['selecting', $event]"),
                        # unhover=(on_event, "['unhover', $event]"),
                    )
                    ctrl.figure_update = figure.update
                    vuetify.VSpacer()


                    

def create_section_parameters():
    with vuetify.VCard(wstyle="width: 440px;") as _card:
        _header = vuetify.VCardTitle()
        vuetify.VDivider()
        _content = vuetify.VCardText()
        
        _header.add_child("Model parameters")
        
        with _content:
#            with vuetify.VRow(dense=True):
            vuetify.VSpacer()
            vuetify.VSlider(                    # Add slider
                label = "number of domain points",
                v_model=("N", 100),      # bind variable with an initial value of 100
                type="int",
                min=10, max=1000,                  # slider range
                change="flushState('N')",
                dense=True, hide_details=True,  # presentation setup
            )
            vuetify.VSpacer()
            vuetify.VSlider(                    # Add slider
                label = "Polynomial order",
                type="int",
                v_model=("order", 5),      # bind variable with an initial value of 5
                min=1, max=7,                  # slider range
                change="flushState('order')",
                dense=True, hide_details=True,  # presentation setup
            )

def on_event(type, e):
    print(type, e)



# ------ old ---
    # Footer
    # layout.footer.hide()

#layout = SinglePage(
#    "UncertainSCI-web-demo",
#    on_ready=ctrl.on_ready,
#)
#
#with layout.toolbar:
#    vuetify.VSpacer()
#    vuetify.VSelect(
#        v_model=("active_plot", "MeanStd"),
#        items=("plots", list(PLOTS.keys())),
#        hide_details=True,
#        dense=True,
#    )
#
#with layout.content:
#    with vuetify.VContainer(fluid=True):
#        with vuetify.VRow(dense=True):
#            vuetify.VSpacer()
#            html_plot = plotly.Plotly(
#                "lapace-ode-1d",
#                display_mode_bar=("true",),
#                selected=(on_event, "['selected', VuePlotly.safe($event)]"),
#                # hover=(on_event, "['hover', VuePlotly.safe($event)]"),
#                # selecting=(on_event, "['selecting', $event]"),
#                # unhover=(on_event, "['unhover', $event]"),
#            )
#            vuetify.VSpacer()
#
#
## -------------------------------
## state variables for the simulation
##-----------------------------------
#
#state.update(
#    {
#        "distribution" : "beta",
#        "model" : "lapace-ode-1d",
#        "alpha" : [ 0.5, 1, 1],
#        "beta" : [ 1, 0.5, 1]
#     }
#
#
#)
