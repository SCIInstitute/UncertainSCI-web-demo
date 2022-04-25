from trame import state, controller as ctrl
from trame.layouts import SinglePage
from trame.html import vuetify, plotly
from uncertainsci_web_demo import html as my_widgets

import plotly.graph_objects as go
import plotly.express as px
from .engine import MeanStd, Quantiles, SensitivityPiechart, html_plot, PLOTS
from .controller import on_event

# Create single page layout type
# (FullScreenPage, SinglePage, SinglePageWithDrawer)
layout = SinglePage(
    "UncertainSCI-web-demo",
    on_ready=ctrl.on_ready,
)

with layout.toolbar:
    vuetify.VSpacer()
    vuetify.VSelect(
        v_model=("active_plot", "MeanStd"),
        items=("plots", list(PLOTS.keys())),
        hide_details=True,
        dense=True,
    )
    
with layout.content:
    with vuetify.VContainer(fluid=True):
        with vuetify.VRow(dense=True):
            vuetify.VSpacer()
            html_plot = plotly.Plotly(
                "demo",
                display_mode_bar=("true",),
                selected=(on_event, "['selected', VuePlotly.safe($event)]"),
                # hover=(on_event, "['hover', VuePlotly.safe($event)]"),
                # selecting=(on_event, "['selecting', $event]"),
                # unhover=(on_event, "['unhover', $event]"),
            )
            vuetify.VSpacer()


# -------------------------------
# state variables for the simulation
#-----------------------------------

state.update(
    {
        "distribution" : "beta",
        "model" : "lapace-ode-1d",
        "alpha" : [ 0.5, 1, 1],
        "beta" : [ 1, 0.5, 1]
     }


)
