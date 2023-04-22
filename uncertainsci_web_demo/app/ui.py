import logging
from trame.app import get_server
from trame.assets.local import to_url
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vuetify, plotly

import plotly.graph_objects as go
import plotly.express as px

import jsonpickle

from pathlib import Path


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

    state.trame__title = "UncertainSCI web demo"

    # Set state variable
    state.active_plot = default_plot

    #    print out so I can copy to @state.change()
    #    dist_param_list = list(default_model_state.keys()) + ["model_name"]
    #    for k in range(default_model_state["Nparams"]):
    #         dist_param_list += list(default_params_state[k].keys())
    #    print(tuple(dist_param_list))

    @state.change("Nparams", "order", "N", "plabels")
    def change_model_params(**kwargs):
        print("changed model params")
        #        state.flush()
        ctrl.build_model()

    @state.change("dist_cov", "dist_mean_0", "dist_mean_1", "dist_mean_2")
    def change_normal_params(**kwargs):
        print("changed normal params")
        ctrl.build_model()

    @state.change(
        "alpha_0",
        "beta_0",
        "dist_domain_0",
        "alpha_1",
        "beta_1",
        "dist_domain_1",
        "alpha_2",
        "beta_2",
        "dist_domain_2",
    )
    def change_beta_params(**kwargs):
        print("changed beta params")
        ctrl.build_model()

    @state.change("loc_0", "lbd_0", "loc_1", "lbd_1", "loc_2", "lbd_2")
    def change_exponential_params(**kwargs):
        print("changed exponential params")
        ctrl.build_model()

    ctrl.change_model_params = change_model_params
    ctrl.change_normal_params = change_normal_params
    ctrl.change_beta_params = change_beta_params
    ctrl.change_exponential_params = change_exponential_params

    @state.change("active_plot")
    def update_plot(**kwargs):
        ctrl.figure_update(PLOTS[state.active_plot]())
        logger.info(f">>> ENGINE(a): updating plot to {state.active_plot}")

    ctrl.update_plot = update_plot

    def MeanStd():
        pce = jsonpickle.decode(state.pce)
        x = np.array(state.x)

        mean = pce.mean()
        stdev = pce.stdev()

        x_rev = x[::-1]
        upper = mean + stdev
        lower = mean - stdev
        lower = lower[::-1]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.hstack((x, x_rev)),
                y=np.hstack((upper, lower)),
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line_color="rgba(255,255,255,0)",
                showlegend=True,
                name="Stdev",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean,
                line_color="rgb(0,100,80)",
                name="mean",
            )
        )

        fig.update_traces(mode="lines")

        return fig

    # -----------------------------------------------------------------------------

    def Quantiles():
        pce = jsonpickle.decode(state.pce)
        x = np.array(state.x)

        bands = 3
        band_mass = 1 / (2 * (bands + 1))
        x_rev = x[::-1]

        dq = 0.5 / (bands + 1)
        q_lower = np.arange(dq, 0.5 - 1e-7, dq)[::-1]
        q_upper = np.arange(0.5 + dq, 1.0 - 1e-7, dq)
        quantile_levels = np.append(np.concatenate((q_lower, q_upper)), 0.5)

        quantiles = pce.quantile(quantile_levels, M=int(2e3))
        median = quantiles[-1, :]

        fig = go.Figure()

        for ind in range(bands):
            alpha = (bands - ind) * 1 / bands - (1 / (2 * bands))
            upper = quantiles[ind, :]
            lower = quantiles[bands + ind, ::-1]
            if ind == 0:
                fig.add_trace(
                    go.Scatter(
                        x=np.hstack((x, x_rev)),
                        y=np.hstack((upper, lower)),
                        fill="toself",
                        fillcolor="rgba(100,0,0," + str(alpha) + ")",
                        line_color="rgba(100,0,0,0)",
                        showlegend=True,
                        name="{0:1.2f} probability mass (each band)".format(band_mass),
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=np.hstack((x, x_rev)),
                        y=np.hstack((upper, lower)),
                        fill="toself",
                        fillcolor="rgba(100,0,0," + str(alpha) + ")",
                        line_color="rgba(100,0,0,0)",
                        showlegend=False,
                    )
                )

        fig.add_trace(
            go.Scatter(
                # x=x, y=median,
                line_color="rgb(0,0,0)",
                name="median",
            )
        )

        fig.update_traces(mode="lines")

        return fig

    # -----------------------------------------------------------------------------

    def SensitivityPiechart():
        pce = jsonpickle.decode(state.pce)
        x = np.array(state.x)

        global_sensitivity, variable_interactions = pce.global_sensitivity()
        scalarized_GSI = np.mean(global_sensitivity, axis=1)
        print(type(scalarized_GSI))
        labels = [
            " ".join([pce.plabels[v] for v in varlist])
            for varlist in variable_interactions
        ]

        # print(type(labels[0]))
        # print(variable_interactions)
        # print(labels)
        # print(scalarized_GSI)

        fig = go.Figure(data=[go.Pie(labels=labels, values=scalarized_GSI.tolist())])

        # labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
        # values = [4500, 2500, 1053, 500]
        #
        # fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

        return fig

    PLOTS = {
        "Mean and Std": MeanStd,
        "Quantiles Plot": Quantiles,
        "Sensitivities": SensitivityPiechart,
    }

    with SinglePageLayout(server) as layout:
        # Toolbar
        with layout.toolbar as tb:
            tb.clear()
            vuetify.VImg(
                src=to_url(str(Path(__file__).with_name("UncertainSCI.png"))),
                height=60,
                contain=True,
                style="max-width: 105px;",
            )
            vuetify.VToolbarTitle("UncertainSCI demo")
            vuetify.VSpacer()
            vuetify.VSelect(
                v_model=("model_name", default_model),
                items=("models", list(MODELS.keys())),
                hide_details=True,
                dense=True,
                style="max-width: 150px;",
                classes="mx-1",
            )
            vuetify.VSelect(
                v_model=("dist", default_model_state["dist"]),
                items=("distributions", list(DISTRIBUTIONS.keys())),
                hide_details=True,
                dense=True,
                style="max-width: 150px;",
                classes="mx-1",
            )
            vuetify.VSelect(
                v_model=("active_plot", "Mean and Std"),
                items=("plots", list(PLOTS.keys())),
                hide_details=True,
                dense=True,
                style="max-width: 150px;",
                classes="mx-1",
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
                create_section_model_parameters(ctrl)

                with vuetify.VRow(dense=True, no_gutters=True):
                    with vuetify.VCol(cols=4):
                        create_section_distribution_parameters(ctrl)
                    with vuetify.VCol(cols=8, classes="pa-2"):
                        figure = plotly.Figure(
                            display_logo=False,
                            display_mode_bar=("true",),
                            selected=(on_event, "['selected', VuePlotly.safe($event)]")
                            # hover=(on_event, "['hover', VuePlotly.safe($event)]"),
                            # selecting=(on_event, "['selecting', $event]"),
                            # unhover=(on_event, "['unhover', $event]"),
                        )
                    ctrl.figure_update = figure.update

    logger.info(f">>> ui initialized")


def create_section_model_parameters(ctrl):
    with vuetify.VCard(classes="mb-2"):
        _header = vuetify.VCardTitle(classes="py-1")
        vuetify.VDivider()
        _content = vuetify.VCardText()

        with _header:
            _header.add_child("Model parameters")
            vuetify.VSpacer()
            vuetify.VBtn(
                "Reset",
                click=ctrl.reset_model,
                small=True,
            )

        with _content:
            vuetify.VSlider(
                hint="Polynomial order",
                persistent_hint=True,
                type="number",
                min=1,
                max=7,
                v_model=("order", default_model_state["order"]),
                change="flushState('order')",
                dense=True,
                hide_details=False,  # presentation setup
                thumb_label="always",
                thumb_size="24",
                ticks="always",
                tick_size="4",
                tick_labels=[str(k + 1) for k in range(7)],
            )
            # Add slider
            vuetify.VSlider(
                hint="Number of domain points",
                persistent_hint=True,
                v_model=("N", default_model_state["N"]),
                type="number",
                min=10,
                max=1000,
                thumb_label="always",
                thumb_size="24",
                change="flushState('N')",
                dense=False,
                hide_details=False,  # presentation setup
            )


def create_section_distribution_parameters(ctrl):
    with vuetify.VCard():
        _header = vuetify.VCardTitle(classes="py-1")
        vuetify.VDivider()
        _content = vuetify.VCardText()

        with _header:
            _header.add_child("Distribution \n Parameters")
            vuetify.VSpacer()
            vuetify.VBtn(
                "Reset",
                click=ctrl.reset_params,
                small=True,
            )

        with _content:
            # vuetify.VTabs(
            #     v_model=("active_tab", default_model_state["dist"]),
            #     items=("Tabs", list(DISTRIBUTIONS.keys()))
            #     )
            # print("tabs")

            # with vuetify.VRow(dense=True):
            # vuetify.VSpacer()
            # with vuetify.VTab("Beta"):
            create_section_beta_parameters()
            # with vuetify.VTab("Normal"):
            #     create_section_beta_parameters()
            create_section_normal_parameters()
            create_section_exponential_parameters()
            create_section_uniform_parameters()


def make_param_slider(state_var, label, plabel, dim, bounds, step, dist):
    vuetify.VSlider(
        # label = state.plabels[0]+" alpha",
        # v_field_label = label,
        persistent_hint=True,
        hint=label,
        v_model=(state_var, default_params_state[dim][state_var]),
        style="track_active_size_offset : 0 px;",
        type="number",
        # disabled=("not dist == '"+dist+"'", ),
        min=bounds[0],
        max=bounds[1],
        step=step,
        change="flushState('" + state_var + "')",
        dense=True,
        hide_details=False,
        thumb_label="always",
        thumb_size="24",
    )


def make_boundary_slider(state_var, label, plabel, dim, bounds, step, dist):
    #    with vuetify.VCol(dense=True):
    #        with vuetify.VRow(dense=True):
    #            vuetify.VTextField(
    #                model_value = (state_var+"[0]", default_params_state[dim][state_var][0]),
    #                dense=True, hide_details=True, type="number",
    #                selected=(on_event, "['selected', VuePlotly.safe($event)]")
    #            )
    vuetify.VRangeSlider(
        persistent_hint=True,
        hint=label,
        v_model=(state_var, default_params_state[dim][state_var]),
        min=bounds[0],
        max=bounds[1],
        step=step,
        # disabled= ("not dist == '"+dist+"'", ),
        change="flushState('" + state_var + "')",
        dense=True,
        hide_details=False,
        type="number",
        thumb_label="always",
        thumb_size="24",
    )


# vuetify.VTextField(
#     model_value = (state_var+"[1]", default_params_state[dim][state_var][1]),
#     dense=True, hide_details=True, type="number",
#     selected=(on_event, "['selected', VuePlotly.safe($event)]")
# )


def create_section_beta_parameters():
    with vuetify.VCard(classes="mt-1"):
        _header = vuetify.VCardTitle(classes="py-1")
        vuetify.VDivider()
        _content = vuetify.VCardText()
        #
        _header.add_child("Beta \n Distribution")

        with _content:
            make_param_slider("alpha_0", "p0 alpha", "p0", 0, [0.0, 1.0], 0.01, "Beta")
            make_param_slider("beta_0", "p0 beta", "p0", 0, [0.0, 1.0], 0.01, "Beta")
            make_boundary_slider(
                "dist_domain_0", "p0 range", "p0", 0, [-5.0, 5.0], 0.1, "Beta"
            )

            make_param_slider("alpha_1", "p1 alpha", "p1", 1, [0.0, 1.0], 0.01, "Beta")
            make_param_slider("beta_1", "p1 beta", "p1", 1, [0.0, 1.0], 0.01, "Beta")
            make_boundary_slider(
                "dist_domain_1", "p1 range", "p1", 1, [-5.0, 5.0], 0.1, "Beta"
            )

            make_param_slider("alpha_2", "p2 alpha", "p2", 2, [0.0, 1.0], 0.01, "Beta")
            make_param_slider("beta_2", "p2 beta", "p2", 2, [0.0, 1.0], 0.01, "Beta")
            make_boundary_slider(
                "dist_domain_2", "p2 range", "p2", 2, [-5.0, 5.0], 0.1, "Beta"
            )


def create_section_normal_parameters():
    with vuetify.VCard(classes="mt-1"):
        _header = vuetify.VCardTitle(classes="py-1")
        vuetify.VDivider()
        _content = vuetify.VCardText()
        _header.add_child("Normal \n Distribution")

        with _content:
            make_param_slider(
                "dist_mean_0", "p0 mean", "p0", 0, [0.0, 1.0], 0.01, "Normal"
            )
            make_param_slider(
                "dist_mean_1", "p1 mean", "p1", 1, [0.0, 1.0], 0.01, "Normal"
            )
            make_param_slider(
                "dist_mean_2", "p2 mean", "p2", 2, [0.0, 1.0], 0.01, "Normal"
            )


def create_section_exponential_parameters():
    with vuetify.VCard(classes="mt-1"):
        _header = vuetify.VCardTitle(classes="py-1")
        vuetify.VDivider()
        _content = vuetify.VCardText()
        _header.add_child("Exponential \n Distribution")

        with _content:
            make_param_slider(
                "loc_0", "p0 mean", "p0", 0, [-5.0, 5.0], 0.01, "Exponential"
            )
            make_param_slider(
                "lbd_1", "p1 mean", "p0", 1, [-5.0, 5.0], 0.01, "Exponential"
            )


def create_section_uniform_parameters():
    with vuetify.VCard(classes="mt-1"):
        _header = vuetify.VCardTitle(classes="py-1")
        vuetify.VDivider()
        _content = vuetify.VCardText()
        _header.add_child("Uniform \n Distribution")

        with _content:
            make_boundary_slider(
                "dist_domain_0", "p0 range", "p0", 0, [-5.0, 5.0], 0.1, "Uniform"
            )
            make_boundary_slider(
                "dist_domain_1", "p1 range", "p1", 1, [-5.0, 5.0], 0.1, "Uniform"
            )
            make_boundary_slider(
                "dist_domain_2", "p2 range", "p2", 2, [-5.0, 5.0], 0.1, "Uniform"
            )


def on_event(type, e):
    print(type, e)
