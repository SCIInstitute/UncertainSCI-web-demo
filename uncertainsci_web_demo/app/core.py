r"""
Define your classes and create the instances that you need to expose
"""
import logging
from trame.app import get_server

from . import ui, state_manager, pce_builder

import numpy as np
import jsonpickle


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------
# Engine class
# ---------------------------------------------------------


class Engine:  # same as controller?
    def __init__(self, server=None):
        if server is None:
            server = get_server()

        self._server = server

        # initialize state + controller
        state, ctrl = server.state, server.controller

        #        ctrl.on_server_reload = ui.initialize

        ctrl.widget_click = self.widget_click
        ctrl.widget_change = self.widget_change

        logger.info(f">>> engine initialized")

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

    def widget_click(self):
        logger.info(">>> ENGINE(a): Widget Click")

    def widget_change(self):
        logger.info(">>> ENGINE(a): Widget Change")
