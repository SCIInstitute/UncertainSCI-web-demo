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


#class PCE_Model:
#    def __init__(self):
#        self.pce = None
#        self.x = np.empty(1)
#
#    def clear(self):
#        self.__init__()
#
#    def set_pce(self, pce):
#        self.pce = pce
#
#    def set_x(self, x):
#        self.x = x
#
#    def set(self, attr, value):
#        eval(attr+" = "+value)
#
#pce_model = PCE_Model()
#
        
    

class Engine: # same as controller?
    def __init__(self, server=None):
        if server is None:
            server = get_server()
            
        Nparams = 3

        self._server = server
        

        # initialize state + controller
        state, ctrl = server.state, server.controller
        
        state_manager.initialize(server)
        pce_builder.initialize(server)
        
        ctrl.reset_all()
        

        # Bind instance methods to controller
#        ctrl.reset_params = self.reset_params
#        ctrl.reset_model = self.reset_model
#        ctrl.reset_all = self.reset_all
        ctrl.on_server_reload = ui.initialize
                
        ctrl.widget_click = self.widget_click
        ctrl.widget_change = self.widget_change
#        ctrl.build_model = self.build_model
#        ctrl.update_plot = self.update_plot
#        ctrl.change_model = self.change_model
#        ctrl.change_distribution = self.change_distribution

        # Bind instance methods to state change
#        state.change("active_plot")(self.update_plot)
#        state.change("model_name")(self.change_model)
#        state.change("dist")(self.change_distribution)

        

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
        
#        self.update_plot()
        
    
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
            

    
    
    

