from .state_defaults import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def initialize(server):

    state, ctrl = server.state, server.controller
    
    state.x = np.empty(1).tolist()
    state.pce = None

    def reset_all():
        state.model_name = default_model
        reset_model()
        
    def reset_model():
        state.update(default_model_state)
        reset_params()
        

    def reset_params():
        for d in range(3):
            state.update(default_params_state[d])
            
        ctrl.build_model()
        
    ctrl.reset_params = reset_params
    ctrl.reset_model = reset_model
    ctrl.reset_all  = reset_all
        
    

