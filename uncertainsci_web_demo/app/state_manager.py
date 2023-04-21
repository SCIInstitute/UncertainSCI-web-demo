from .state_defaults import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def initialize(server):

    state, ctrl = server.state, server.controller
    
    state.x = np.empty(1).tolist()
    state.pce = None
    
    state.active_tab = default_model_state["dist"]

    def reset_all():
        logger.info(f">>> reseting all")
        
        state.x = np.empty(1).tolist()
        state.pce = None
        
        state.model_name = default_model
        
        state.update(default_model_state)
        for d in range(3):
            state.update(default_params_state[d])
        ctrl.build_model()
        
    def reset_model():
        logger.info(f">>> reseting model params")
        state.update(default_model_state)
        ctrl.build_model()
        

    def reset_params():
        logger.info(f">>> reseting distribution params")
        for d in range(3):
            state.update(default_params_state[d])
            
        ctrl.build_model()
        
    ctrl.reset_params = reset_params
    ctrl.reset_model = reset_model
    ctrl.reset_all  = reset_all
    
    logger.info(f">>> state manager initialized")
        
    

