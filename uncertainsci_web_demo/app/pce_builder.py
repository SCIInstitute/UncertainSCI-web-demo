import logging
import numpy as np

from UncertainSCI.distributions import BetaDistribution, UniformDistribution, ExponentialDistribution, NormalDistribution
from UncertainSCI.model_examples import laplace_ode_1d, sine_modulation
from UncertainSCI.pce import PolynomialChaosExpansion

import jsonpickle



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def initialize(server):

    state, ctrl = server.state, server.controller
    
    def build_model():
        
        state.x = np.empty(1).tolist()
        state.pce = None
    
        
#        print(state.dist_domain_0)
#        print(state.dist_domain_1)
#        print(state.dist_domain_2)
        
        print(state.model_name)
        
        
        param_dict = {
            "alpha" : [state.alpha_0, state.alpha_1, state.alpha_2],
            "beta" : [state.beta_0, state.beta_1, state.beta_2],
            "mean" : [state.dist_mean_0, state.dist_mean_1, state.dist_mean_2],
            "cov" : state.dist_cov,
            "loc" : [state.loc_0, state.loc_1, state.loc_2],
            "lbd" : [state.lbd_0, state.lbd_1, state.lbd_2],
            "domain" : [ [state.dist_domain_0[0], state.dist_domain_1[0], state.dist_domain_2[0]],  [state.dist_domain_0[1], state.dist_domain_1[1], state.dist_domain_2[1]]]
        }
                
        p = make_distribution(state.dist, param_dict)
        
        order = state.order
        plabels = state.plabels

        pce = PolynomialChaosExpansion(distribution=p, order=order, plabels=plabels)
        pce.generate_samples()
        
        print('This queries the model {0:d} times'.format(pce.samples.shape[0]))
        
        

        # evaluate switcher
        x, model_output = make_model(state.model_name, state.Nparams, state.N, pce.samples)
        
        state.x = x.tolist()
        pce.build(model_output=model_output)
        state.pce = jsonpickle.encode(pce)
        
        ctrl.update_plot()
    
    
    def make_distribution(dist_type, param_dict, **kwargs):
        func = DISTRIBUTIONS.get(dist_type, lambda: "Invalid distribution")
#        print(dist_type)
#        print(param_dict)
#        print(func)
        return func(param_dict, **kwargs)
        
    def make_model(model_name, Nparams, N, samples, **kwargs):
        print(model_name)
        func = MODELS.get(model_name, lambda: "Invalid model")
        return func(Nparams, N, samples, **kwargs)
             
    @state.change("model_name")
    def change_model(**kwargs):
#        state.flush()
        print(state.model_name)
        build_model()
        logger.info(f">>> ENGINE(a): model changed to {state.model_name}")

    @state.change("dist")
    def change_distribution(**kwargs):
#        state.active_tab = state.dist
        build_model()
        logger.info(f">>> ENGINE(a):distribution changed to {state.dist}")
     
    ctrl.build_model = build_model
    ctrl.change_model = change_model
    ctrl.change_distribution = change_distribution
    
    logger.info(f">>> pce_builder initialized")
        


        
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
#    print(param_dict["lbd"])
#    print(param_dict["loc"])
    return ExponentialDistribution(lbd=np.array(param_dict["lbd"]), loc=np.array(param_dict["loc"]))
    
def make_ConstantParam(param_dict):
    return param_dict["constant"]
    
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
