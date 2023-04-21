import numpy as np

# -------------------------------
# state variables for the simulation
#-----------------------------------


Nparams = 3

default_model = "1D Lapace Ode"
default_plot = "Mean and Std"

default_params_state= [[]]*Nparams

default_params_state[0] = {
    "alpha_0" : 0.5,
    "beta_0" : 1.0,
    "dist_mean_0" : 0.0,
    "loc_0" : 0.0,
    "lbd_0" : 3.0,
    "dist_domain_0" : [ 0.0,  1.0],
    "constant_0" : 0.5
}

default_params_state[1] = {
    "alpha_1" : 1.0,
    "beta_1" : 0.5,
    "dist_mean_1" : 1.0,
    "loc_1" : 0.0,
    "lbd_1" : 3.0,
    "dist_domain_1" : [ 0.0,  1.0],
    "constant_1" : 0.5
}

default_params_state[2] = {
    "alpha_2" : 1.0,
    "beta_2" : 1.0,
    "dist_mean_2" : 1.0,
    "loc_2" : 0.0,
    "lbd_2" : 3.0,
    "dist_domain_2" : [ 0.0,  1.0],
    "constant_2" : 0.5
}


cov = np.random.randn(Nparams, Nparams)
cov = np.matmul(cov.T, cov)/(4*Nparams)  # Some random covariance matrix
# Normalize so that parameters with larger index have smaller importance
D = np.diag(1/np.sqrt(np.diag(cov)) * 1/(np.arange(1, Nparams+1)**2))
cov = D @ cov @ D


default_model_state = {
    "Nparams" : Nparams,
    "order" : 4,
    "N" : 100,
    "plabels" : ['a', 'b', 'z'],
    "dist" : "Beta",
    "dist_cov" : cov.tolist()
}
