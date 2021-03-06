import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import astropy.units as u
import numpy as np
from functools import partial
import scipy
import os
from alabi.cache_utils import load_model_cache
import random
#random.seed(0)

# ========================================================
# Configure vplanet forward model
# ========================================================

#inpath = os.path.join(vpi.INFILE_DIR, "/Users/rory/src/vplanet_inference")
inpath = "."

inparams  = {"star.dMass": u.Msun,          
            "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),   
            "star.dSatXUVTime": u.Gyr,    
            "vpl.dStopTime": u.Gyr,       
            "star.dXUVBeta": -u.dimensionless_unscaled}

outparams = {"final.star.Luminosity": u.Lsun,
            "final.star.LXUVStellar": u.Lsun}

vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams)

# ========================================================
# Observational constraints
# ========================================================

# Data: (mean, stdev)
prior_data = [(0.181, 0.019),     # mass [Msun]
            (-2.92, 0.26),    # log(fsat) 
            (None, None),     # tsat [Gyr]
            (8, 2),       # age [Gyr]
            (-1.18, 0.31)]    # beta

like_data = np.array([[4.38e-3, 3.4e-4],   # Lbol [Lsun]
                    [-4.5, 1]])    # Lxuv/Lbol

# Prior bounds
bounds = [(0.1, 0.3),        
        (-4.0, -1.0),
        (0.1, 5),
        (5, 12.0),
        (-2.0, 0.0)]

# ========================================================
# Configure prior 
# ========================================================

# Prior sampler - alabi format
ps = partial(ut.prior_sampler_normal, prior_data=prior_data, bounds=bounds)

# Prior - emcee format
lnprior = partial(ut.lnprior_normal, bounds=bounds, data=prior_data)

# Prior - dynesty format
prior_transform = partial(ut.prior_transform_normal, bounds=bounds, data=prior_data)

# ========================================================
# Configure likelihood
# ========================================================

# vpm.initialize_bayes(data=like_data, bounds=bounds, outparams=outparams)

def lnlike(theta):
    out = vpm.run_model(theta, remove=False)
    mdl = np.array([out[0], np.log10(out[1]/out[0])])
    lnl = -0.5 * np.sum(((mdl - like_data.T[0])/like_data.T[1])**2)
    return lnl

def lnpost(theta):
    return lnlike(theta) + lnprior(theta)

# ========================================================
# Run alabi
# ========================================================

kernel = "ExpSquaredKernel"

labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$",
        r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

if __name__ == "__main__":

    sm = SurrogateModel(fn=lnpost, bounds=bounds, prior_sampler=ps, 
                          savedir=f"results/{kernel}", labels=labels)
    sm.init_samples(ntrain=400, ntest=100, reload=False)
    sm.init_gp(kernel=kernel, fit_amp=False, fit_mean=True, white_noise=-15)
    sm.active_train(niter=100, algorithm="bape", gp_opt_freq=10, save_progress=True)
    sm.plot(plots=["gp_all"])

    #sm = load_model_cache(f"results/{kernel}")
    sm.run_emcee(lnprior=lnprior, nwalkers=50, nsteps=int(1e5), opt_init=False)
    sm.plot(plots=["emcee_corner"])

    sm.run_dynesty(ptform=prior_transform, mode='dynamic')
    sm.plot(plots=["dynesty_all"])

 