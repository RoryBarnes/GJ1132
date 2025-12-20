import alabi
from alabi import SurrogateModel
from alabi import utility as ut
import vplanet_inference as vpi
import astropy.units as u
import numpy as np
from functools import partial
from sklearn import preprocessing
import matplotlib.pyplot as plt
import corner
import os
import vplot
import multiprocessing

# # Force single-threaded execution for underlying libraries to prevent nested threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def plot_1to1_test_error(sm):
    surrogate_fn = sm.create_cached_surrogate_likelihood(iter=nactive)
    ypred = surrogate_fn(sm.theta_test).flatten()
    ytrue = sm.y_test.flatten()
    ydiff = ypred - ytrue
    bias, scatter = np.mean(ydiff), np.std(ydiff)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(ytrue, ypred, c="r", s=5, label=f"bias={bias:.3f}, scatter={scatter:.3f}")
    plt.plot([np.min(ytrue), 0], [np.min(ytrue), 0], 'k--', lw=2)
    plt.xlim(np.min(ytrue), 0)
    plt.ylim(np.min(ytrue), 0)
    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel("True log-likelihood", fontsize=16)
    plt.ylabel("Surrogate log-likelihood", fontsize=16)
    plt.title(f"Gaussian {ndim}d", fontsize=18)
    plt.close()
    return fig

# ===========================================================================
# Configure vplanet forward model
# ===========================================================================

inpath = os.path.join(vpi.INFILE_DIR, "stellar")

inparams  = {"star.dMass": u.Msun,          
            "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),   
            "star.dSatXUVTime": u.Gyr,    
            "vpl.dStopTime": u.Gyr,       
            "star.dXUVBeta": -u.dimensionless_unscaled}

outparams = {"final.star.Luminosity": u.Lsun,
            "final.star.LXUVStellar": u.Lsun}

vpm = vpi.VplanetModel(inparams, inpath=inpath, outparams=outparams, verbose=False)

# ===========================================================================
# Observational constraints
# ===========================================================================

# Data: (mean, stdev)
prior_data = [(0.181, 0.019),       # mass [Msun]
            (-2.92, 0.26),          # log(fsat) 
            (None, None),           # tsat [Gyr]
            (10**0.893, 10**0.144),       # age [Gyr]
            (-1.18, 0.31)]          # beta

like_data = np.array([[4.38e-3, 3.4e-4],   # Lbol [Lsun]
                        [-4.26, 0.15]])    # log(Lxuv/Lbol)

# Prior bounds
bounds = [(0.1, 0.3),        
        (-4.0, -1.0),
        (0.1, 5),
        (5, 12.0),
        (-2.0, 0.0)]

# ===========================================================================
# Configure prior and likelihood for dynesty
# ===========================================================================

# Prior transform - dynesty format
prior_transform = partial(ut.prior_transform_normal, bounds=bounds, data=prior_data)

def lnlike(theta):
    out = vpm.run_model(theta, remove=True)
    mdl = np.array([out[0], np.log10(out[1]/out[0])])
    lnl = -0.5 * np.sum(((mdl - like_data.T[0])/like_data.T[1])**2)
    return lnl

# ===========================================================================
# Training configuration 
# ===========================================================================

ndim = len(bounds)
ntrain = 100*ndim               # number of initial training samples
ntest = 100*ndim                # number of initial test samples
nactive = 500                   # number of active learning iterations
savedir = "gj1132_results/"     # where to save/load results
#savedir = "trappist_results/"     # where to save/load results
ncore = max(1, multiprocessing.cpu_count() - 1)  # use all cores minus one

labels = [r"$m_{\star}$ [M$_{\odot}$]", r"$f_{sat}$", r"$t_{sat}$ [Gyr]", r"Age [Gyr]", r"$\beta_{XUV}$"]

# optimize hyperparameters using cross-validation
gp_kwargs = {"kernel": "ExpSquaredKernel",
             "fit_amp": True,
             "fit_mean": False,
             "fit_white_noise": False,
             "white_noise": -12,
             "uniform_scales": False,
             "hyperopt_method": "cv",
             "gp_opt_method": "l-bfgs-b",
             "gp_scale_rng": [-2,6],
             "gp_amp_rng": [-1,1],
             "cv_folds": 20,
             "cv_n_candidates": 400,
             "cv_stage2_candidates": 200,
             "cv_stage2_width": 0.3,
             "cv_stage3_candidates": 100,
             "cv_stage3_width": 0.1,
             "cv_weighted_factor": 1.0,
             "multi_proc": True}

al_kwargs = {"algorithm": "bape", 
            #  "nchains": 1,
             "gp_opt_freq": 100, 
             "obj_opt_method": "nelder-mead", 
             "use_grad_opt": False,
             "nopt": 1,
             "optimizer_kwargs": {"max_iter": 200, "xatol": 1e-3, "fatol": 1e-2, "adaptive": True}}

if __name__ == '__main__':
# ===========================================================================
# Run alabi
# ===========================================================================

        sm = SurrogateModel(lnlike_fn=lnlike,
            bounds=bounds,
            theta_scaler=preprocessing.StandardScaler(),
            y_scaler=preprocessing.StandardScaler(),
            savedir=savedir,
            cache=True,
            verbose=True,
            ncore=ncore)

        sm.init_samples(ntrain=ntrain, ntest=ntest, reload=True, sampler="lhs")
        sm.init_gp(**gp_kwargs)
        sm.active_train(niter=nactive, **al_kwargs)

        # ------------------------------------------------------------------
        # If alabi is already trained, comment out the section above and reload the surrogate model

        sm = alabi.load_model_cache(savedir)

        # Plot 1-1 test error
        fig = plot_1to1_test_error(sm)
        fig.savefig(savedir + f"1to1_test_error_iter_{nactive}.png", dpi=300)

        # ------------------------------------------------------------------
        # Run MCMC

        # # create a cached version of the trained likelihood for quicker evaluation
        surrogate_fn = sm.create_cached_surrogate_likelihood(iter=nactive)

        dynesty_sampler_kwargs = {"bound": "single",
                                "nlive": 100*ndim,
                                "sample": "auto"}

        dynesty_run_kwargs = {"wt_kwargs": {"pfrac": 1.0}, 
                        "stop_kwargs": {"pfrac": 1.0}, 
                        "maxiter": None, 
                        "dlogz_init": 0.5, 
                        "n_effective": int(1e4)*ndim}

        emcee_kwargs = {"nwalkers": 50*ndim,
                        "nsteps": int(1e5),
                        "burn": int(1e4)}

        pymultinest_kwargs = {"n_live_points": 100*ndim,
                        "sampling_efficiency": 0.8,
                        "evidence_tolerance": 0.5,
                        "max_modes": 8}

        ultranest_kwargs = {"min_num_live_points": 100*ndim,
                        "min_ess": int(5e3)*ndim,
                        "frac_remain": 0.5,
                        "dlogz": 0.5,
                        "dKL": 0.5}

        sm.run_dynesty(like_fn=surrogate_fn,  
                prior_transform=prior_transform, 
                sampler_kwargs=dynesty_sampler_kwargs,  
                run_kwargs=dynesty_run_kwargs,  
                        multi_proc=False,
                samples_file=f"dynesty_samples_final_custom_iter_{nactive}.npz")


        # sm.run_pymultinest(like_fn=surrogate_fn,
        #                    prior_transform=prior_transform,
        #                    sampler_kwargs=pymultinest_kwargs,
        #                    samples_file=f"pymultinest_samples_final_custom_iter_{nactive}.npz")

        # sm.run_ultranest(like_fn=surrogate_fn,
        #                  prior_transform=prior_transform,
        #                  run_kwargs=ultranest_kwargs,
        #                  samples_file=f"ultranest_samples_final_custom_iter_{nactive}.npz")
        
        # # ------------------------------------------------------------------
        # # Plot corner

        fig = corner.corner(sm.dynesty_samples, color="k", labels=labels, range=sm.bounds, bins=30, hist_kwargs={"density":True}, label_kwargs={"fontsize":20})
        fig.savefig(savedir + f"gj1132_corner.png", dpi=300)

