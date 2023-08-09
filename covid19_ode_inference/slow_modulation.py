import numpy as np
import jax.numpy as jnp

import pymc as pm
import pytensor.tensor as pt
from pytensor.link.jax.dispatch.basic import jax_funcify
from pytensor.tensor.sort import ArgSortOp

from .tools import hierarchical_priors


def sigmoidal_changepoints(
    ts_out, positions_cp, magnitudes_cp, durations_cp, reorder_cps=False
):
    """
    Modulation of a time series by sigmoidal changepoints. The changepoints are defined
    by their position, magnitude and duration. The resulting equation is:
    $$ f(t) = \sum_{i=1}^{num_cps} \frac{magnitudes[i]}{1 + exp(-4 * slope[i] * (t - positions[i]))} $$
    where $slope[i] = magnitudes[i] / durations[i]$

    Parameters
    ----------
    t_out: 1d numpy array
        timepoints where modulation is evaluated, shape: (time, )
    t_cp: nd-array
        timepoints of the changepoints, shape: (num_cps, further dims...)
    magnitudes: nd-array
        magnitude of the changepoints, shape: (num_cps, further dims...)
    durations: nd-array
        magnitude of the changepoints, shape: (num_cps, further dims...)
    reorder_cps: bool, default=False
        reorder changepoints such that their timepoints are linearly increasing
    Returns
    -------
        nd-array, shape: (time, further dims...)

    """
    if reorder_cps:
        order = pt.argsort(positions_cp, axis=0)
        positions_cp = positions_cp[order, ...]
        magnitudes_cp = magnitudes_cp[order, ...]
        durations_cp = durations_cp[order, ...]

    # add necessary empty dimensions to time axis
    ts_out = np.expand_dims(ts_out, axis=tuple(range(1, max(1, positions_cp.ndim) + 1)))
    slope_cp = 1 / durations_cp
    modulation_t = (
        pt.sigmoid((ts_out - positions_cp) * slope_cp * 4) * magnitudes_cp
    )  # 4*slope_cp because the derivative of the sigmoid at zero is 1/4, we want to set it to slope_cp

    modulation_t = pt.sum(modulation_t, axis=1)

    return modulation_t


def priors_for_cps(
    cp_dim,
    time_dim,
    name_positions,
    name_magnitudes,
    name_durations,
    beta_magnitude=1,
    sigma_magnitude_fix=None,
    model=None,
):
    """
    Create priors for changepoints. Their positions are uniformly distributed between
    the first and last timepoint. The magnitudes are sampled from a hierarchical prior
    with a beta distribution. The durations are sampled from a normal distribution with
    mean equal to the mean distance between changepoints and standard deviation equal to
    the standard deviation of the distance between changepoints.

    Parameters
    ----------
    cp_dim : str
        dimension of the pm.Model for the changepoints. Define it by passing
        coords={cp_dim: np.arange(num_cps)} to pm.Model at creation. The length of this
        dimension determines the number of changepoints.
    time_dim : str
        dimension of the pm.Model for the time.
    name_positions : str
        name under which the positions of the changepoints are stored in pm.Model
    name_magnitudes : str
        name under which the magnitudes of the changepoints are stored in pm.Model
    name_durations : str
        name under which the durations of the changepoints are stored in pm.Model
    beta_magnitude : float, default=1
        beta parameter of the hierarchical prior for the magnitudes
    sigma_magnitude_fix : float, default=None
        if not None, the standard deviation from which the magnitudes are sampled is fixed
    model : pm.Model, default=None
        pm.Model in which the priors are created. If None, the pm.Model is taken from the
        the context.

    Returns
    -------
    positions, magnitudes, durations : pm.Variable
        pm.Variables for the positions, magnitudes and durations of the changepoints

    """
    model = pm.modelcontext(model)

    time_arr = model.coords[time_dim]
    num_cps = len(model.coords[cp_dim])
    interval_cps = (max(time_arr) - min(time_arr)) / (num_cps + 1)

    ### Positions
    std_Delta_pos = interval_cps / 3
    Delta_pos = pm.Normal(f"Delta_{name_positions}", 0, std_Delta_pos, dims=(cp_dim,))
    positions = Delta_pos + np.arange(1, num_cps + 1) * interval_cps
    pm.Deterministic(f"{name_positions}", positions)

    ### Magnitudes:
    magnitudes = hierarchical_priors(
        name_magnitudes,
        dims=(cp_dim,),
        beta=beta_magnitude,
        fix_hyper_sigma=sigma_magnitude_fix,
    )

    ### Durations:
    mean_duration_len = interval_cps / 3
    std_duration_len = interval_cps / 6
    softplus_scaling = interval_cps / 12
    durations_raw = pm.Normal(
        f"{name_durations}_raw", mean_duration_len, std_duration_len, dims=(cp_dim,)
    )
    durations = pm.Deterministic(
        f"{name_durations}",
        pt.softplus(durations_raw / softplus_scaling) * softplus_scaling,
    )

    order = pt.argsort(positions, axis=0)
    positions = positions[order, ...]
    magnitudes = magnitudes[order, ...]
    durations = durations[order, ...]

    return positions, magnitudes, durations


@jax_funcify.register(ArgSortOp)
def jax_funcify_Argsort(op, node, **kwargs):
    def argsort(x, axis=None):
        return jnp.argsort(x, axis=axis)

    return argsort
