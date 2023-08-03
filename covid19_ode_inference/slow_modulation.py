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
    slope_cp = pt.abs(magnitudes_cp) / durations_cp
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
