import numpy as np
import pymc as pm
import pytensor.tensor as pt


def sigmoidal_changepoints(time, time_cp, magnitude, durations, reorder_cps=True):
    """

    Parameters
    ----------
    time: 1d numpy array
        timepoints where modulation is evaluated, shape: (time, )
    time_cp: nd-array
        timepoints of the changepoints, shape: (num_cps, further dims...)
    magnitudes: nd-array
        magnitude of the changepoints, shape: (num_cps, further dims...)
    durations: nd-array
        magnitude of the changepoints, shape: (num_cps, further dims...)
    reorder_cps: bool, default=True
        reorder changepoints such that their timepoints are linearly increasing
    Returns
    -------
        nd-array, shape: (time, further dims...)

    """
    # if reorder_cps:
    #    order = pt.argsort(time_cp, axis=0)
    #    time_cp = time_cp[order, ...]
    #    magnitude = time_cp[order, ...]
    #    durations = time_cp[order, ...]

    # add necessary empty dimensions to time axis
    time = np.expand_dims(time, axis=tuple(range(1, max(1, time_cp.ndim) + 1)))
    slope_cp = pt.abs(magnitude) / durations
    modulation_t = (
        pt.sigmoid((time - time_cp) * slope_cp * 4) * magnitude
    )  # 4*slope_cp because the derivative of the sigmoid at zero is 1/4, we want to set it to slope_cp

    modulation_t = pt.sum(modulation_t, axis=1)

    return modulation_t
