import functools

import diffrax
import pytensor.tensor as pt
import numpy as np
from pytensor.tensor.type import TensorType

from covid19_ode_inference.pytensor_op import create_and_register_jax


def SIR_eqs(t, y, args):
    S, I, R = y
    β, γ, N = args
    dS_dt = -β.evaluate(t) * I * S / N
    dI_dt = β.evaluate(t) * I * S / N - γ * I
    dR_dt = -γ * I
    return dS_dt, dI_dt, dR_dt


def SIR_integrator(
    S_0,
    I_0,
    R_0,
    beta_t,
    gamma,
    N,
    time,
    dt,
    t_0,
    t_end,
):
    cubic_interp = True
    if cubic_interp:
        coeffs_beta = diffrax.backward_hermite_coefficients(time, beta_t)
        beta_interp = diffrax.CubicInterpolation(time, coeffs_beta)
    else:
        beta_interp = diffrax.LinearInterpolation(time, beta_t)

    term = diffrax.ODETerm(SIR_eqs)
    y0 = S_0, I_0, R_0
    args = beta_interp, gamma, N
    solver = diffrax.Dopri5()
    saveat = diffrax.SaveAt(ts=time)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t_0,
        t_end,
        dt,
        y0,
        args=args,
        saveat=saveat,
        # adjoint=diffrax.BacksolveAdjoint(),
    )
    return sol.ys[0], sol.ys[1], sol.ys[2]


def ODE_to_pytensor_op(integrator, num_outputs, time, dt, name, t_0=None, t_end=None):
    if t_0 is None:
        t_0 = min(time)
    if t_end is None:
        t_end = max(time)
    t_end = float(t_end)

    ODE_with_defined_time = functools.partial(
        integrator, time=time, t_0=t_0, t_end=t_end, dt=dt
    )

    pytensor_op = create_and_register_jax(
        ODE_with_defined_time,
        output_types=[
            TensorType(dtype="float64", shape=(len(time),)) for _ in range(num_outputs)
        ],
        name=name,
    )

    return pytensor_op
