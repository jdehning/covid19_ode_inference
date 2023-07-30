import functools

import diffrax
import pytensor.tensor as pt
import numpy as np
from pytensor.tensor.type import TensorType

from covid19_ode_inference.pytensor_op import create_and_register_jax


def SIR_eqs(t, y, args):
    S, I, R = y
    β, (γ, N) = args
    dS_dt = -β.evaluate(t) * I * S / N
    dI_dt = β.evaluate(t) * I * S / N - γ * I
    dR_dt = -γ * I
    return dS_dt, dI_dt, dR_dt


def SIR_integrator(
    S_0,
    I_0,
    R_0,
    beta_t,
    other_args,
    time,
    dt,
    t_0,
    t_1,
):
    cubic_interp = True
    if cubic_interp:
        coeffs_beta = diffrax.backward_hermite_coefficients(time, beta_t)
        beta_interp = diffrax.CubicInterpolation(time, coeffs_beta)
    else:
        beta_interp = diffrax.LinearInterpolation(time, beta_t)

    term = diffrax.ODETerm(SIR_eqs)
    y0 = S_0, I_0, R_0
    args = beta_interp, other_args
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=time)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t_0,
        t_1,
        dt,
        y0,
        args=args,
        saveat=saveat,
        # adjoint=diffrax.BacksolveAdjoint(),
        # adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=int((t_1 - t_0) / dt)),
    )
    return sol.ys[0], sol.ys[1], sol.ys[2]


class CompModelsIntegrator:
    """
    Creates an integrator for compartmental models. If called with an ODE as arguments,
    it returns
    """

    def __init__(
        self,
        ts_out,
        t_0,
        dt,
        ts_arg=None,
        interp="cubic",
        solver=diffrax.Tsit5(),
        t_1=None,
    ):
        self.ts_out = ts_out
        self.t_0 = t_0
        if t_1 is None:
            t_1 = max(self.ts_out)
        self.t_1 = float(t_1)
        self.dt = dt
        self.ts_arg = ts_arg
        self.interp = interp
        self.solver = solver

    def get_func(self, ODE):
        def integrator(y0, arg_t=None, constant_args=None):
            if arg_t is not None:
                arg_t_interp = interpolation(
                    ts=self.ts_arg, x=arg_t, method=self.interp
                )

            term = diffrax.ODETerm(ODE)

            if arg_t is None:
                args = constant_args
            elif constant_args is None:
                args = arg_t_interp
            else:
                args = (
                    arg_t_interp,
                    constant_args,
                )
            saveat = diffrax.SaveAt(ts=self.ts_out)
            sol = diffrax.diffeqsolve(
                term,
                self.solver,
                self.t_0,
                self.t_1,
                self.dt,
                tuple(y0),
                args=args,
                saveat=saveat,
                # adjoint=diffrax.BacksolveAdjoint(),
            )

            return tuple([sol.ys[i] for i in range(len(y0))])

        return integrator

    def get_Op(self, ODE, name, return_shapes=((),)):
        integrator = self.get_func(ODE)

        pytensor_op = create_and_register_jax(
            integrator,
            output_types=[
                TensorType(
                    dtype="float64", shape=tuple([len(self.ts_out)] + list(shape))
                )
                for shape in return_shapes
            ],
            name=name,
        )
        return pytensor_op


def interpolation(ts, x, method):
    if ts is not None:
        if method == "cubic":
            coeffs = diffrax.backward_hermite_coefficients(ts, x)
            interp = diffrax.CubicInterpolation(ts, coeffs)
        elif method == "linear":
            interp = diffrax.LinearInterpolation(ts, x)
        else:
            raise RuntimeError(
                f'Interpoletion method {self.interp} not known, possibilities are "cubic" or "linear"'
            )
    return interp


def comp_models_integrator(
    ODE_eqs, y0, arg_t, ts_arg, other_args, ts_out, t_0, t_1, dt, interp="cubic"
):
    if interp == "cubic":
        coeffs_beta = diffrax.backward_hermite_coefficients(ts_arg, arg_t)
        arg_t_interp = diffrax.CubicInterpolation(ts_arg, coeffs_beta)
    elif interp == "linear":
        arg_t_interp = diffrax.LinearInterpolation(ts_arg, arg_t)

    term = diffrax.ODETerm(SIR_eqs)
    args = (
        arg_t_interp,
        other_args,
    )
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=time)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t_0,
        t_1,
        dt,
        y0,
        args=args,
        saveat=saveat,
        # adjoint=diffrax.BacksolveAdjoint(),
        # adjoint=diffrax.RecursiveCheckpointAdjoint(checkpoints=int(t_1 - t_0) / dt),
    )
    return tuple([sol.ys[i] for i in range(len(y0))])


def ODE_to_pytensor_op(integrator, num_outputs, time, dt, name, t_0=None, t_1=None):
    if t_0 is None:
        t_0 = min(time)
    if t_1 is None:
        t_1 = max(time)
    t_1 = float(t_1)

    ODE_with_defined_time = functools.partial(
        integrator, time=time, t_0=t_0, t_1=t_1, dt=dt
    )

    pytensor_op = create_and_register_jax(
        ODE_with_defined_time,
        output_types=[
            TensorType(dtype="float64", shape=(len(time),)) for _ in range(num_outputs)
        ],
        name=name,
    )

    return pytensor_op
