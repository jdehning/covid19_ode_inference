import functools

import diffrax
import pytensor.tensor as pt
import numpy as np
from pytensor.tensor.type import TensorType

from covid19_ode_inference.pytensor_op import create_and_register_jax


def SIR(t, y, args):
    S, I, R = y
    β, (γ, N) = args
    dS = -β.evaluate(t) * I * S / N
    dI = β.evaluate(t) * I * S / N - γ * I
    dR = γ * I
    return dS, dI, dR


class CompModelsIntegrator:
    """
    Creates an integrator for compartmental models. If called with an ODE as arguments,
    it returns
    """

    def __init__(
        self,
        ts_out,
        t_0,
        ts_solver=None,
        ts_arg=None,
        interp="cubic",
        solver=diffrax.Tsit5(),
        t_1=None,
        **kwargs,
    ):
        self.ts_out = ts_out
        self.t_0 = t_0
        if t_1 is None:
            t_1 = max(self.ts_out)
        self.t_1 = float(t_1)
        if ts_solver is None:
            self.ts_solver = self.ts_out
        else:
            self.ts_solver = ts_solver
        self.ts_arg = ts_arg
        self.interp = interp
        self.solver = solver
        self.kwargs_solver = kwargs

    def get_func(self, ODE):
        def integrator(y0, arg_t=None, constant_args=None):
            if arg_t is not None:
                arg_t_interp = interpolation_func(
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
                dt0=None,
                stepsize_controller=diffrax.StepTo(ts=self.ts_solver),
                y0=tuple(y0),
                args=args,
                saveat=saveat,
                **self.kwargs_solver,
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


def interpolation_func(ts, x, method):
    if method == "cubic":
        coeffs = diffrax.backward_hermite_coefficients(ts, x)
        interp = diffrax.CubicInterpolation(ts, coeffs)
    elif method == "linear":
        interp = diffrax.LinearInterpolation(ts, x)
    else:
        raise RuntimeError(
            f'Interpoletion method {method} not known, possibilities are "cubic" or "linear"'
        )
    return interp


def interpolate(ts_in, ts_out, y, method, ret_gradients=False):
    def interpolator(ts_out, y):
        interp = interpolation_func(ts_in, y, method)
        if ret_gradients:
            return interp.derivative(ts_out)
        else:
            return interp.evaluate(ts_out)

    interpolator_op = create_and_register_jax(
        interpolator,
        output_types=[
            TensorType(dtype="float64", shape=(len(ts_out),)),
        ],
    )

    return interpolator_op(ts_out, y)
