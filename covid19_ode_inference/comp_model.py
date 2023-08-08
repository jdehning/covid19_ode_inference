import diffrax
from pytensor.tensor.type import TensorType
import graphviz

from covid19_ode_inference.pytensor_op import create_and_register_jax


def SIR(t, y, args):
    β, const_arg = args
    γ = const_arg["gamma"]
    N = const_arg["N"]
    dS = -β(t) * y["I"] * y["S"] / N
    dI = β(t) * y["I"] * y["S"] / N - γ * y["I"]
    dR = γ * y["I"]

    return {"S": dS, "I": dI, "R": dR}


def Erlang_SEIR(t, y, args):
    beta, const_arg = args
    N = const_arg["N"]
    dComp = {}

    I = sum(y["Is"])
    dComp["S"] = -beta(t) * I * y["S"] / N

    # Latent period
    dEs, outflow = erlang_kernel(
        inflow=beta(t) * I * y["S"] / N,
        Vars=y["Es"],
        rate=const_arg["rate_latent"],
    )
    dComp["Es"] = dEs

    # Infectious period
    dIs, outflow = erlang_kernel(
        inflow=outflow,
        Vars=y["Is"],
        rate=const_arg["rate_infectious"],
    )
    dComp["Is"] = dIs

    dComp["R"] = outflow
    return dComp


def Erlang_SEIRS(t, y, args):
    """
    Differential equations of an SEIRS model with Erlang distributed latent, infectious
    and recovering (recovered-to-susceptible) periods.
    Parameters
    ----------
    t: float
        time variable
    y: dict
        dictionary of compartments. Has to include keys "S", "Es", "Is", "Rs". The value
        of "Es", "Is", "Rs" have to be a list of length `n` where `n` is the number of
        compartments/shape of the Erlang distribution/kernel.
    args: tuple: (callable, dict)
        Callable is a function with argument t that returns the value of the transmission
        rate beta(t)
        dict is the constant arguments of the model. Has to include keys "N", "rate_latent",
        "rate_infectious" and "rate_recovery".
    Returns
    -------
    dComp: dict
        Same as y, but with the derivatives of the compartments.
    """

    beta, const_arg = args
    N = const_arg["N"]
    dComp = {}

    I = sum(y["Is"])
    dComp["S"] = -beta(t) * I * y["S"] / N

    # Latent period
    dEs, outflow = erlang_kernel(
        inflow=beta(t) * I * y["S"] / N,
        Vars=y["Es"],
        rate=const_arg["rate_latent"],
    )
    dComp["Es"] = dEs

    # Infectious period
    dIs, outflow = erlang_kernel(
        inflow=outflow,
        Vars=y["Is"],
        rate=const_arg["rate_infectious"],
    )
    dComp["Is"] = dIs

    # Recovered/non-susceptible period
    dRs, outflow = erlang_kernel(
        inflow=outflow,
        Vars=y["Rs"],
        rate=const_arg["rate_recovery"],
    )
    dComp["Rs"] = dRs

    dComp["S"] = dComp["S"] + outflow

    return dComp


def erlang_kernel(inflow, Vars, rate):
    """
    Erlang kernel for a compartmental model. The shape is determined by the length of Vars.
    Parameters
    ----------
    inflow: float or ndarray
        The inflow into the first compartment of the kernel
    Vars: list of floats or list of ndarrays
        The compartments of the kernel
    rate: float or ndarray
        The rate of the kernel, 1/rate is the mean time spent in total in all compartments
    Returns
    -------
    dVars: list of floats or list of ndarrays
        The derivatives of the compartments
    out_flow: float or ndarray
        The outflow from the last compartment
    """

    dVars = []
    m = len(Vars)
    for i, Var_i in enumerate(Vars):
        dVars.append(-m * rate * Var_i)
        if i == 0:
            dVars[0] = dVars[0] + inflow
        else:
            dVars[i] = dVars[i] + m * rate * Vars[i - 1]
    out_flow = m * rate * Vars[-1]
    return dVars, out_flow


class CompModel:
    """
    Class to build a compartmental model. The model is built by adding flows between
    compartments. The model is then compiled into a function that can be used in an ODE.

    Parameters
    ----------
    Comp_dict: dict
        Dictionary of compartments. Keys are the names of the compartments and values
        are floats or ndarrays that represent their value.
    """

    def __init__(self, Comp_dict):
        self.Comp = Comp_dict
        self.dComp = {}
        self.graph = graphviz.Digraph("comp_model")
        for key in self.Comp.keys():
            self.graph.node(key)
        self.graph.attr(rankdir="LR")

    def flow(self, start_comp, end_comp, flow, label=None):
        """
        Add a flow from start_comp to end_comp with rate flow.

        Parameters
        ----------
        start_comp: str or list
            Key of the start compartment. Can also be a list of keys in which case an
            identical flow is added from each compartment of the list to end_com
        end_comp: str
            Key of the end compartment
        flow: float or ndarray
            flow to add between compartments, is multiplied by start_comp, so it should
            be broadcastable whith it.
        label: str
            label of the edge between the compartments that will be used when displaying
            a graph of the compartmental model.

        Returns
        -------
        None
        """
        if isinstance(start_comp, str):
            start_comp_list = [start_comp]
        else:
            start_comp_list = start_comp
        for start_comp in start_comp_list:
            if not start_comp in self.dComp.keys():
                self.dComp[start_comp] = 0
            if not end_comp in self.dComp.keys():
                self.dComp[end_comp] = 0
            self.dComp[start_comp] = (
                self.dComp[start_comp] - flow * self.Comp[start_comp]
            )
            self.dComp[end_comp] = self.dComp[end_comp] + flow * self.Comp[start_comp]
            self.graph.edge(start_comp, end_comp, label=label)

    @property
    def dy(self):
        return self.dComp

    def view_graph(self, on_display=True):
        if on_display:
            try:
                from IPython.display import display

                display(self.graph)
            except:
                self.graph.view()
        else:
            self.graph.view()


def delayed_copy(initial_var, delayed_vars, tau_delay):
    """
    Delayed copy of a compartment. The delay has the form of an Erlang kernel with shape
    parameter len(delayed_vars)
    Parameters
    ----------
    initial_var
    delayed_vars
    tau_delay

    Returns
    -------

    """

    shape = len(delayed_vars)
    inflow = initial_var / tau_delay * shape
    if shape == 1:
        d_delayed_vars = []
        d_delayed_vars_last = inflow
    elif shape > 1:
        d_delayed_vars, outflow = erlang_kernel(
            inflow, delayed_vars[:-1], 1 / tau_delay * shape
        )
        d_delayed_vars_last = outflow
    return d_delayed_vars + [d_delayed_vars_last]


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
        elif isinstance(ts_solver, diffrax.AbstractStepSizeController):
            self.ts_solver = ts_solver
        else:
            self.ts_solver = diffrax.StepTo(ts=ts_solver)
        self.ts_arg = ts_arg
        self.interp = interp
        self.solver = solver
        self.kwargs_solver = kwargs

    def get_func(self, ODE):
        def integrator(y0, arg_t=None, constant_args=None):
            if arg_t is not None:
                if not callable(arg_t):
                    arg_t_func = interpolation_func(
                        ts=self.ts_arg, x=arg_t, method=self.interp
                    ).evaluate
                else:
                    arg_t_func = arg_t

            term = diffrax.ODETerm(ODE)

            if arg_t is None:
                args = constant_args
            elif constant_args is None:
                args = arg_t_func
            else:
                args = (
                    arg_t_func,
                    constant_args,
                )
            saveat = diffrax.SaveAt(ts=self.ts_out)

            sol = diffrax.diffeqsolve(
                term,
                self.solver,
                self.t_0,
                self.t_1,
                dt0=None,
                stepsize_controller=self.ts_solver,
                y0=y0,
                args=args,
                saveat=saveat,
                **self.kwargs_solver,
                # adjoint=diffrax.BacksolveAdjoint(),
            )

            return sol.ys

        return integrator

    def get_Op(self, ODE, name=None, return_shapes=((),)):
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


def interpolation_func(ts, x, method="cubic"):
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


def interpolate(ts_in, ts_out, y, method="cubic", ret_gradients=False):
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
