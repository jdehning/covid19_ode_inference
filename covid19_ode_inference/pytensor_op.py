import functools
import inspect

import jax
import numpy as np

import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from pytensor.gradient import DisconnectedType


def create_and_register_jax(
    jax_func,
    output_types=(),
    input_dtype="float64",
    name=None,
):
    """
    Returns a pytensor from a jax jittable function. It requires to define the output
    types of the returned values as pytensor types. Beware that default values defined
    in the function definition of jax_func won't be evaluated. A unique name should also
    be passed in case the name of the jax_func is identical to some other node. The
    design of this function is based on https://www.pymc-labs.io/blog-posts/jax-functions-in-pymc-3-quick-examples/

    Parameters
    ----------
    jax_func : jax jittable function
        function for which the node is created, can return multiple tensors.
    output_types : list of pt.TensorType
        The shape of the TensorType has to be defined.
    input_dtype : str
        inputs are converted to this dtype
    name: str
        Name of the created pytensor Op, defaults to the name of the passed function.
        Should be unique so that jax_juncify won't ovewrite another when registering it
    Returns
    -------
        A Pytensor Op which can be used in a pm.Model as function, is differentiable
        and compilable with both JAX and C backend.

    """
    jitted_sol_op_jax = jax.jit(jax_func)
    len_gz = len(output_types)

    def vjp_sol_op_jax(*args):
        y0 = args[:-len_gz]
        gz = tuple(args[-len_gz:])
        _, vjp_fn = jax.vjp(jax_func, *y0)
        return vjp_fn(gz)

    jitted_vjp_sol_op_jax = jax.jit(vjp_sol_op_jax)

    class SolOp(Op):
        def make_node(self, *inputs, **kwinputs):
            # Convert keyword inputs to positional arguments
            func_signature = inspect.signature(jax_func)
            arguments = func_signature.bind(*inputs, **kwinputs)
            all_inputs = arguments.args

            # Convert our inputs to symbolic variables
            all_inputs = [
                pt.as_tensor_variable(inp).astype(input_dtype) for inp in all_inputs
            ]

            # Define our output variables
            outputs = [pt.as_tensor_variable(type()) for type in output_types]

            return Apply(self, all_inputs, outputs)

        def perform(self, node, inputs, outputs):
            """This function is called by the C backend, thus the numpy conversion"""
            results = jitted_sol_op_jax(*inputs)
            for i, _ in enumerate(output_types):
                outputs[i][0] = np.array(results[i], output_types[i].dtype)

        def grad(self, inputs, output_gradients):
            # If a output is not used, it is disconnected and doesn't have a gradient.
            # Set gradient here to zero for those outputs.
            for i, otype in enumerate(output_types):
                if isinstance(output_gradients[i].type, DisconnectedType):
                    output_gradients[i] = pt.zeros(otype.shape, otype.dtype)
            result = vjp_sol_op(inputs, output_gradients)
            results = [result[i] for i, _ in enumerate(inputs)]
            return results

    # vector-jacobian product Op
    class VJPSolOp(Op):
        def make_node(self, y0, gz):
            inputs = [
                pt.as_tensor_variable(
                    _y,
                ).astype(input_dtype)
                for _y in y0
            ] + list(gz)

            outputs = [input.type() for input in y0]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            results = jitted_vjp_sol_op_jax(*inputs)
            for i, result in enumerate(results):
                outputs[i][0] = np.array(result, input_dtype)

    if name is None:
        name = jax_func.__name__
    SolOp.__name__ = name
    SolOp.__qualname__ = ".".join(SolOp.__qualname__.split(".")[:-1] + [name])
    VJPSolOp.__name__ = "VJP" + name
    VJPSolOp.__qualname__ = ".".join(
        VJPSolOp.__qualname__.split(".")[:-1] + ["VJP" + name]
    )

    sol_op = SolOp()
    vjp_sol_op = VJPSolOp()

    @jax_funcify.register(SolOp)
    def sol_op_jax_funcify(op, **kwargs):
        return jax_func

    @jax_funcify.register(VJPSolOp)
    def vjp_sol_op_jax_funcify(op, **kwargs):
        return vjp_sol_op_jax

    return sol_op
