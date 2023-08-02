import pymc as pm


def hierarchical_priors(name, dims, beta=1, fix_hyper_sigma=None):
    sigma = (
        pm.HalfCauchy(f"sigma_{name}", beta=beta)
        if fix_hyper_sigma is None
        else fix_hyper_sigma
    )
    values = (pm.Normal(f"{name}_raw", 0, 1, dims=dims)) * sigma
    values = pm.Deterministic(f"{name}", values, dims=dims)
    return values
