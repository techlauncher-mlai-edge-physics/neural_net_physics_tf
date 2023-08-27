import torch
    
def gaussian_log_prob(data, mean, cov):
    """
    Calculate a Gaussian likelihood given dist params.
    Expensive.
    """
    return torch.distributions.MultivariateNormal(
        mean, cov).log_prob(data)

def rsolve(A, B):
    """
    Solve a system of linear equations XA=B for x, where A is a square matrix and B is a matrix.

    Nearly trivial but annoying to work out in my tiny brain every time.
    """
    return torch.linalg.solve(B.T, A.T).T

def rcholesky_solve(C, B):
    """
    Solve a system of linear equations XA=B for x, where A=C @ C.T is a square matrix and B is a matrix.

    Nearly trivial but annoying to work out in my tiny brain every time.
    """
    return torch.cholesky_solve(B.T, C).T


class NumericalError(Exception):
    pass
