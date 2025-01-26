import numpy as np

def partial_dMF(x, mf_definition, partial_parameter):
    """Calculates the partial derivative of a membership function at a point x.

    Parameters
    ----------
    x : float
        The point at which the partial derivative is calculated.
    mf_definition : list
        The membership function definition, where the first element is the function name and the second element is a dictionary of parameters.
    partial_parameter : str
        The parameter for which the partial derivative is calculated ('sigma', 'mean', 'a', 'b', or 'c').

    Returns
    -------
    result : float
        The calculated partial derivative.
    """
    mf_name = mf_definition[0]

    if mf_name == 'gaussmf':
        sigma = mf_definition[1]['sigma']
        mean = mf_definition[1]['mean']

        if partial_parameter == 'sigma':
            result = (2. / sigma**3) * np.exp(-(((x - mean)**2) / (sigma)**2)) * (x - mean)**2
        elif partial_parameter == 'mean':
            result = (2. / sigma**2) * np.exp(-(((x - mean)**2) / (sigma)**2)) * (x - mean)

    elif mf_name == 'gbellmf':
        a = mf_definition[1]['a']
        b = mf_definition[1]['b']
        c = mf_definition[1]['c']

        if partial_parameter == 'a':
            result = (2. * b * np.power((c - x), 2) * np.power(np.absolute((c - x) / a), ((2 * b) - 2))) / \
                    (np.power(a, 3) * np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
        elif partial_parameter == 'b':
            result = -1 * (2 * np.power(np.absolute((c - x) / a), (2 * b)) * np.log(np.absolute((c - x) / a))) / \
                    (np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))
        elif partial_parameter == 'c':
            result = (2. * b * (c - x) * np.power(np.absolute((c - x) / a), ((2 * b) - 2))) / \
                    (np.power(a, 2) * np.power((np.power(np.absolute((c - x) / a), (2 * b)) + 1), 2))

    elif mf_name == 'sigmf':
        b = mf_definition[1]['b']
        c = mf_definition[1]['c']

        if partial_parameter == 'b':
            result = -1 * (c * np.exp(c * (b + x))) / \
                    np.power((np.exp(b * c) + np.exp(c * x)), 2)
        elif partial_parameter == 'c':
            result = ((x - b) * np.exp(c * (x - b))) / \
                    np.power((np.exp(c * (x - c))) + 1, 2)

    return result
