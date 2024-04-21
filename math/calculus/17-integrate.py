def poly_integral(poly, C=0):
    """Calculate the integral of a polynomial represented by coefficients."""
    if isinstance(poly, list) and all(isinstance(x, (int, float)) for x in poly) and isinstance(C, (int, float)):
        if not poly:  # Handle empty polynomial list explicitly
            return [C]
        integral = [C] + [poly[i] / (i + 1) for i in range(len(poly))]
        return [int(x) if isinstance(x, float) and x.is_integer() else x for x in integral]
    return None
