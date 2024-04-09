def matrix_shape(matrix):
    """Calculate the shape of a matrix represented as a list of lists."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
