import numpy as np

def invert_triangular(matrix, lower=True):
    """
    Invert a triangular matrix using NumPy with vectorization.
    
    Parameters:
    matrix (numpy.ndarray): The triangular matrix to invert.
    lower (bool): True if the matrix is lower triangular, False if upper triangular.
    
    Returns:
    numpy.ndarray: The inverted triangular matrix.
    """
    
    # Check if the input is a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    
    n = matrix.shape[0]
    inverted = np.zeros_like(matrix, dtype=float)
    
    if lower:
        # Lower triangular matrix inversion
        for i in range(n):
            inverted[i, i] = 1 / matrix[i, i]
            inverted[i, :i] = -np.dot(matrix[i, :i], inverted[:i, :i]) / matrix[i, i]
    else:
        # Upper triangular matrix inversion
        for i in range(n - 1, -1, -1):
            inverted[i, i] = 1 / matrix[i, i]
            inverted[i, i+1:] = -np.dot(matrix[i, i+1:], inverted[i+1:, i+1:]) / matrix[i, i]
    
    return inverted

# Example usage and testing
if __name__ == "__main__":
    # Lower triangular matrix
    L = np.array([
        [1, 0, 0],
        [2, 3, 0],
        [4, 5, 6]
    ])
    
    # Upper triangular matrix
    U = np.array([
        [1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]
    ])
    
    # Test lower triangular
    L_inv = invert_triangular(L, lower=True)
    print("Lower triangular matrix:")
    print(L)
    print("\nInverted lower triangular matrix:")
    print(L_inv)
    print("\nL * L_inv (should be identity):")
    print(np.round(np.matmul(L, L_inv), decimals=10))
    
    print("\n" + "="*50 + "\n")
    
    # Test upper triangular
    U_inv = invert_triangular(U, lower=False)
    print("Upper triangular matrix:")
    print(U)
    print("\nInverted upper triangular matrix:")
    print(U_inv)
    print("\nU * U_inv (should be identity):")
    print(np.round(np.matmul(U, U_inv), decimals=10))