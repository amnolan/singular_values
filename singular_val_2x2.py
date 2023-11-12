def convert_to_float(input_row):
    output_row = []
    for i in input_row:
        output_row.append(float(i))
    return output_row

def two_d_deep_copy(src_array):
    dest_array = []
    for i in range(0,len(src_array)):
        sub_ra = []
        for j in range(0,len(src_array[i])):
            sub_ra.append(src_array[i][j])
        dest_array.append(sub_ra)
    return dest_array

def total_up_mean_matrix(ra):
    totals_list = []
    for i in range(0,len(ra)):
        total = 0
        for j in range(0,len(ra[i])):
            total += ra[i][j]
        totals_list.append(total)
    return totals_list

def get_x_hat_array(ra,mean_vector):
    #iterate over vectors
    for i in range(0,len(ra[0])):
        ra[0][i] = (ra[0][i] - mean_vector[0])
        ra[1][i] = (ra[1][i] - mean_vector[1])
    return ra

def transpose_matrix(matrix):

    rows = len(matrix)
    cols = len(matrix[0])
    
    transposed_matrix = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

def matrix_multiply(matrix1, matrix2):
    # Get the dimensions of the matrices
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    # Check if the matrices can be multiplied
    if cols1 != rows2:
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    # Initialize the result matrix with zeros
    result_matrix = [[0] * cols2 for _ in range(rows1)]

    # Perform matrix multiplication
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result_matrix[i][j] += matrix1[i][k] * matrix2[k][j]

    return result_matrix

def multiply_matrix_by_scalar(matrix, scalar):
    # Iterate through the matrix and multiply each element by the scalar
    result_matrix = [[element * scalar for element in row] for row in matrix]
    return result_matrix

def sample_mean(matrix, n):
    top = []
    bottom = []
    container = []
    for count in range(0,n):
        top.append((1/n)*parent_ra[0][count])
        bottom.append((1/n)*parent_ra[1][count])
    container.append(top)
    container.append(bottom)
    return container

def find_eigenvalues(matrix):
    try:
        # Check if the matrix is 2x2
        if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
            raise ValueError("Input matrix must be a 2x2 matrix")

        # Get matrix elements
        a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

        # Calculate eigenvalues using the quadratic formula
        discriminant = (a + d)**2 - 4 * (a * d - b * c)

        if discriminant >= 0:
            # Real eigenvalues
            eigenvalue1 = (a + d + discriminant**0.5) / 2
            eigenvalue2 = (a + d - discriminant**0.5) / 2
            return eigenvalue1, eigenvalue2
        else:
            # Complex eigenvalues
            real_part = (a + d) / 2
            imaginary_part = abs(discriminant)**0.5 / 2
            eigenvalue1 = complex(real_part, imaginary_part)
            eigenvalue2 = complex(real_part, -imaginary_part)
            return eigenvalue1, eigenvalue2

    except Exception as e:
        return f"Error: {str(e)}"

print("Singular Values")
print("Find A^T*A then find eigenvals")
print("Take sqrt() of eigenvals")
print("Input top row of matrix comma separated")
top_row = convert_to_float((input()).split(","))
print("Input bottom row of matrix comma separated")

bottom_row = convert_to_float((input()).split(","))
# top_row = [14,6,9,16,6,19]
# bottom_row = [3,15,8,7,6,5]
parent_ra = []
parent_ra.append(top_row)
parent_ra.append(bottom_row)
#save the original state
original_ra = two_d_deep_copy(parent_ra)
a_t = transpose_matrix(parent_ra)
a_t_a = matrix_multiply(parent_ra,a_t)
print(a_t_a)

find_eigenvalues
