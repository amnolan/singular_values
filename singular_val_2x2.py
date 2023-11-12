import math
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

def find_eigenvalues(matrix):
    try:
        
        if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
            raise ValueError("Input matrix must be a 2x2 matrix")

        a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]

        discriminant = (a + d)**2 - 4 * (a * d - b * c)

        if discriminant >= 0:
            eigenvalue1 = (a + d + discriminant**0.5) / 2
            eigenvalue2 = (a + d - discriminant**0.5) / 2
            return eigenvalue1, eigenvalue2
        else:
            real_part = (a + d) / 2
            imaginary_part = abs(discriminant)**0.5 / 2
            eigenvalue1 = complex(real_part, imaginary_part)
            eigenvalue2 = complex(real_part, -imaginary_part)
            return eigenvalue1, eigenvalue2

    except Exception as e:
        return print("Error: ",str(e))

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
print("A^T*A\n")
print(a_t_a)
print("Any key to continue")
input()
eigenvals = find_eigenvalues(a_t_a)
print("Eigenvals:\n")
print(eigenvals)
print("eigenval sqrt() o-1 =")
print(math.sqrt(eigenvals[0]))
print("eigenval sqrt() o-2 =")
print(math.sqrt(eigenvals[1]))
