import numpy as np

import src.matrices as mat
import src.rankmetric as rm

def main():
    n, m = 2, 3
    p = 2

    # Base
    B = mat.random_nonzero_matrices(n, m, 3, p)
    print("Base:", mat.show_matrices(B), sep="\n")

    # Rank metric code
    C = rm.linear_space_spanned_by(B, p)
    print("Generated Rank-metric code:", mat.show_matrices(C), sep="\n")

    print("γ(C) =", rm.gamma(C, p, get_evidence=True))

    print("--------------------------------------------------------------")

    B1= [
         np.array([[1,0,0],
                   [0,1,1],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [0,0,1]])
        ]
    C1 = rm.linear_space_spanned_by(B1, p)
    print("Pseudo-generator matrix of C1:")
    print(mat.show_matrix(rm.pseudo_generator_matrix(B1)))
    print("γ(C1) =", rm.gamma(C1, p))

    B2 = [
         np.array([[1,0,0],
                   [0,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [1,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [0,1,0]])
        ]
    C2 = rm.linear_space_spanned_by(B2, p)
    print("Pseudo-generator matrix of C2:")
    print(mat.show_matrix(rm.pseudo_generator_matrix(B2)))
    print("γ(C2) =", rm.gamma(C2, p))

    B3= [
         np.array([[1,0,0],
                   [0,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [1,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [1,0,0]])
        ]
    C3 = rm.linear_space_spanned_by(B3, p)
    print("Pseudo-generator matrix of C3:")
    print(mat.show_matrix(rm.pseudo_generator_matrix(B3)))
    print("γ(C3) =", rm.gamma(C3, p))

    B4= [
         np.array([[1,0,0],
                   [0,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,1,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [0,0,1]])
        ]
    C4 = rm.linear_space_spanned_by(B4, p)
    print("Pseudo-generator matrix of C4:")
    print(mat.show_matrix(rm.pseudo_generator_matrix(B4)))
    print("γ(C4) =", rm.gamma(C4, p))


if __name__ == "__main__":
    main()