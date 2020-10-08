import numpy as np

import src.matrices as matrices
import src.rankmetric as rankmetric

def main():
    n, m = 2, 3
    p = 2

    # Base
    B = matrices.random_nonzero_matrices(n, m, 3, p)
    print("Base:", matrices.show_matrices(B), sep="\n")

    # Rank metric code
    C = rankmetric.linear_space_spanned_by(B, p)
    print("Generated Rank-metric code:", matrices.show_matrices(C), sep="\n")

    print("γ(C) =", rankmetric.gamma(C, p))

    C1 = rankmetric.linear_space_spanned_by(
        [
         np.array([[1,0,0],
                   [0,1,1],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [0,0,1]])
        ], p)
    print("γ(C1) =", rankmetric.gamma(C1, p))

    C2 = rankmetric.linear_space_spanned_by(
        [
         np.array([[1,0,0],
                   [0,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [1,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [0,1,0]])
        ], p)
    print("γ(C2) =", rankmetric.gamma(C2, p))

    C3 = rankmetric.linear_space_spanned_by(
        [
         np.array([[1,0,0],
                   [0,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [1,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [1,0,0]])
        ], p)
    print("γ(C3) =", rankmetric.gamma(C3, p))

    C4 = rankmetric.linear_space_spanned_by(
        [
         np.array([[1,0,0],
                   [0,0,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,1,0],
                   [0,0,0]]),
         np.array([[0,0,0],
                   [0,0,0],
                   [0,0,1]])
        ], p)
    print("γ(C4) =", rankmetric.gamma(C4, p))


if __name__ == "__main__":
    main()