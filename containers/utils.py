from typing import List, Tuple


def to_matrix_dims(dims: List[int]) -> Tuple[int, int]:
    if len(dims) == 0:
        return (1, 1)

    cols = 1
    for idx in range(len(dims) - 1):
        cols *= dims[idx]
    return cols, dims[-1]
