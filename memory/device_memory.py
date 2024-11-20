from memory.tensor_info import TensorInfo
from math import prod


class DeviceMemory:
    def __init__(self, alignment: int) -> None:
        self.alignment = alignment

    @staticmethod
    def align_up(value: int, align: int) -> int:
        return (value + align - 1) // align * align

    def align_size(self, size: int, type: str) -> int:
        if self.alignment % DeviceMemory.data_type_to_bytes[type]:
            raise ValueError(f"Invalid alignment {size} for type {type}")
        return DeviceMemory.align_up(
            size, self.alignment // DeviceMemory.data_type_to_bytes[type]
        )

    def align_device_block(self, size: int) -> int:
        return DeviceMemory.align_up(size, self.alignment)

    def get_matrix_size(
        self, cols: int, rows: int, type: str, align_rows: bool = True
    ) -> int:
        # In terms of column-major matrices
        rows_aligned = self.align_size(rows, type) if align_rows else rows
        return self.align_device_block(
            cols * rows_aligned * DeviceMemory.data_type_to_bytes[type]
        )

    @staticmethod
    def to_matrix_dims(dims: list[int]) -> tuple[int, int]:
        if not dims:
            return 1, 1
        return prod(dims[:-1]), dims[-1]
    
    def get_tensor_size(self, tensor: TensorInfo) -> int:
        cols, rows = DeviceMemory.to_matrix_dims(tensor.dims)
        return self.get_matrix_size(cols, rows, tensor.dtype, True)

    data_type_to_bytes = {
        "FLOAT": 4,
        "UINT8": 1,
        "INT8": 1,
        "UINT16": 2,
        "INT16": 2,
        "INT32": 4,
        "INT64": 8,
        # 'STRING': 12,
        "BOOL": 1,
        "FLOAT16": 2,
        "DOUBLE": 8,
        "UINT32": 4,
        "UINT64": 8,
        # 'COMPLEX64': 8,
        # 'COMPLEX128': 16,
        "BFLOAT16": 2,
        # 'FLOAT8E4M3FN': 1,
        # 'FLOAT8E4M3FNUZ': 1,
        # 'FLOAT8E5M2': 1,
        # 'FLOAT8E5M2FNUZ': 1,
        # 'UINT4': 1,
        # 'INT4': 1,
    }

class DevicePointer:
    def __init__(self, type: str, segment: int, offset: int):
        self.type = type
        self.segment = segment
        self.offset = offset
