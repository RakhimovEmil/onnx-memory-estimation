from typing import List, Optional, Tuple
from math import prod

import onnx

from containers.utils import to_matrix_dims

class ConstTensor:
    def __init__(self, dims: List[int]):
        self.dims = dims


class ConstTensorInfo:
    def __init__(self, tensor: onnx.TensorProto, continuous: bool):  # continuous -- strange thing
        self.external: bool = tensor.data_location == onnx.TensorProto.DataLocation.Value('EXTERNAL')
        self.dims = list(tensor.dims)
        self.source = tensor.raw_data
        self.data_type = onnx.TensorProto.DataType.Name(tensor.data_type)
        self.cols, self.rows = to_matrix_dims(self.dims)
        self.stride_pos: Optional[int] = None

        if len(self.dims) > 2 and not continuous:
            self.stride_pos = len(tensor.dims) - 2
        
        if self.external:
            for item in tensor.external_data:
                if item.key == 'location':
                    self.source = item.value
                elif item.key == 'offset':
                    self.offset = int(item.value)
                elif item.key == 'length':
                    self.length = int(item.value)

