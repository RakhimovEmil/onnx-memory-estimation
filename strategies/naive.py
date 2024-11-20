from memory import TensorInfo, DeviceMemory
from expression import Expression
from containers import MutableTensorInfo

from typing import List
import onnx
from google.protobuf.json_format import MessageToDict


class NaiveTensorMemoryEstimator:
    def __init__(
        self,
        model: onnx.ModelProto,
        max_symbolic_param_values: dict[str, int],
        device_memory: DeviceMemory,
    ) -> None:
        self.memory = device_memory
        self.num_nodes = len(model.graph.node)
        self.total_allocated_bytes = 0
        self.max_symbolic_param_values = max_symbolic_param_values

    def _eval_dim(self, dim: onnx.TensorShapeProto.Dimension) -> int:
        dim_dict = MessageToDict(dim)
        if "dimValue" in dim_dict:
            value = Expression(
                dim_dict["dimValue"], self.max_symbolic_param_values.keys()
            ).evaluate(self.max_symbolic_param_values)
        elif "dimParam" in dim_dict:
            value = Expression(
                dim_dict["dimParam"], self.max_symbolic_param_values.keys()
            ).evaluate(self.max_symbolic_param_values)
        else:
            raise ValueError(f"Dimension is neither a constant nor expression")

        truncated = int(value)
        if truncated != value:
            raise ValueError(f"Expression returned non-integer result {value}")

        return truncated

    def estimate_single_tensor(self, tensor_name: str, tensor: onnx.TensorProto) -> TensorInfo:
        tensor_info = MutableTensorInfo(
            name=tensor_name,
            lifetime_begin=0,
            lifetime_end=self.num_nodes
        )

        dims = [self._eval_dim(dim) for dim in tensor.shape.dim]
        cols, rows = self.memory.to_matrix_dims(dims)
        
        tensor_info.set_estimated_parameters(
            cols=cols,
            rows=rows,
            size=self.memory.get_matrix_size(cols, rows, onnx.TensorProto.DataType.Name(tensor.elem_type))
        )

        self.total_allocated_bytes += tensor_info.estimated_size
        return tensor_info


def estimate_mutable_tensors_naive(
    model: onnx.ModelProto,
    max_symbolic_param_values: dict[str, int],
    memory: DeviceMemory,
    logs_enabled: False,
) -> dict[str, TensorInfo]:
    tensors: List[MutableTensorInfo] = list()

    estimator = NaiveTensorMemoryEstimator(model, max_symbolic_param_values, memory)

    # for input in model.graph.input:
    #     tensors.append(estimator.estimate_single_tensor(input.name, input.type.tensor_type))

    for value in model.graph.value_info:
        tensors.append(estimator.estimate_single_tensor(value.name, value.type.tensor_type))

    return estimator.total_allocated_bytes, tensors
