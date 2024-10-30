from __future__ import annotations
from typing import cast, Dict, Iterable, List, Optional, Set, Tuple, Union
from time import time

from google.protobuf.json_format import MessageToDict
import onnx
import pulp
import pulp.constants as const

from containers import (
    ConstTensor,
    ConstTensorInfo,
    # DeviceMemory,
    # Expression,
    MutableTensor,
    MutableTensorInfo,
    NodeWrapper
)

from expression import Expression
from memory import DeviceMemory

from graph.utils import (
    evaluate_tensor_dimensions,
    load_const_tensors,
    load_nodes,
    load_values
)
from containers.utils import (
    to_matrix_dims
)

ValueType = Union[Expression, List[Expression], ConstTensor, MutableTensor]


class Graph:
    class GraphData:
        def __init__(self):
            self.input_names: Set[str] = set()
            self.output_names: Set[str] = set()
            self.values: Dict[str, ValueType] = dict()
            self.nodes: List[NodeWrapper] = list()


    def __init__(self, model: onnx.ModelProto, model_memory: DeviceMemory, placeholder_names: Set[str]): # reader = ?
        self.storage = Graph.GraphData()
        self.model_memory = model_memory

        self.storage.values, self.storage.nodes = load_nodes(model, placeholder_names)
        self.storage.values.update(load_const_tensors(model))
        self.storage.values.update(load_values(model, placeholder_names))

        if '' not in self.storage.values:
            self.storage.values[''] = ConstTensor([0, 1])

        self.storage.input_names = list(map(lambda x: x.name, model.graph.input))
        self.storage.output_names = list(map(lambda x: x.name, model.graph.output))


    def get_last_input_user_index(self):
        last_input_user: Dict[str, int] = {
            name: idx
            for idx, node in enumerate(self.storage.nodes)
            for name in node.input_names
        }
        last_input_user.update({name: len(self.storage.nodes) for name in self.storage.output_names})
        last_input_user.update({name: len(self.storage.nodes) for name in self.storage.input_names})
        return last_input_user
    

    def collect_inplace_mappings(self, last_input_user: Dict[str, int], max_batch_params: Dict[str, int], external_tensors: Set[str]):
        return dict()


    def estimate_single_tensor(self, info: MutableTensorInfo, max_batch_params: Dict[str, int]):
        tensor = self.storage.values[info.tensor_name]
        if isinstance(tensor, MutableTensor): # can be Expression, List[Expression], ConstTensor, MutableTensor
            dims = evaluate_tensor_dimensions(tensor.dim_exprs, max_batch_params)
            align_rows = False # info.tensor_name in self.storage.contiguous_tensor_names
            cols, rows = to_matrix_dims(dims)

            size = self.model_memory.get_matrix_size(cols, rows, tensor.type, align_rows)
            info.set_estimated_parameters(cols, rows, size)


    def find_optimal_strategy(self, max_batch_params: Dict[str, int], debug_model_opt: bool = False, time_limit: int = 1800) -> List[MutableTensorInfo]:

        def get_mutable_tensor_names(tensor_names: Iterable[str]) -> Set[str]:
            return {
                tensor_name for tensor_name in tensor_names
                if isinstance(self.storage.values[tensor_name], MutableTensor)
            }

        start_time = time()
        step_limit = len(self.storage.nodes) + 1

        mutable_tensor_names: Set[str] = set()
        tensor_name_to_birth_node: Dict[str, NodeWrapper] = dict()

        input_names, output_names = set(), set()

        for node in self.storage.nodes:
            mutable_input_names = get_mutable_tensor_names(node.input_names)
            mutable_tensor_names.update(mutable_input_names)

            # every single node outputs ONLY mutable tensors
            mutable_tensor_names.update(node.output_names)

            for out_name in node.output_names:
                tensor_name_to_birth_node[out_name] = node

            input_names.update(node.input_names)
            output_names.update(node.output_names)

        print(f'Mutable tensors: {len(mutable_tensor_names)}')

        tensor_names_to_tensors: Dict[str, MutableTensorInfo] = dict()
        for tensor_name in mutable_tensor_names:
            tensor_names_to_tensors[tensor_name] = MutableTensorInfo(tensor_name, 0, step_limit) # lifetime is to be determined
            self.estimate_single_tensor(tensor_names_to_tensors[tensor_name], max_batch_params)

        problem = pulp.LpProblem(name='model_strategy', sense=pulp.LpMinimize)
        cvars: Dict[str, List[pulp.LpVariable]] = dict()
        pvars: Dict[str, List[pulp.LpVariable]] = dict()
        for tensor_name in mutable_tensor_names:
            cvars[tensor_name] = list()
            pvars[tensor_name] = list()
            for ts in range(0, step_limit):
                cvars[tensor_name].append(pulp.LpVariable(name=f'create_{tensor_name}_at_{ts}', cat=const.LpBinary))

                # hack for no preservations at the first step without prior creation
                pvars[tensor_name].append(
                    pulp.LpVariable(name=f'preserve_{tensor_name}_at_{ts}', cat=const.LpBinary) 
                    if ts > 0 else
                    pulp.LpVariable(name=f'preserve_{tensor_name}_at_{ts}', lowBound=0, upBound=0, cat=const.LpInteger) 
                )

        # Validation constraints
        last_cvar, last_pvar = None, None
        for tensor_name in mutable_tensor_names:
            for ts in range(0, step_limit):
                cur_cvar = cvars[tensor_name][ts]
                cur_pvar = pvars[tensor_name][ts]

                # Constraint (1)
                problem += cur_cvar + cur_pvar <= 1

                # Constraint (2)
                if last_cvar is not None and last_pvar is not None:
                    problem += cur_pvar <= last_pvar + last_cvar

                last_cvar = cur_cvar
                last_pvar = cur_pvar

                # check whether it is a model input (should be equivalent to checking presence in the corresponding set)
                birth_node = tensor_name_to_birth_node.get(tensor_name, None)
                if birth_node is None:
                    continue

                # Constraint (4)
                for in_name in get_mutable_tensor_names(birth_node.input_names):
                    problem += cur_cvar <= pvars[in_name][ts]
                
                # Constraint (5)
                for out_name in birth_node.output_names:
                    problem += cur_cvar == cvars[out_name][ts]

            # Constraint (3)
            problem += pulp.lpSum(cvars[tensor_name][ts] for ts in range(0, step_limit)) == 1

        # Location constraints
        max_memory_bound = sum(tensor.estimated_size for tensor in tensor_names_to_tensors.values())
        memory_bound = pulp.LpVariable(name='peak_mem', lowBound=0, upBound=max_memory_bound, cat=const.LpInteger)
        offvars: Dict[str, pulp.LpVariable] = dict()

        for tensor_name in mutable_tensor_names:
            offvars[tensor_name] = pulp.LpVariable(name=f'offset_{tensor_name}', lowBound=0, upBound=max_memory_bound, cat=const.LpInteger)

            # Constraint (8)
            problem += offvars[tensor_name] + tensor_names_to_tensors[tensor_name].estimated_size <= memory_bound

        avars: Dict[Tuple[str, str], pulp.LpVariable] = dict()
        bvars: Dict[Tuple[str, str], pulp.LpVariable] = dict()

        for tensor_name_i in mutable_tensor_names:
            for tensor_name_j in mutable_tensor_names:
                if tensor_name_i == tensor_name_j:
                    continue

                avar = pulp.LpVariable(name=f'a_{tensor_name_i}_{tensor_name_j}', cat=const.LpBinary)
                bvar = pulp.LpVariable(name=f'b_{tensor_name_i}_{tensor_name_j}', cat=const.LpBinary)
                avars[tensor_name_i, tensor_name_j] = avar
                avars[tensor_name_j, tensor_name_i] = avar
                bvars[tensor_name_i, tensor_name_j] = bvar
                bvars[tensor_name_j, tensor_name_i] = bvar

                # Constraint (6)
                problem += avar + bvar <= 1
                for ts in range(0, step_limit):
                    problem += avar + bvar >= cvars[tensor_name_i][ts] + pvars[tensor_name_i][ts] + cvars[tensor_name_j][ts] + pvars[tensor_name_j][ts] - 1
                
                # Constraint (7)
                problem += offvars[tensor_name_i] + tensor_names_to_tensors[tensor_name_i].estimated_size - offvars[tensor_name_j] <= (1 - avar) * max_memory_bound
                problem += offvars[tensor_name_i] - tensor_names_to_tensors[tensor_name_j].estimated_size - offvars[tensor_name_j] >= (bvar - 1) * max_memory_bound

        problem += memory_bound
        problem.solve(pulp.PULP_CBC_CMD(msg=debug_model_opt, timeLimit=time_limit))

        end_time = time()
        if debug_model_opt:
            print(f'MODeL estimated memory: {int(cast(float, pulp.value(memory_bound)))} bytes')
            print(f'Time taken: {end_time - start_time:.3f} seconds')
            print()

        for tensor_name in mutable_tensor_names:
            lifetime_begin = [cast(float, cvar.value()) for cvar in cvars[tensor_name]].index(1)
            lifetime_end = step_limit - 1 - [cast(float, cvar.value()) + cast(float, pvar.value()) for cvar, pvar in zip(cvars[tensor_name], pvars[tensor_name])][lifetime_begin:][::-1].index(1)

            tensor_names_to_tensors[tensor_name].lifetime_begin = lifetime_begin
            tensor_names_to_tensors[tensor_name].lifetime_end = lifetime_end
        
        if debug_model_opt:
            tensor_list: List[str] = list()
            tensor_set: Set[str] = set()

            print('Final schedule:')
            for node in self.storage.nodes:
                mutable_input_names = get_mutable_tensor_names(node.input_names)
                for tensor_name in list(mutable_input_names) + node.output_names:
                    if tensor_name not in tensor_set:
                        tensor_list.append(tensor_name)
                        tensor_set.add(tensor_name)

                print(
                    f'''{node.name}: {", ".join(f"{tensor_names_to_tensors[input_name].lifetime_begin}-{tensor_names_to_tensors[input_name].lifetime_end}" for input_name in mutable_input_names)} -> '''
                    f'''{", ".join(f"{tensor_names_to_tensors[output_name].lifetime_begin}-{tensor_names_to_tensors[output_name].lifetime_end}" for output_name in node.output_names)}'''
                )
            print()
                
            print('Tensor placements:')
            for tensor_name in tensor_list:
                print(
                    f'{tensor_name}: from {tensor_names_to_tensors[tensor_name].lifetime_begin} to {tensor_names_to_tensors[tensor_name].lifetime_end} @ '
                    f'offset {cast(float, offvars[tensor_name].value())} + {tensor_names_to_tensors[tensor_name].estimated_size}'
                )
            print()

        return list(tensor_names_to_tensors.values())

    # @staticmethod
    # def can_be_in_place(input: MutableTensor, output: MutableTensor, max_batch_params: Dict[str, int]) -> bool:
    #     if input.type != output.type:
    #         return False

    #     # maybe should find another place for this static method
    #     input_size = ConstTensorInfo.to_matrix_dims(evaluate_tensor_dimensions(input.dim_exprs, max_batch_params))
    #     output_size = ConstTensorInfo.to_matrix_dims(evaluate_tensor_dimensions(output.dim_exprs, max_batch_params))
    #     return input_size == output_size
