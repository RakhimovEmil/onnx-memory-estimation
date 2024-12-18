from __future__ import annotations
from typing import cast, Dict, Iterable, List, Optional, Set, Tuple, Union
from time import time

from google.protobuf.json_format import MessageToDict
import onnx
import pulp
import pulp.constants as const

from containers import (
    ConstTensor,
    MutableTensor,
    MutableTensorInfo,
    NodeWrapper
)

from expression import Expression
from memory import DeviceMemory

from graph.utils import (
    evaluate_tensor_dimensions,
    get_ancestors_max_asap,
    get_nodes_alap,
    load_const_tensors,
    load_nodes,
    load_values
)
from containers.utils import (
    to_matrix_dims
)

ValueType = Union[Expression, List[Expression], ConstTensor, MutableTensor]


def _get_mutable_tensor_names(values: Dict[str, ValueType], tensor_names: Iterable[str]) -> Set[str]:
    return {
        tensor_name for tensor_name in tensor_names
        if isinstance(values[tensor_name], MutableTensor)
    }


def _find_graph_blocks(nodes: List[NodeWrapper], edges: Dict[str, Tuple[Optional[str], List[str]]], values: Dict[str, ValueType], capacity: int = 10) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    start, block_started = 0, False
    block_outputs_count = 0
    
    for i, node in enumerate(nodes):
        node_inputs_count = len(_get_mutable_tensor_names(values, node.input_names))
        node_outputs_count = len(edges[node.output_names[0]][1])

        if not block_started:
            if node_inputs_count == 1 and node_outputs_count == 1 and (i - start) >= capacity:
                blocks.append((start, i))
                start = i

            if node_outputs_count > 1:
                if start != i:
                    blocks.append((start, i))

                block_outputs_count = node_outputs_count
                block_started = True
                start = i
        
        elif block_started:
            block_outputs_count -= node_inputs_count

            if block_outputs_count == 0:
                blocks.append((start, i))
                block_started = node_outputs_count > 1
                start = i

            block_outputs_count += node_outputs_count
        
    if start != i:
        blocks.append((start, i))

    return blocks


def _fix_lifetime(tensors: List[MutableTensorInfo], new_tensors: List[MutableTensorInfo], cur_lifetime_max: int) -> List[MutableTensorInfo]:
    if not tensors:
        cur_lifetime_max = max([tensor.lifetime_end for tensor in new_tensors])
        return new_tensors, cur_lifetime_max
    
    result = {tensor.tensor_name: tensor for tensor in tensors}
    name_set = set(result.keys())
    offset = cur_lifetime_max - 1

    for tensor in new_tensors:
        if tensor.tensor_name in name_set:
            continue

        tensor.lifetime_begin += offset
        tensor.lifetime_end += offset
        cur_lifetime_max = max(cur_lifetime_max, tensor.lifetime_end)

        result[tensor.tensor_name] = tensor

    return list(result.values()), cur_lifetime_max


class Graph:
    class GraphData:
        def __init__(self):
            self.input_names: Set[str] = set()
            self.output_names: Set[str] = set()
            self.values: Dict[str, ValueType] = dict()
            self.nodes: List[NodeWrapper] = list()
            self.edges: Dict[str, Tuple[Optional[str], List[str]]] = dict()


    def __init__(self, model: onnx.ModelProto, model_memory: DeviceMemory, placeholder_names: Set[str]): # reader = ?
        self.storage = Graph.GraphData()
        self.model_memory = model_memory

        self.storage.nodes, self.storage.edges = load_nodes(model, placeholder_names)
        self.storage.values = load_const_tensors(model)
        self.storage.values.update(load_values(model, placeholder_names))

        if '' not in self.storage.values:
            self.storage.values[''] = ConstTensor([0, 1])

        self.storage.input_names = list(map(lambda x: x.name, model.graph.input))
        self.storage.output_names = list(map(lambda x: x.name, model.graph.output))


    def get_tensor_to_lifetime_end(self):
        tensor_to_lifetime_end: Dict[str, int] = {
            name: idx
            for idx, node in enumerate(self.storage.nodes)
            for name in node.input_names
        }
        tensor_to_lifetime_end.update({name: len(self.storage.nodes) for name in self.storage.output_names})
        tensor_to_lifetime_end.update({name: len(self.storage.nodes) for name in self.storage.input_names})
        return tensor_to_lifetime_end


    def estimate_single_tensor(self, info: MutableTensorInfo, max_symbolic_var_params: Dict[str, int]):
        tensor = self.storage.values[info.tensor_name]
        if isinstance(tensor, MutableTensor): # can be Expression, List[Expression], ConstTensor, MutableTensor
            dims = evaluate_tensor_dimensions(tensor.dim_exprs, max_symbolic_var_params)
            align_rows = False # info.tensor_name in self.storage.contiguous_tensor_names
            cols, rows = to_matrix_dims(dims)

            size = self.model_memory.get_matrix_size(cols, rows, tensor.type, align_rows)
            info.set_estimated_parameters(cols, rows, size)


    def _find_optional_strategy(self, ind_start: int, ind_stop: int, max_symbolic_var_params: Dict[str, int], logs_enabled: bool = False, time_limit: int = 18000) -> List[MutableTensorInfo]:
        nodes = self.storage.nodes[ind_start : ind_stop + 1]

        start_time = time()
        step_limit = len(nodes) + 1

        mutable_tensor_names: Set[str] = set()
        tensor_name_to_birth_node: Dict[str, NodeWrapper] = dict()

        input_names, output_names = set(), set()
        asap, alap = dict(), dict()
        max_asap = 0

        for node in nodes:
            mutable_input_names = _get_mutable_tensor_names(self.storage.values, node.input_names)
            mutable_tensor_names.update(mutable_input_names)

            # every single node outputs ONLY mutable tensors
            if node != nodes[-1]:
                mutable_tensor_names.update(node.output_names)

            for out_name in node.output_names:
                tensor_name_to_birth_node[out_name] = node

            input_names.update(node.input_names)
            output_names.update(node.output_names)

            asap[node.name] = get_ancestors_max_asap(node, self.storage.edges, asap) + 1
            max_asap = max(max_asap, asap[node.name])

        alap = get_nodes_alap(nodes, self.storage.edges, max_asap)

        if logs_enabled:
            print(f'Mutable tensors: {len(mutable_tensor_names)}')

        tensor_names_to_tensors: Dict[str, MutableTensorInfo] = dict()
        for tensor_name in mutable_tensor_names:
            tensor_names_to_tensors[tensor_name] = MutableTensorInfo(tensor_name, 0, step_limit) # lifetime is to be determined
            self.estimate_single_tensor(tensor_names_to_tensors[tensor_name], max_symbolic_var_params)

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

        max_peak_mem = sum(tensor.estimated_size for tensor in tensor_names_to_tensors.values())
        peak_mem = pulp.LpVariable(name='peak_mem', lowBound=0, upBound=max_peak_mem, cat=const.LpInteger)
        peak_mem_no_frag = pulp.LpVariable(name='peak_mem_no_frag', lowBound=0, upBound=max_peak_mem, cat=const.LpInteger)

        # Validation constraints
        last_cvar, last_pvar = None, None
        for tensor_name in mutable_tensor_names:
            for ts in range(0, step_limit):
                cur_cvar = cvars[tensor_name][ts]
                cur_pvar = pvars[tensor_name][ts]
                
                src  = self.storage.edges[tensor_name][0]
                snks = self.storage.edges[tensor_name][1]

                # Constraint 1
                problem += cur_cvar + cur_pvar <= 1

                # Constraint 2
                if last_cvar is not None and last_pvar is not None:
                    problem += cur_pvar <= last_pvar + last_cvar

                last_cvar = cur_cvar
                last_pvar = cur_pvar

                # check whether it is a model input (should be equivalent to checking presence in the corresponding set)
                birth_node = tensor_name_to_birth_node.get(tensor_name, None)
                if birth_node is None:
                    continue

                # Constraint 4
                for in_name in _get_mutable_tensor_names(self.storage.values, birth_node.input_names):
                    problem += cur_cvar <= pvars[in_name][ts]
                
                # Constraint 5
                for out_name in birth_node.output_names:
                    problem += cur_cvar == cvars[out_name][ts]

                # Constraint 10
                if ts < asap[src] or ts > asap[src]:
                    problem += cur_cvar == 0
                
                # Constraint 11
                max_alap = max((alap[snk] for snk in snks), default=step_limit)
                if ts < asap[src] or ts > max_alap:
                    problem += cur_pvar == 0
                
                # Constraint 12
                max_asap = max((asap[snk] for snk in snks), default=step_limit)
                if ts >= alap[src] + 1 and ts <= max_asap:
                    problem += cur_pvar == 1 

            # Constraint 3
            problem += pulp.lpSum(cvars[tensor_name][ts] for ts in range(0, step_limit)) == 1

            # Constraint 13
            problem += pulp.lpSum(
                (cvars[tensor_name][ts] + pvars[tensor_name][ts]) *
                tensor_names_to_tensors[tensor_name].estimated_size
                for ts in range(0, step_limit)
            ) <= peak_mem_no_frag

        # Constraint 14 (operations cannot be calculated concurrently)
        for ts in range(0, step_limit):
            problem += pulp.lpSum(cvars[tensor_name][ts] for tensor_name in mutable_tensor_names) >= 0
            problem += pulp.lpSum(cvars[tensor_name][ts] for tensor_name in mutable_tensor_names) <= 1
        # for ts in range(step_limit):
        #     for i, tensor_name_1 in enumerate(mutable_tensor_names):
        #         for tensor_name_2 in mutable_tensor_names:
        #             problem += cvars[tensor_name_1][ts] + cvars[tensor_name_2][ts] <= 1


        # Location constraints
        offvars: Dict[str, pulp.LpVariable] = dict()

        # Constraint 8
        for tensor_name in mutable_tensor_names:
            offvars[tensor_name] = pulp.LpVariable(name=f'offset_{tensor_name}', lowBound=0, upBound=max_peak_mem, cat=const.LpInteger)
            problem += offvars[tensor_name] + tensor_names_to_tensors[tensor_name].estimated_size <= peak_mem

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

                # Constraint 6: p1
                problem += avar + bvar <= 1
                for ts in range(0, step_limit):
                    # Constraint 6: p2
                    problem += avar + bvar >= (cvars[tensor_name_i][ts] + pvars[tensor_name_i][ts]) + (cvars[tensor_name_j][ts] + pvars[tensor_name_j][ts]) - 1
                
                # Constraint 7a
                problem += offvars[tensor_name_i] + tensor_names_to_tensors[tensor_name_i].estimated_size - offvars[tensor_name_j] <= (1 - avar) * max_peak_mem

                # Constraint 7b
                problem += offvars[tensor_name_i] - tensor_names_to_tensors[tensor_name_j].estimated_size - offvars[tensor_name_j] >= (bvar - 1) * max_peak_mem

        problem += peak_mem
        problem += peak_mem_no_frag
        problem.solve(pulp.PULP_CBC_CMD(msg=logs_enabled, timeLimit=time_limit))

        end_time = time()

        if logs_enabled:
            print(f'MODeL estimated memory: {pulp.value(peak_mem_no_frag)} bytes')
            print(f'Time taken: {end_time - start_time:.3f} seconds\n')

        for tensor_name in mutable_tensor_names:
            # breakpoint()
            lifetime_begin = [cast(float, cvar.value()) for cvar in cvars[tensor_name]].index(1)
            lifetime_end = step_limit - 1 - [cast(float, cvar.value()) + cast(float, pvar.value()) for cvar, pvar in zip(cvars[tensor_name], pvars[tensor_name])][lifetime_begin:][::-1].index(1)

            tensor_names_to_tensors[tensor_name].lifetime_begin = lifetime_begin
            tensor_names_to_tensors[tensor_name].lifetime_end = lifetime_end
        
        if logs_enabled:
            tensor_list: List[str] = list()
            tensor_set: Set[str] = set()

            print('Final schedule:')
            for node in nodes:
                mutable_input_names = _get_mutable_tensor_names(self.storage.values, node.input_names)
                for tensor_name in list(mutable_input_names) + node.output_names:
                    if tensor_name not in tensor_set:
                        tensor_list.append(tensor_name)
                        tensor_set.add(tensor_name)

                print(
                    f'{node.name}: {", ".join(f"{tensor_names_to_tensors[input_name].lifetime_begin}-{tensor_names_to_tensors[input_name].lifetime_end}" for input_name in mutable_input_names)} -> '
                    f'{", ".join(f"{tensor_names_to_tensors[output_name].lifetime_begin}-{tensor_names_to_tensors[output_name].lifetime_end}" for output_name in node.output_names)}\n'
                )
                
            print('Tensor placements:')
            for tensor_name in tensor_list:
                print(
                    f'{tensor_name}: from {tensor_names_to_tensors[tensor_name].lifetime_begin} to {tensor_names_to_tensors[tensor_name].lifetime_end} @ '
                    f'offset {cast(float, offvars[tensor_name].value())} + {tensor_names_to_tensors[tensor_name].estimated_size}\n'
                )

        return list(tensor_names_to_tensors.values())


    def find_optimal_strategy(self, max_symbolic_var_params: Dict[str, int], logs_enabled: bool = False, time_limit: int = 18000) -> List[MutableTensorInfo]:
        indexes: List[Tuple[int, int]] = _find_graph_blocks(self.storage.nodes, self.storage.edges, self.storage.values)
        tensors: List[MutableTensorInfo] = []
        cur_lifetime_max = 0

        for (l, r) in indexes:
            print('lol')
            new_tensors: List[MutableTensorInfo] = self._find_optional_strategy(l, r, max_symbolic_var_params, logs_enabled, time_limit)
            tensors, cur_lifetime_max = _fix_lifetime(tensors, new_tensors, cur_lifetime_max)
        return tensors
    
    def straight_find_optional_strategy(self, max_symbolic_var_params: Dict[str, int], logs_enabled: bool = False, time_limit: int = 18000) -> List[MutableTensorInfo]:
        return self._find_optional_strategy(0, len(self.storage.nodes) - 1, max_symbolic_var_params, logs_enabled, time_limit)
