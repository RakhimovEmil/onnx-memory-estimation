from typing import List


class NodeWrapper:
    def __init__(self, name: str, type: str, input_names: List[str], output_names: List[str]):
        self.name = name
        self.type = type
        self.input_names = input_names
        self.output_names = output_names
        # add runner for the initialized node
