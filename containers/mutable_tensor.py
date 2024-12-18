from __future__ import annotations

from typing import cast, List, Optional

from expression import Expression

class MutableTensor:
    def __init__(self, dim_exprs: List[Expression], type: str):
        self.dim_exprs = dim_exprs
        self.type = type

class MutableTensorInfo:
    def __init__(self, name: str, lifetime_begin: int, lifetime_end: int):
        self.tensor_name = name
        self.lifetime_begin = lifetime_begin
        self.lifetime_end = lifetime_end

        self.__estimated_size: Optional[int] = None
        self.__cols: Optional[int] = None
        self.__rows: Optional[int] = None
        self.__estimated = False


    def set_estimated_parameters(self, cols: int, rows: int, size: int):
        self.__cols = cols
        self.__rows = rows
        self.__estimated_size = size
        self.__estimated = True


    @property
    def cols(self) -> int:
        self.__check_estimated()
        return cast(int, self.__cols)


    @property
    def rows(self) -> int:
        self.__check_estimated()
        return cast(int, self.__rows)


    @property
    def estimated_size(self) -> int:
        self.__check_estimated()
        return cast(int, self.__estimated_size)


    def __lt__(self, other: MutableTensorInfo):
        return (other.estimated_size, self.lifetime_begin, self.lifetime_end) < (self.estimated_size, other.lifetime_begin, other.lifetime_end)


    def __check_estimated(self):
        if not self.__estimated:
            raise RuntimeError(f'Tried to get a property of an unestimated tensor {self.tensor_name}')
