from typing import List, Union, Any, Dict, Sequence, Mapping, Callable

"""
typing 模块在python3.5以上的版本才可以使用
# 数据类型：int str list Dict

# Dict Tuple Sequece, Mapping
其中Sequence，表示的是序列，list、tuple、collections.deque、str、bytes的都属于序列
Mapping表示的映射类型，包含dict


Callable[[Arg1Type, Arg2Type], ReturnType]
 
 
其中[Arg1Type, Arg2Type]为传递的参数类型（也就是signature的类型）
    ReturnType为返回参数的类型
 
如果传递参数和返回参数均没有，直接写Callable即可

int、long、float：整型、长整形、浮点型
bool、str：布尔型、字符串类型
List、 Tuple、 Dict、 Set：列表、元组、字典、集合
Iterable、Iterator：可迭代类型、迭代器类型
Generator：生成器类型
"""


def simple_type1(a: int, b: Dict[str, str], d: List, c: str = ""):
    """

    :param a: a: int           表示a是一个int类型，无默认值
    :param b: Dict[str: str]   表示b是一个字典类型，且key和value均是str类型
    :param d: List             表示d是一个list类型，无默认值
    :param c: str = ""         表示b是一个str类型，默认值为 ""
    :return:
    """
    pass


def simple_type2(a: Sequence, c: Dict[str:str], d: List[int, List], b: Mapping[str: int]):
    pass


Vector = List[float]


def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]


# type checks, a list of floats qualifies as a vector
new_vector = scale(2.0, [1.0, 2.0, 3.0])


# 如果传递参数和返回参数均没有，直接写Callable即可
def feeder1(get_next_item: Callable[[], str]) -> None:
    pass


def feeder2(get_next_item:Callable[..., str]) -> None:
    pass


# 其中[Arg1Type, Arg2Type]为传递的参数类型（也就是signature的类型）
#    ReturnType为返回参数的类型
def async_query(on_success: Callable[[int], None],
                on_error: Callable[[int, Exception], None]) -> None:
    pass

# Union type; Union[X, Y] means either X or Y。也就是Union里面的参数，任选其一。
Union[Union[int, str], float] == Union[int, str, float]
Union[int] == int  # The constructor actually returns int
Union[int, str, int] == Union[int, str]
Union[int, str] == Union[str, int]