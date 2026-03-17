from cutlass import cute
from cutlass import const_expr


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


# TODO will this work on runtime shapes?
def select_and_combine_batch_dim(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    if const_expr(mode is not None):
        a = select(a, mode)
    if const_expr(cute.rank(a) > 2):
        a = cute.group_modes(a, 2, cute.rank(a))
    return a
