import numpy as np
import torch

from procgen_adventure.utils.torch_utils import zeros


def get_values_from_list_dict(epinfobuf, key):
    list_values = [epinfo[key] for epinfo in epinfobuf if key in epinfo]
    return list_values


def discount_return_n_step(
    reward,
    done,
    n_step,
    discount,
    return_dest=None,
    done_n_dest=None,
    do_truncated=False,
):
    """Time-major inputs, optional other dimension: [T], [T,B], etc.  Computes
    n-step discounted returns within the timeframe of the of given rewards. If
    `do_truncated==False`, then only compute at time-steps with full n-step
    future rewards are provided (i.e. not at last n-steps--output shape will
    change!).  Returns n-step returns as well as n-step done signals, which is
    True if `done=True` at any future time before the n-step target bootstrap
    would apply (bootstrap in the algo, not here)."""
    rlen = reward.shape[0]
    if not do_truncated:
        rlen -= n_step - 1
    return_ = (
        return_dest
        if return_dest is not None
        else zeros((rlen,) + reward.shape[1:], dtype=reward.dtype)
    )
    done_n = (
        done_n_dest
        if done_n_dest is not None
        else zeros((rlen,) + reward.shape[1:], dtype=done.dtype)
    )
    return_[:] = reward[:rlen]  # 1-step return is current reward.
    done_n[:] = done[:rlen]  # True at time t if done any time by t + n - 1
    is_torch = isinstance(done, torch.Tensor)
    if is_torch:
        done_dtype = done.dtype
        done_n = done_n.type(reward.dtype)
        done = done.type(reward.dtype)
    if n_step > 1:
        if do_truncated:
            for n in range(1, n_step):
                return_[:-n] += (
                    (discount ** n) * reward[n : n + rlen] * (1 - done_n[:-n])
                )
                done_n[:-n] = np.maximum(done_n[:-n], done[n : n + rlen])
        else:
            for n in range(1, n_step):
                return_ += (discount ** n) * reward[n : n + rlen] * (1 - done_n)
                done_n = np.maximum(done_n, done[n : n + rlen])  # Supports tensors.
    if is_torch:
        done_n = done_n.type(done_dtype)
    return return_, done_n
