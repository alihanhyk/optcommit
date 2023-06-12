# pylint: disable=unused-argument
# pyright: reportGeneralTypeIssues=false

from functools import partial

import jax
import jax.numpy as np


@jax.jit
def policy_rct(xset, test, post, hyper, key):
    return xset, None


@jax.jit
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def _policy_utility(key, xset, test, post, hyper):
    _prop = np.where(xset, hyper["envr_prop"], np.zeros(hyper["envr_prop"].size))
    _prop = _prop / np.sum(_prop)
    _mean = post["mean"] + np.sqrt(1 / post["size"]) * jax.random.normal(key, shape=(post["mean"].size,))
    _mean = np.dot(_prop, _mean)
    util = hyper["envr_rwrdf"](xset) * hyper["test_predict"](test, _mean)
    util = util - hyper["envr_costf"](xset) * (hyper["test_horizon"] - test["size"])
    return util


@jax.jit
def _policy_search(key, xset, test, post, hyper):
    best_xset = hyper["xset_nil"]
    best_utils = np.zeros(1000)

    for _xset in hyper["xsets"]:
        _utils = _policy_utility(jax.random.split(key, 1000), _xset, hyper["test0"], post, hyper)

        cond = np.logical_and(np.all(_xset <= xset), np.any(_xset != xset))
        best_xset = np.where(np.logical_and(cond, _utils.mean() > best_utils.mean()), _xset, best_xset)
        best_utils = np.where(np.logical_and(cond, _utils.mean() > best_utils.mean()), _utils, best_utils)

    return best_xset, best_utils


@jax.jit
def policy_adaptiveenrichment(xset, test, post, hyper, key):
    _xset, _utils = _policy_search(key, xset, test, post, hyper)
    utils = _policy_utility(jax.random.split(key, 1000), xset, test, post, hyper)

    cond = np.logical_and(_utils.mean() > utils.mean(), test["size"] == hyper["test_horizon"] / 2)
    return np.where(cond, _xset, xset), None


@jax.jit
def policy_bayesocp(xset, test, post, hyper, key):
    _xset, _utils = _policy_search(key, xset, test, post, hyper)
    utils = _policy_utility(jax.random.split(key, 1000), xset, test, post, hyper)

    cond = np.mean(_utils > utils) >= hyper["policy_threshold"]
    return np.where(cond, _xset, xset), None


@jax.jit
def policy_greedybayesocp(xset, test, post, hyper, key):
    _xset, _utils = _policy_search(key, xset, test, post, hyper)
    utils = _policy_utility(jax.random.split(key, 1000), xset, test, post, hyper)

    cond = _utils.mean() > utils.mean()
    return np.where(cond, _xset, xset), None


@jax.jit
def policy_futilitystopping(xset, test, post, hyper, key):
    _xset, _utils = hyper["xset_nil"], np.zeros(1000)
    utils = _policy_utility(jax.random.split(key, 1000), xset, test, post, hyper)

    cond = np.mean(_utils > utils) >= hyper["policy_threshold"]
    return np.where(cond, _xset, xset), None
