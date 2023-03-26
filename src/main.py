# pylint: disable=redefined-outer-name

import dill
import jax
import jax.numpy as np
import tqdm
from jax.scipy.stats import norm
from jax.tree_util import Partial

from algs import (
    policy_adaptiveenrichment,
    policy_bayesocp,
    policy_futilitystopping,
    policy_greedybayesocp,
    policy_rct,
)

jax.config.update("jax_platform_name", "cpu")

hyper = dict()
hyper["horizon"] = 1200

hyper["n"] = 2
hyper["xsets"] = np.array([[1, 0], [0, 1]])
hyper["xset_nil"] = np.zeros(hyper["n"], dtype=int)
hyper["xset_all"] = np.ones(hyper["n"], dtype=int)

hyper["envr_prop"] = np.ones(hyper["n"]) / hyper["n"]
hyper["envr_costf"] = Partial(lambda xset: 1 / np.sum(hyper["envr_prop"], where=xset) ** 0.1)
hyper["envr_rwrdf"] = Partial(lambda xset: 1000 * np.sum(hyper["envr_prop"], where=xset) ** 0.1)

###

hyper["test0"] = dict(size=0, mean=0.0, stat=-1)
hyper["test_horizon"] = 600
hyper["test_threshold"] = norm.ppf(0.95)


@jax.jit
def test_update(test, y):
    test["size"] = test["size"] + np.where(test["stat"] < 0, 1, 0)
    test["mean"] = test["mean"] + np.where(test["stat"] < 0, (y - test["mean"]) / test["size"], 0.0)
    test["stat"] = np.where(
        test["size"] == hyper["test_horizon"],
        test["mean"] / np.sqrt(1 / test["size"]) > hyper["test_threshold"],
        -1,
    )
    return test


@jax.jit
def test_predict(test, envr_mean):
    _mean = (hyper["test_threshold"] * np.sqrt(hyper["test_horizon"]) - test["mean"] * test["size"]) / (
        hyper["test_horizon"] - test["size"]
    )
    _norm = (_mean - envr_mean) / np.sqrt(1 / (hyper["test_horizon"] - test["size"]))
    return np.where(test["stat"] < 0, 1 - norm.cdf(_norm), test["stat"])


hyper["test_predict"] = Partial(test_predict)

###

hyper["post0"] = dict(size=np.ones(hyper["n"], dtype=int) * 10, mean=np.ones(hyper["n"]) * 0.1)


@jax.jit
def post_update(post, x, y):
    post["size"] = post["size"].at[x].add(1)
    post["mean"] = post["mean"].at[x].add((y - post["mean"][x]) / post["size"][x])
    return post


###


def _simulate(arg0, arg1, policy, envr_mean):
    (xset, test, post), key = arg0, arg1

    _prop = np.where(xset, hyper["envr_prop"], np.zeros(hyper["envr_prop"].size))
    _prop = _prop / np.sum(_prop)
    key, *subkeys = jax.random.split(key, 3)
    x = jax.random.choice(subkeys[0], np.arange(hyper["envr_prop"].size), p=_prop)
    y = envr_mean[x] + jax.random.normal(subkeys[1])

    post = jax.lax.cond(np.any(xset), lambda post: post_update(post, x, y), lambda post: post, post)
    test = jax.lax.cond(np.any(xset), lambda test: test_update(test, y), lambda test: test, test)

    key, subkey = jax.random.split(key)
    _xset, info = policy(xset, test, post, hyper, subkey)
    _xset = np.where(np.any(xset), _xset, xset)

    _xset = np.where(test["stat"] == 1, hyper["xset_nil"], _xset)
    _xset = np.where(
        np.logical_and(test["stat"] == 0, np.all(_xset == xset)),
        hyper["xset_nil"],
        _xset,
    )

    _test = jax.lax.cond(np.any(_xset != xset), lambda test: hyper["test0"], lambda test: test, test)
    return (_xset, _test, post), (xset, test, post, info)


@jax.jit
def simulate(key, policy, envr_mean):
    _, (xsets, tests, posts, infos) = jax.lax.scan(
        Partial(_simulate, policy=policy, envr_mean=envr_mean),
        (hyper["xset_all"], hyper["test0"], hyper["post0"]),
        jax.random.split(key, hyper["horizon"]),
    )
    return xsets, tests, posts, infos


###

hyper["policy_threshold"] = 0.80

key = jax.random.PRNGKey(0)
envrs_mean = list()

for k in tqdm.trange(5000):
    key, subkey = jax.random.split(key)
    envr_mean = hyper["post0"]["mean"]
    envr_mean = envr_mean + np.sqrt(1 / hyper["post0"]["size"]) * jax.random.normal(subkey, shape=(hyper["n"],))
    envrs_mean.append(envr_mean)

    key, subkey = jax.random.split(key)

    results = simulate(subkey, Partial(policy_rct), envr_mean)
    with open(f"res/rct-k{k}.obj", "wb") as f:
        dill.dump(results, f)

    results = simulate(subkey, Partial(policy_adaptiveenrichment), envr_mean)
    with open(f"res/adaptiveenrichment-k{k}.obj", "wb") as f:
        dill.dump(results, f)

    results = simulate(subkey, Partial(policy_futilitystopping), envr_mean)
    with open(f"res/futilitystopping-k{k}.obj", "wb") as f:
        dill.dump(results, f)

    results = simulate(subkey, Partial(policy_greedybayesocp), envr_mean)
    with open(f"res/greedybayesocp-k{k}.obj", "wb") as f:
        dill.dump(results, f)

    results = simulate(subkey, Partial(policy_bayesocp), envr_mean)
    with open(f"res/bayesocp-k{k}.obj", "wb") as f:
        dill.dump(results, f)

with open("res/envrs.obj", "wb") as f:
    dill.dump((envrs_mean, hyper), f)
