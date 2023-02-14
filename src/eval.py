import dill
import jax
import jax.numpy as np
from jax.scipy.stats import norm
from tabulate import tabulate
import tqdm

jax.config.update('jax_platform_name', 'cpu')

with open('res/envrs.obj', 'rb') as f:
    envrs_mean, hyper = dill.load(f)

freq = dict(green=0., amber=0., red=0.)
utility = dict(all=list(), green=list(), amber=list())
success = dict(all=list(), green=list(), amber=list())

envrs_type = list()
for envr_mean in envrs_mean:

    succ = dict(green=None, amber=None, red=0.)
    util = dict(green=None, amber=-np.inf, red=0.)

    _mean = np.dot(hyper['envr_prop'], envr_mean)
    succ['green'] = hyper['test_predict'](hyper['test0'], _mean)
    util['green'] = hyper['envr_rwrdf'](hyper['xset_all']) * succ['green']
    util['green'] += -hyper['envr_costf'](hyper['xset_all']) * hyper['test_horizon']

    for xset in hyper['xsets']:
        _prop = np.where(xset, hyper['envr_prop'], np.zeros(hyper['envr_prop'].size))
        _prop = _prop / np.sum(_prop)
        _mean = np.dot(_prop, envr_mean)
        _succ = hyper['test_predict'](hyper['test0'], _mean)
        _util = hyper['envr_rwrdf'](xset) * _succ - hyper['envr_costf'](xset) * hyper['test_horizon']
        succ['amber'] = _succ if _util > util['amber'] else succ['amber']
        util['amber'] = _util if _util > util['amber'] else util['amber']

    _type = max(util, key=util.get)
    envrs_type.append(_type)

    utility['all'].append(util[_type])
    success['all'].append(succ[_type])
    if _type != 'red':
        utility[_type].append(util[_type])
        success[_type].append(succ[_type])


freq = {_type: sum(1 if _type == envr_type else 0 for envr_type in envrs_type) / len(envrs_type) for _type in freq}
utility = {_type: np.array(utility[_type]).mean() for _type in utility}
success = {_type: np.array(success[_type]).mean() for _type in success}

###

fold_n = 5
fold_size = 1000

key_typs = ['all', 'green', 'amber', 'red']
key_algs = ['rct', 'adaptiveenrichment', 'futilitystopping', 'greedybayesocp', 'bayesocp']
key_mets = ['utility', 'fwer', 'switches', 'success', 'timetosuccess', 'timetofailure']

res = {(typ, alg, met): [list() for _ in range(fold_n)]
        for typ in key_typs for alg in key_algs for met in key_mets}

envr_costsf = jax.jit(jax.vmap(hyper['envr_costf']))


for k in tqdm.trange(fold_n * fold_size):
    i = k // fold_size

    for alg in key_algs:
        with open(f'res/{alg}-k{k}.obj', 'rb') as f:
            xsets, tests, posts, infos = dill.load(f)

        _time = np.argwhere(np.all(xsets == 0, axis=-1))
        _time = (_time[0,0] if _time.size > 0 else hyper['horizon']) - 1

        _util = hyper['envr_rwrdf'](xsets[_time]) if tests['stat'][_time] == 1 else 0.
        _util += -np.sum(envr_costsf(xsets), where=np.any(xsets, axis=-1))
            
        _mean = None
        if tests['stat'][_time] == 1:
            _prop = np.where(xsets[_time], hyper['envr_prop'], np.zeros(hyper['envr_prop'].size))
            _prop = _prop / np.sum(_prop)
            _mean = np.dot(_prop, envrs_mean[k])

        _uniq = np.unique(xsets, axis=0).shape[0]
        _uniq += -1 -(1 if np.any(tests['stat'] > -1) else 0)
            
        for typ in ['all', envrs_type[k]]:
            res[(typ,alg,'utility')][i].append(_util)
            res[(typ,alg,'fwer')][i].append(1 if _mean is not None and _mean < 0 else 0)
            res[(typ,alg,'switches')][i].append(_uniq)
            res[(typ,alg,'success')][i].append(1 if _util > 0 else 0)
            res[(typ,alg,'timetosuccess')][i].append(_time + 1 if _util > 0 else np.nan)
            res[(typ,alg,'timetofailure')][i].append(_time + 1 if _util <= 0 else np.nan)

res = {(typ, alg, met):
    np.array([np.nanmean(np.array(res[(typ, alg, met)][i])) for i in range(fold_n)])
    for typ in key_typs for alg in key_algs for met in key_mets}

res = {(typ, alg, met):
    f"{res[(typ,alg,met)].mean()*100:.1f}% ({res[(typ,alg,met)].std()*100:.1f}%)"
    if met == 'fwer' or met == 'success'
    else f"{res[(typ,alg,met)].mean():.1f} ({res[(typ,alg,met)].std():.1f})"
    for typ in key_typs for alg in key_algs for met in key_mets}

values = [['oraclerct',
    f'{utility["all"]:.1f}', '','', f'{success["all"]*100:.1f}%', '','', 
    f'{utility["green"]:.1f}', '','', f'{success["green"]*100:.1f}%', '','',
    f'{utility["amber"]:.1f}', '','', f'{success["amber"]*100:.1f}%', '','',
    '','','','','','']]
values += [[alg] + [res[typ, alg, met] for typ in key_typs for met in key_mets] for alg in key_algs]
headers = [f'{typ}{f" ({freq[typ]*100:.1f}%)" if typ != "all" else ""}\n{met}' if met == 'utility' else f'\n{met}'
    for typ in key_typs for met in key_mets]

print(tabulate(values, headers=headers))
