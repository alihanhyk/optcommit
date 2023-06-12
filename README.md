# When to Make and Break Commitments?
Code author: Alihan Hüyük ([ah2075@cam.ac.uk](mailto:ah2075@cam.ac.uk))

This repository is for reproducing the main experimental results in the [ICLR'23 paper "When to make and break commitments?"](https://openreview.net/forum?id=q8vgHfPdoQP) The method we propose, *Bayes-OCP*, is implemented as the function `policy_bayesocp` in `src/algs.py`.

### Usage

First, clone the repository, then install the required python packages by running:
```shell
python -m pip install -r requirements.txt
```

Then, the main results presented in Table 3 can be reproduced by running:
```shell
python src/main.py
python src/eval.py > res.txt
```

### Notes
* Note there are empty cells in the results table you will see when running `python src/eval.py` for Oracle RCT. These results are determined by the problem setting directly (and they do not need to be computed numerically through simulations).
* Note that for some folds, Greedy Bayes OCP never succeeds (hence it is not possible to compute time-to-success and you may see a `nan` value in the `python src/eval.py` results table). The paper reports the average (and variance) for folds where Greedy Bayes OCP succeeded at least once.

### Contributing

If you would like to contribute to the code, please install and enable [`pre-commit`](https://pre-commit.com/)
before making any commits and submitting a PR:
```shell
python -m pip install pre-commit
pre-commit install
```
The checks `pre-commit` will run can be found in [`.pre-commit-config.yaml`](./.pre-commit-config.yaml). You will need to fix any problems identified by these checks before you can commit your changes.

### Citing

If you use this software, please cite as follows:
```
@inproceedings{huyuk2023when,
  author={Alihan H\"uy\"uk and Zhaozhi Qian and Mihaela van der Schaar},
  title={When to make and break commitments?},
  booktitle={Proceedings of the 11th International Conference on Learning Representations},
  year={2023}
}
```
