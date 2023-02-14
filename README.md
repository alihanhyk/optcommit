# When to Make and Break Commitments?
Code author: Alihan Hüyük ([ah2075@cam.ac.uk](mailto:ah2075@cam.ac.uk))

This repository is for replicating the main experimental results in the ICLR 2023 paper ''when to make and break commitments?'' The method we propose, *Bayes-OCP* is implemented as function `policy_bayesocp` in `src/algs.py`.

### Usage

First, install the required python packages by running:
```shell
    python -m pip install -r requirements.txt
```

Then, the main results presented in Table 3 can be reproduced by running:
```shell
    python src/main.py
    python src/eval.py > res.txt
```
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
