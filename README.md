# RLC4CLR: Reinforcement Learning Control for Critical Load Restoration

## Description

Due to the complexities stemming from the large policy search space, renewable uncertainty, and nonlinearity in a complex grid control problem, directly applying RL algorithms to train a satisfactory policy requires extensive tuning to be successful. To address this challenge, this repository provides users an example on using the curriculum learning (CL) technique to design a training curriculum involving a simpler steppingstone problem that guides the RL agent to learn to solve the original hard problem in a progressive and more effective manner. 

Specifically, the optimal grid control problem considered is the critical load restoration (CLR) problem after a distribution system is islanded due to a substation outage. As we provide a reinforcement learning control (RLC) solution to the CLR problem, the repo is named __RLC4CLR__.

Please refer to our [published paper](https://ieeexplore.ieee.org/abstract/document/9903581) and [preprint on arXiv](https://arxiv.org/abs/2203.04166) for more details.

## Installation

Prepare the environment

```
git clone https://github.com/NREL/rlc4clr.git
cd rlc4clr

conda env create -n rlc4clr python=3.10
conda activate rlc4clr
pip install -r requirements.txt

cd rlc4clr 
pip install -e .
```

Download renewable generation profiles and synthetic forecasts data from the [OEDI website](https://data.openei.org/submissions/5978), unzip the data file and place it under a desired folder. Configuring the path to the renewable data at [DEFAULT_CONFIG.py](rlc4clr/clr_envs/envs/DEFAULT_CONFIG.py#L29)

To test if the environment is properly installed, run the `explore_env.ipynb` under the `train` folder.


## Funding Acknowledgement

This work was authored by the [National Renewable Energy Laboratory (NREL)](https://www.nrel.gov), operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the U.S. Department of Energy __Office of Electricity (OE) Advanced Grid Modeling (AGM) Program__. 

## Citation

If citing this work, please use the following:

```bibtex
@article{zhang2022curriculum,
  title={Curriculum-based reinforcement learning for distribution system critical load restoration},
  author={Zhang, Xiangyu and Eseye, Abinet Tesfaye and Knueven, Bernard and Liu, Weijia and Reynolds, Matthew and Jones, Wesley},
  journal={IEEE Transactions on Power Systems},
  year={2022},
  publisher={IEEE}
}
```
