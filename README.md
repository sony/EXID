# Enhancing Generalization of Offline RL in Data-Limited Settings with Heuristic Rules

Official implementation of Enhancing Generalization of Offline RL in Data-Limited Settings with Heuristic Rules (Accepted in IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE 2025)
Authors : Briti Gangoppadhyay, Wang Zhao, Jia-Fong Yeh, and Shingo Takamatsu

## System Requirements

The code has been tested on systems with the following OS:
- **Ubuntu 20.04.2 LTS**

## Dependencies

To reproduce all results, use the provided `requirements.txt` file to set up a Conda environment with the required packages.

Run the following commands to create and activate the environment:

```sh
conda create --name exidenv --file requirements.txt
conda activate exidenv
pip install -e .
```

Alternatively, you can create the environment manually:

```sh
conda create --name exidenv python=3.7.0
conda activate exidenv
python -m pip install -r piprequirements.txt
```

## Data Creation and Processing

We use datasets from [OfflineRL](https://github.com/kschweig/OfflineRL) and conduct our experiments using the **run3 dataset**.

### Steps to Create the Dataset:
1. Clone and install dependencies from [OfflineRL](https://github.com/kschweig/OfflineRL).
2. Run `expnumber.py --online` (e.g., for the Mountain Car environment: `ex02.py --online`).
3. Process the data into our buffer type:
   ```sh
   python process.py --exp_name "path_to_dataset"
   ```

Additional experiment files for **Lunar-Lander (`ex_07.py`)** and **Acrobot (`ex_08.py`)** are provided in the `experiment/` folder.

## Hyperparameter Configurations

Hyperparameters for each environment are listed in the `config/` folder. Below are some important hyperparameters:

| Parameter        | Description |
|-----------------|-------------|
| `episodes`      | Number of episodes to train on |
| `seed`          | Default: 1 (Experiments conducted on seeds: 1, 42, 76) |
| `data_file`     | Data file name (e.g., `data/Mountain_car_expertRun3.pkl`) |
| `data_type`     | Data type (`er`: expert, `rep`: replay, `ns`: noisy) |
| `data_percent`  | Percentage of data used for training (Default: `0.1`) |
| `use_heur`      | Set to `True` when evaluating baseline with domain knowledge, otherwise `False` |
| `use_teach`     | Set to `True` when training with the teacher network, `False` when training baseline CQL |
| `warm_start`    | Specifies if the student starts by learning only from the teacher |
| `teacher_update` | The episode interval for teacher updates |
| `lam`           | Lambda value for regularization (Default: `0.5`) |
| `algo_type`     | Supported algorithms: `QRDQN`, `REM`, `BVE`, `CRR`, `MCE`, `BC`, `BCQ` |

## Constructing Teacher Actor using BC and Expert Policy

The code for constructing the teacher policy is available in **`ConstructingTeacherActorusingBC.ipynb`**.
Random data from the **run3 dataset** is used as states for the teacher policy.

## Training the Baseline Algorithm

To train baselines and evaluate with domain knowledge, set `use_heur: True` and `algo_type` to one of (`QRDQN`, `REM`, `BVE`, `CRR`, `MCE`, `BC`, `BCQ`) in the config file.

Run the following command:
```sh
python train_baseline.py --config_file config/mountain.config
```

To train the **baseline CQL**, set `use_heur: True`, `use_teach: False` in the config file, and run:
```sh
python train.py --config_file config/mountain.config
```

## Training with EXID

To train an **EXID** agent, set `use_heur: False`, `use_teach: True`, and `algo_type: ExID` in the config file.
Run the following command:

```sh
python train.py --config_file config/mountain.config
```

## Sales Promotion Environment Experiments

To run experiments in a **Sales Promotion** environment:
1. Install and set up **NeORL** environments following [NeORL Benchmark](https://github.com/polixir/NeoRL/tree/benchmark).
2. Install **CORL** libraries from [CORL Repository](https://github.com/tinkoff-ai/CORL).
3. Run:
   ```sh
   python exidsp.py
   ```

## Data Distribution and Coverage Plots

All data distribution and state-action coverage plots can be generated using **`Plotting State Distribution and State Action Coverage.ipynb`**.

## Citation

If you use this work, please cite:

```bibtex
@article{gangopadhyay2024exid,
  title={ExID: Offline RL with Intuitive Expert Insights in Limited-Data Settings},
  author={Gangopadhyay, Briti and Wang, Zhao and Yeh, Jia-Fong and Takamatsu, Shingo},
  year={2024}
}
```

