
<h1 align="center">HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking</h1>
<p align="center">
[<a href="https://aintdoin.github.io/hypertree-planning.github.io/">Website</a>]
[<a href="https://arxiv.org/pdf/2505.02322">Paper</a>] 

</p>

## What's New
+  [2025.05.05] The paper is available at [HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking](https://arxiv.org/pdf/2505.02322).
+  [2025.05.30] The paper has been accepted at ICML.
+  [2025.06.30] The website is available at [HTP](https://aintdoin.github.io/hypertree-planning.github.io/).

## HTP

HyperTree Planning (HTP) is a novel reasoning paradigm that enhances Large Language Model (LLM) performance in complex planning tasks. To address the challenges of extended reasoning steps, diverse constraints, and multiple distinct sub-tasks, HTP constructs hypertree-structured planning outlines. Motivated by hierarchical thinking, HTP enables LLMs to flexibly employ a divide-and-conquer strategy, effectively breaking down intricate reasoning steps, accommodating diverse constraints, and managing multiple distinct sub-tasks in a well-organized manner. This framework further introduces an autonomous planning process that iteratively refines and expands these hypertree-structured planning outlines to complete the planning.

<p align="center">
<img width="100%" alt="image" src="./figures/Figure2.png">    
</p>

More details are in the paper.

## Experiment

### TravelPlanner
We test HTP on the sole-planning mode. The sole-planning mode ensures that no crucial information is missed, thereby enabling agents to focus on planning itself.
Please take a look at the paper for more details.

#### Prerequisites

1. Create a conda environment and install dependency:
```bash
cd travelplanner/
conda create -n travelplanner python=3.9
conda activate travelplanner
pip install -r requirements.txt
```
2. Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `TravelPlanner` directory (i.e., `your/path/TravelPlanner`).

#### Evaluation

```bash
cd travelplanner/scripts
bash eval.sh
```

#### Reasoning

This code is adapted from [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner)

```bash
cd travelplanner/scripts
bash run-soleplan.sh
```

### Natural Plan
We test HTP on the Trip Planning task.
Please take a look at the paper for more details.

#### Evaluation

```bash
cd natural_plan/scripts
bash evaluate.sh
```

#### Reasoning

This code is adapted from [Natural Plan](https://github.com/google-deepmind/natural-plan)

```bash
cd natural_plan/scripts
bash response.sh
```


## Citation

If you find this work useful in your method, you can cite the paper as below:

```bash
@inproceedings{gui2025hypertree,
  title={HyperTree Planning: Enhancing LLM Reasoning via Hierarchical Thinking},
  author={Gui, Runquan and Wang, Zhihai and Wang, Jie and Ma, Chi and Zhen, Huiling and Yuan, Mingxuan and Hao, Jianye and Lian, Defu and Chen, Enhong and Wu, Feng},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```
