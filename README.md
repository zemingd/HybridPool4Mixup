Implementation of the paper: On the Effectiveness of Hybrid Pooling in Mixup-Based Graph Learning for Language Processing [[arxiv]](https://arxiv.org/abs/2210.03123).

## Requirements
On Ubuntu:

### Installation
- install python packages
```shell
pip install tqdm
pip install pandas
pip install ogb
pip install keras
pip install scikit-learn
pip install scipy
```

- Task: Problem Classification

```shell
Python>=3.6
cuDNN>=7.6
PyTorch>=version 1.8.0) 
Pytorch Geometric>=version 1.6.3
CUDA 11.0
```

- Task: Fake News Detection
```shell
Python>=3.6
cuDNN>=7.6
Pytorch>=version 1.8.0
Pytorch Geometric>=version 1.6.3
CUDA 11.0
```

## Dataset
- Java250: https://developer.ibm.com/exchanges/data/all/project-codenet/
- Python800: https://developer.ibm.com/exchanges/data/all/project-codenet/
- Gossipcop/Politifact: https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR?usp=sharing

## Citation
If you use the code in your research, please cite:
```bibtex
    @article{dong2022enhancing,
    title={Enhancing Mixup-Based Graph Learning for Language Processing via Hybrid Pooling},
    author={Dong, Zeming and Hu, Qiang and Guo, Yuejun and Cordy, Maxime and Papadakis, Mike and Traon, Yves Le and Zhao, Jianjun},
    journal={arXiv preprint arXiv:2210.03123},
    year={2022}
}
```
