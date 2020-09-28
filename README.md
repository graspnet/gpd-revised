# gpd-revised

### Introductions

GPD-revised is a repository that originated from the [gpd](https://github.com/atenpas/gpd) [1] repository by [atenpas](https://github.com/atenpas/). We make some modifications to the gpd network in order to adapt the new dataset generalized by [Haoshu Fang](https://github.com/fang-haoshu) and [Chenxi Wang](https://github.com/chenxi-wang).

### Files

```
gpd-revised
├── original-gpd
|   └── ...
├── pytorch
|   ├── train_generator.py
|   ├── test_generator.py
|   ├── network.py
|   ├── json_dataset.py
|   ├── train_net.py
|   └── continue_train_net.py
└── README.md    
```

The `original-gpd` folder includes all the codes in the original [gpd](https://github.com/atenpas/gpd) repository, and the `pytorch` folder is the code after modifications that can adapt the new dataset. Here are some specific explanations.

- `train_generator.py` and `test_generator.py` are data generator program, including the selection of the dataset, that is, balance the positive and negative samples;
- `json_dataset.py` contains the dataset manager, which are vital to our codes. Notice that we have changed the data manager file from `h5` to `json` in order to satisfy the needs of the new dataset;
- `network.py` is the main structure of the network, and some vital modifications are made in order to suit the dataset well;
- `train_net.py` and `continue_train_net.py` are the trainning codes, and the latter one supports reloading the current model and continuing the training process. Both of the codes are modified from the original code `train_net3.py`.

### Requirements

In order to use our codes to train the model, you need the Minkowski Engine dataset generalized by [Haoshu Fang](https://github.com/fang-haoshu) and [Chenxi Wang](https://github.com/chenxi-wang), and the file structure of the dataset should be as follows.

```
gpd_data
├── train1
|   ├── image
|   |   ├── 000000.jpg
|   |   ├── 000001.jpg
|   |   └── ...
|   └── labels_train1.npy
├── train2
|   └── ...
├── train3
|   └── ...
├── train4
|   └── ...
└── test_seen
    └── ...
```

You may need to do some simple modifications to the `train_generator.py` and `test_generator.py` to satisfy your own path requirements.

Other requirements include `pytorch` framework and some other dependencies, you can refer to the codes for details.

### Usages

To use our codes to train the net, you may follow the steps listed here.

- Make some modifications to the `train_generator.py` and `test_generator.py` to satisfy your own path requirements;
- Run `train_generator.py` and `test_generator.py`;
- Run `train_net.py`, and if you want to reload the old model and continue the training progress, you can run `continue_train_net.py` instead of the previous one. 

### References

[1] Andreas ten Pas, Marcus Gualtieri, Kate Saenko, and Robert Platt. [**Grasp Pose Detection in Point Clouds**](http://arxiv.org/abs/1706.09911). The International Journal of Robotics Research, Vol 36, Issue 13-14, pp. 1455-1473. October 2017.