## Overview
DTANet is a deep learning model inspired by GraphDTA and Graph Kolmogorov-Arnold Networks
to learn and predict drug-target interactions.

## Resources
| Files/Folders            | Description                                                                                         |
| :----------------------- | --------------------------------------------------------------------------------------------------- |
| model.py                 | DTANet = GraphKAN (kanChebConv + KANLinear) + 1D CNN + MLP                                          |
| learn.py                 | Learning process without validation                                                                 |
| learn_with_validation.py | Learning process with validation                                                                    |
| utils.py                 | TestbedDataset + performance measures                                                               |
| create_data.py           | Manages preprocessing steps and finally creates proper data in PyTorch format for learning process. |
| data                     | Davis and Kiba datasets                                                                             |
| final results            | Learned parameters of the model plus the best result of each dataset                                |

## Step-by-step running
Soon...


## License
under [MIT-licensed](./LICENSE).