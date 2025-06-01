## Overview
DTANet is a deep learning model inspired by Graph Kolmogorov-Arnold Networks and GraphDTA<br>
to learn and predict drug-target interactions with promising performance.<br><br>
![Architecture of DTANet](https://github.com/kourosh-y/DTANet/blob/main/white%20paper/DTANet_arch.png)

## Resources
| Files/Folders            | Description                                                                                         |
| :----------------------- | --------------------------------------------------------------------------------------------------- |
| model.py                 | DTANet = GraphKAN (kanChebConv + KANLinear) + 1D CNN + MLP                                          |
| train_test.py            | Learning process without validation                                                                 |
| train_validate_test.py   | Learning process with validation                                                                    |
| utils.py                 | TestbedDataset + performance measures                                                               |
| create_data.py           | Manages preprocessing steps and finally creates proper data in PyTorch format for learning process. |
| data                     | Davis and Kiba datasets                                                                             |
| final results            | Learned parameters of the model plus the best result of each dataset                                |
| white paper              | Soon ...                                                                                             |

## Step-by-step running
Soon ...


## License
This repo is provided for showcasing results of my final BSs project only. 
You may use, modify, and redistribute this code, provided that you:

1. Do not use this code for academic publication without the explicit permission of the author.
2. Cite the original repository if you use this code in any academic or research context.
3. Contact the author before publishing any paper that directly builds upon or benchmarks against this work.

All other rights reserved.

