# ThermalGuardian: Temperature-Aware Testing of Automotive Deep Learning Frameworks

This is the implementation repository of our *ISSTA'26* paper: **ThermalGuardian: Temperature-Aware Testing of Automotive Deep Learning Frameworks**.



## 1. Description

Deep learning models play a vital role in autonomous driving systems, supporting critical functions such as environmental perception. To accelerate model inference, these deep learning models' deployment relies on automotive deep learning frameworks, for example, PaddleInference in Apollo and TensorRT in AutoWare. However, unlike deploying deep learning models on the cloud, vehicular environments experience extreme ambient temperatures varying from -40°C to 50°C, significantly impacting GPU temperature. Additionally, heats generated when computing further lead to the GPU temperature increase. These temperature fluctuations lead to dynamic GPU frequency adjustments through mechanisms such as DVFS. However, automotive deep learning frameworks are designed without considering the impact of temperature-induced frequency variations. When deployed on temperature-varying GPUs, these frameworks suffer critical quality issues: compute-intensive operators face delays or errors, high/mixed-precision operators suffer from precision errors, and time-series operators suffer from synchronization issues. The above quality issues cannot be detected by existing deep learning framework testing methods because they ignore temperature's effect on the deep learning framework quality. To bridge this gap, we propose ThermalGuardian, the first automotive deep learning framework testing method under temperature-varying environments. Specifically, ThermalGuardian generates test input models using model mutation rules targeting temperature-sensitive operators, simulates GPU temperature fluctuations based on Newton's law of cooling, and controls GPU frequency based on real-time GPU temperature. Evaluated on PaddleInference and TensorRT, ThermalGuardian successfully detects 18 crashes and 3 NaN bugs, outperforming all baseline methods. Moreover, it runs twice as fast as the top-performing baseline, while achieving 85% operator coverage, 89.02% code coverage, and 100% temperature-sensitive operator coverage—surpassing all baselines.

You can access this repository using the following command:

```shell
git clone https://github.com/ThermalGuardian/ThermalGuardian.git
```



## 2. API version

We deploy our method in the most widely used open-source autonomous driving system, Apollo and Autoware, and test their native automotive DL framework, PaddleInference and TensorRT. The adopted API versions are as follows.

| PaddlePaddle | PaddleInference | CUDA | CUDNN | NVIDIA-driver | Autoware | Apollo | TensorRT |
| :----------: | :-------------: | :--: | :---: | :-----------: | :------: | :----: | :------: |
|    2.6.2     |      2.6.2      | 12.4 | 9.6.0 |  535.216.01   | universe |  9.0   | 8.5.3.1  |



## 3. Environment

**Step 0:** Clone the source code of Apollo. Run the following command. Move it under the folder ***ThermalGuardian***.

```sh
git clone git@github.com:ApolloAuto/apollo.git
cd apollo
git checkout master
```

Clone the source code of Autoware. Run the following command. Move it under the folder ***ThermalGuardian***.

```
git clone https://github.com/autowarefoundation/autoware_universe.git
cd autoware_universe
git checkout main
```

**Make sure you have configured CUDA, CUDNN, and NVIDIA-driver properly**

**Step 1:** Modify the VERSION_X86_64 image version in docker/scripts/dev_start.sh as follows.

```sh
VERSION_X86_64="dev-x86_64-18.04-20231128_2222"
```

**Step 2:** Set up the container. Run the following command.

```sh
bash docker/scripts/dev_start.sh
```

**Step 3:** Enter the container.

```sh
bash docker/scripts/dev_into.sh
```

**Step 4:** Change the content in third_party/centerpoint_infer_op/workspace.bzl as follows.

```
"""Loads the paddlelite library"""
﻿
# Sanitize a dependency so that it works correctly from code that includes
# Apollo as a submodule.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
﻿
def clean_dep(dep):
    return str(Label(dep))
﻿
def repo():
    http_archive(
        name = "centerpoint_infer_op-x86_64",
        sha256 = "038470fc2e741ebc43aefe365fc23400bc162c1b4cbb74d8c8019f84f2498190",
        strip_prefix = "centerpoint_infer_op",
        urls = ["https://apollo-pkg-beta.bj.bcebos.com/archive/centerpoint_infer_op_cu118.tar.gz"],
    )
﻿
    http_archive(
        name = "centerpoint_infer_op-aarch64",
        sha256 = "e7c933db4237399980c5217fa6a81dff622b00e3a23f0a1deb859743f7977fc1",
        strip_prefix = "centerpoint_infer_op",
        urls = ["https://apollo-pkg-beta.bj.bcebos.com/archive/centerpoint_infer_op-linux-aarch64-1.0.0.tar.gz"],
    )
﻿
```

**Step 5:** Change the content in third_party/paddleinference/workspace.bzl as follows.

```
"""Loads the paddlelite library"""
﻿
# Sanitize a dependency so that it works correctly from code that includes
# Apollo as a submodule.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
﻿
def clean_dep(dep):
    return str(Label(dep))
﻿
def repo():
    http_archive(
        name = "paddleinference-x86_64",
        sha256 = "7498df1f9bbaf5580c289a67920eea1a975311764c4b12a62c93b33a081e7520",
        strip_prefix = "paddleinference",
        urls = ["https://apollo-pkg-beta.cdn.bcebos.com/archive/paddleinference-cu118-x86.tar.gz"],
    )
﻿
    http_archive(
        name = "paddleinference-aarch64",
        sha256 = "048d1d7799ffdd7bd8876e33bc68f28c3af911ff923c10b362340bd83ded04b3",
        strip_prefix = "paddleinference",
        urls = ["https://apollo-pkg-beta.bj.bcebos.com/archive/paddleinference-linux-aarch64-1.0.0.tar.gz"],
    )
﻿
```

**Step 6:** Set up the APIs. Run the following command. 

```
pip install -r requirements.txt
```

**Step 7:** Download the KITTI dataset in [The KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/user_submit.php). Among them, the left color images of object data set (12GB) is adopted for Single Camera Detection and Multiple Camera Detection, and the Velodyne point clouds (29GB) is adopted for LiDAR Detection. In addition, the camera calibration matrices of object data set (16MB), and the training labels of object data set (5MB) are also necessary for data preprocess.

**Step 8:** Download the dataset split file list using the following command:

```
wget https://bj.bcebos.com/paddle3d/datasets/KITTI/ImageSets.tar.gz
```

**Step 9:** Organize the extracted data for image meta data according to the directory structure below.

```
$ tree kitti_dataset_root
kitti_dataset_root
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   └── val.txt
└── training
    ├── calib
    ├── image_2
    └── label_2
    └── velodyne
```

**(If you want to adopt Nuscenes dataset, please download it and organize it in the same way.)**

**Step 10:** Set paths in ***ThermalGuardianDatastruct/globalConfig.py*** to your path, including ***modeltype_and_configpath***, ***exported_model_path***, and ***exported_model_weight_path***.

**Step 11:** Set dataset paths in ***ThermalGuardian/configs*** to the path of your dataset!

**Step 12:** Run ThermalGuardian using the following command:

```python
python main.py
```

During execution, guarantee the Apollo container is on!!!

## 4. File structure

This project contains six folders. The **LEMON-master** folder is the downloaded open source code for LEMON. The **Muffin-main** folder is the downloaded open source code for Muffin. The **Gandalf-main** folder is the downloaded open source code for Gandalf. The **DevMut-master** folder is the downloaded open source code for DevMut. The **ThermalGuardian** folder is the source code for our method. The **result** folder is the experimental result data. To know the execution methods of our baselines, please refer to the corresponding research papers. In this document, we will introduce how to run the source code for **ThermalGuardian**.

In the source code for ThermalGuardian, the program entry of the method is **main.py**. Run **main.py** to run ThermalGuardian after installing the experimental environment.

If you do not want to reproduce the experiment, experimental results are available in the folder **result**. There are two folders in the folder **result**: 1) Folder **crash_logs** for the logs of all detected crashes. 2) Folder **NaN&inconsistency** for the logs of all detected NaNs & Inconsistencies.
