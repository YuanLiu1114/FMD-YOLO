# FMD-YOLO

This repository provides the reproducibility materials for the manuscript:

**Contextual Modulation and Task-Specific Routing for Fish Disease Monitoring in Recirculating Aquaculture Systems**

## Released materials

- `code/`: FMD-YOLO implementation based on Ultralytics YOLOv8.2.42
- `code/train.py`: training entry script and the hyperparameters used in the manuscript
- `test_data/`: held-out test images and YOLO-format annotations
- `prompt_zh.txt`: Chinese prompt template used for video generation
- `prompt_eng.txt`: English translation of the prompt template
- `code/ultralytics/cfg/models/v8/yolov8l-decoupledhead_with_C2fFM.yaml`:
  FMD-YOLO model configuration

## Test-set classes

- 0: Bass
- 1: Hemorrhage
- 2: Ocular lesion
- 3: Whitish

The released test data are provided for research reproducibility and must not
be used for model training or hyperparameter selection when reporting results
on this test set.

## Licence

The source code is distributed under the GNU AGPL-3.0 licence retained in
`code/LICENSE`.

## Contact

Yuan Liu: liu_yuan@nwafu.edu.cn  
Zhenhua Wang: zhenhuawang@nwafu.edu.cn