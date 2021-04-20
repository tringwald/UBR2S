#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
_ADD_FLAGS=(
"--configs" "configs/single-source/office-caltech.yaml"
)
_EXP="final"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/amazon --target-dataset Office-Caltech/caltech --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/amazon --target-dataset Office-Caltech/dslr --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/amazon --target-dataset Office-Caltech/webcam --sub-dir $_EXP "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/caltech --target-dataset Office-Caltech/amazon --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/caltech --target-dataset Office-Caltech/dslr --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/caltech --target-dataset Office-Caltech/webcam --sub-dir $_EXP "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/dslr --target-dataset Office-Caltech/amazon --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/dslr --target-dataset Office-Caltech/caltech --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/dslr --target-dataset Office-Caltech/webcam --sub-dir $_EXP "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/webcam --target-dataset Office-Caltech/amazon --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/webcam --target-dataset Office-Caltech/caltech --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Caltech/webcam --target-dataset Office-Caltech/dslr --sub-dir $_EXP "${_ADD_FLAGS[@]}"