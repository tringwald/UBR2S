#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
_ADD_FLAGS=(
"--configs" "configs/single-source/office-home.yaml"
)
_EXP="final"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Art --target-dataset Office-Home/Clipart --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Art --target-dataset Office-Home/Product --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Art --target-dataset Office-Home/Real\ World --sub-dir $_EXP "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Clipart --target-dataset Office-Home/Art --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Clipart --target-dataset Office-Home/Product --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Clipart --target-dataset Office-Home/Real\ World --sub-dir $_EXP "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Product --target-dataset Office-Home/Art --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Product --target-dataset Office-Home/Clipart --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Product --target-dataset Office-Home/Real\ World --sub-dir $_EXP "${_ADD_FLAGS[@]}"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Real\ World --target-dataset Office-Home/Art --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Real\ World --target-dataset Office-Home/Clipart --sub-dir $_EXP "${_ADD_FLAGS[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Real\ World --target-dataset Office-Home/Product --sub-dir $_EXP "${_ADD_FLAGS[@]}"