#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
_ADD_FLAGS=(
"--configs" "configs/multi-source/office-home.yaml"
)
_EXP="final_multisource"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Product Office-Home/Art Office-Home/Clipart --target-dataset Office-Home/Real\ World --sub-dir $_EXP "${_ADD_FLAGS[@]}"

_ADD_FLAGS=(
"--configs" "configs/multi-source/office-home_merged.yaml"
)
_EXP="final_multisource_merged"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Product Office-Home/Art Office-Home/Clipart --target-dataset Office-Home/Real\ World --sub-dir $_EXP "${_ADD_FLAGS[@]}"