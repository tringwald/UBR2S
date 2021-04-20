#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
_ADD_FLAGS=(
"--configs" "configs/multi-source/office-31.yaml"
)
_EXP="final_multisource"
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-31/dslr Office-31/webcam --target-dataset Office-31/amazon --sub-dir $_EXP "${_ADD_FLAGS[@]}"


_ADD_FLAGS=(
"--configs" "configs/multi-source/office-31_merged.yaml"
)
_EXP="final_multisource_merged"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-31/dslr Office-31/webcam --target-dataset Office-31/amazon --sub-dir $_EXP "${_ADD_FLAGS[@]}"