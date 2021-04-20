#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
for i in configs/seeds/*.yaml;
do
  _ADD_FLAGS=(
  "--configs" "configs/single-source/visda17.yaml" "$i"
  )
  _EXP="ablation_VISDA-VAL_$i"
  PYTHONHASHSEED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset VISDA17/train --target-dataset VISDA17/validation --sub-dir $_EXP "${_ADD_FLAGS[@]}";
done;
