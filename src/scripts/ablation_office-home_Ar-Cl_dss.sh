#!/bin/bash
exit_script() {
  echo "Killing child processes."
  trap - INT TERM # clear the trap
  kill -- -$$     # Sends SIGTERM to child/sub processes
}
trap exit_script INT TERM

###############################################################################################################################
for i in configs/ablations_dss/*.yaml;
do
  _ADD_FLAGS=(
  "--configs" "configs/single-source/office-home.yaml" "$i"
  )
  _EXP="ablation_OH_Ar-Cl_$i"
  PYTHONHASHSEED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train.py --source-dataset Office-Home/Art --target-dataset Office-Home/Clipart --sub-dir $_EXP "${_ADD_FLAGS[@]}";
done;
