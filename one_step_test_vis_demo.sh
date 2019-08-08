#!/bin/bash

SNAPSHOT_ITER=""
SPECIFIC_MODEL_DIR=$(cd `dirname $0`; pwd)'/results/vis_zebra2_10'

#python3 "./tools/run_test_in_folder.py" "$SPECIFIC_MODEL_DIR" "'test_subset':'test', 'test_limit':None" "test.test" "$SNAPSHOT_ITER" "False" "True"
 
TEST_PRED_FILE=$SPECIFIC_MODEL_DIR'/test.test/posterior_param.mat'
matlab -nosplash -nodesktop -r "tmp=load('$TEST_PRED_FILE'); cd demo; vppAutoKeypointShow(tmp.data, tmp.encoded.structure_param,'output');exit()"
