#!/usr/bin/bash

SIZE=(512 1024)
SIZE_AFTER_RESHAPE=(320)
STEP_SIZE_RATIO=(0.25 0.5 0.75)
LOSS_WEIGHTS=("1 3 1" "1 2 3")
NOT_EMPTY_RATIO=(0.5)
EPOCHS=(30 40 50)
BS=(32)
MAX_LR=(0.0002 0.0007)


for size in ${SIZE[@]} ; do
    for size_after_reshape in ${SIZE_AFTER_RESHAPE[@]} ; do
        for step_size_ratio in ${STEP_SIZE_RATIO[@]} ; do
            for loss_weights in "${LOSS_WEIGHTS[@]}" ; do
                for not_empty_ratis in ${NOT_EMPTY_RATIO[@]} ; do
                    for epochs in ${EPOCHS[@]} ; do 
                        for bs in ${BS[@]} ; do
                            for max_lr in ${MAX_LR[@]} ; do
                                echo $size $size_after_reshape $step_size_ratio $loss_weights $not_empty_ratis $epochs $bs $max_lr
                            done
                        done
                    done
                done
            done
        done
    done

done


