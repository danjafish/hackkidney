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



#LR=(1e-4 1e-5)
#DECAY=(0.5 0.3 0.2 0.1 0.05 0.01 0.005)
#BACKBONE=('unen','effnet')


#for lr in "${LOSS_WEIGHTS[@]}" ; do
#    echo $lr
#done

#A = ((1 2) (1 2))
#declare -A arr
#arr = (['LR']=(0.1 0.01 0.02) ['BACKBONE']=('unet' 'effnet'))
#arr["key1"]=val1

#arr+=( ["key2"]=val2 ["key3"]=val3 )
#params=(['LR']=(0.1 0.01 0.02) ['BACKBONE']=('unet' 'effnet'))
#params(["LR"]=(1 2 3))



# declare -A X=(
#   ['p1']="1 2 3"
#   ['p2']="A B C"
#   #['s3']="other"
# )

#declare -a names=(${X[s1]})

#for key in ${!X[@]}
#do
    #echo $key '=>' {$X[$key]}
    #echo "$key => ${X[$key]}"
    #for value in  ${X[$key]}
    #do
        #echo $key $value
    #done
#done





#for i in "${!params[@]}"
#do
#  echo "key  : $i"
#  echo "value: ${array[$i]}"
#done



#for lr in ${LR[@]} ; do
#    for backbone in ${BACKBONE[@]} ; do
#        echo $lr $backbone
#    done
#done