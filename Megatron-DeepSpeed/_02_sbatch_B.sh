#!/bin/bash
#SBATCH --nodes=1              
#SBATCH --partition=gpu3
#SBATCH --gres=gpu:a10:2
#SBATCH --cpus-per-task=14
#SBATCH -o ./_log2/_sbatch-log/%j.sbatch.%N.out         
#SBATCH -e ./_log2/_sbatch-log/%j.sbatch.%N.err
#SBATCH --job-name=A10
#SBATCH --nodelist=n071
#************************************************************
GRES="gpu:a10:2"
TP=$1
PP=$2
DP=$3
PARTITION=$4
CPUS_PER_TASK=14
GPU_TYPE=A10
. _00_conf.sh $TP $PP $DP $PARTITION
#************************************************************
echo "true" > BRUN.txt
ARUN=$(<ARUN.txt)
BRUN=$(<BRUN.txt)
CRUN=$(<CRUN.txt)
while !([ "$ARUN" = "true" ] && [ "$BRUN" = "true" ] && [ "$CRUN" = "true" ])
        do  
            echo "ARUN: $ARUN"
            echo "BRUN: $BRUN"
            echo "CRUN: $CRUN"
            
            ARUN=$(<ARUN.txt)
            BRUN=$(<BRUN.txt)
            CRUN=$(<CRUN.txt)
            sleep 1
        done
#************************************************************

cd "/home2/eung0/Megatron-DeepSpeed.240904"

SHELL_PATH=`pwd -P`
echo $SHELL_PATH

mkdir -p ./_log2/$SLURM_JOB_ID

# get_master_adress
B_NODES=($B_NODES_STR)
function get_master_adress(){
    NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
    MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
    MASTER_ADDR_LIST=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`

    MASTER_ADDR_ARR=($MASTER_ADDR_LIST)
    MASTER_ADDR=${MASTER_ADDR_ARR[1]}
    echo B_NODES MASTER_ADDR_ARR:$MASTER_ADDR_ARR
    echo B_NODES MASTER_ADDR:$MASTER_ADDR
}

if [ "${B_NODES[0]}" = 0 ] ; then 
    get_master_adress
    read -ra MASTER_ADDR <<< "$MASTER_ADDR"
    echo ${MASTER_ADDR[0]} > MASTER_ADDR.txt
    echo ${MASTER_ADDR[0]}
    #JOBID 저장
    echo $SLURM_JOB_ID > MASTERJOBID.txt
else
    sleep 1
    MASTER_ADDR=$(<MASTER_ADDR.txt)
    #JOBID 저장
    echo $SLURM_JOB_ID >> SLAVEJOBID.txt

fi
echo B_NODES MASTER_ADDR:$MASTER_ADDR
echo B_NODES CONTAINER_PATH:$CONTAINER_PATH


INIT_CONTAINER_SCRIPT=$(cat <<EOF
    
    if $RELOAD_CONTAINER ; then
        rm -rf $CONTAINER_PATH
    fi

    if [ -d "$CONTAINER_PATH" ] ; then 
        echo "container exist";
    else
        enroot create -n $CONTAINER_NAME $CONTAINER_IMAGE_PATH ;
    fi

EOF
)

ENROOT_SCRIPT="cd /Megatron-DeepSpeed && \
                bash _01_run_inter.sh $TP $PP $DP $PARTITION $SLURM_JOB_ID"

SRUN_SCRIPT=$(cat <<EOF

    $INIT_CONTAINER_SCRIPT

    free -m

    # hostname
    hostnode=\$(hostname -s) # 작업을 던지는 node의 hostname이 담김
    echo "I am: \$hostnode"
    
    # SLURM_HOST_ARR
    SLURM_HOST_LIST=\$(scontrol show hostnames \$SLURM_JOB_NODELIST) # slurm node 숫자 순서대로 담기는 array
    SLURM_HOST_ARR=(\$SLURM_HOST_LIST) # slurm node 숫자 array
    echo "B NODES $GPU_TYPE SLURM_HOST_ARR: \${SLURM_HOST_ARR[@]}" # B type GPU의 slurm node 이름 array. ex) [n061 n062 n063 n064] 
    
    # B_NODES_INDEX_ARR
    B_NODES_INDEX_ARR=($B_NODES_STR) # defined in _00_conf.sh
    echo "B NODES $GPU_TYPE B_NODES_INDEX_ARR: \${B_NODES_INDEX_ARR[@]}" # B type GPU의 index가 있는 array. ex) [0 2 4 5]

    for index in "\${!SLURM_HOST_ARR[@]}"; do
        # SLURM_HOST_ARR에서 현재 SLRUM NODE의 순서 INDEX를 얻음 이를 B_NODES_INDEX_ARR에서 INDEX와 매치하여 NODE_RANK를 얻는 코드
        h=\${SLURM_HOST_ARR[\$index]} # n061

        if [[ "\$hostnode" == "\$h" ]]; then
            NODE_RANK=\${B_NODES_INDEX_ARR[\$index]} # 0
            echo "NODE_RANK : \$NODE_RANK"
            break 2
        fi
    done
    
    [ \$NODE_RANK == "0" ] && echo "I AM MASTER_NODE \$hostname" || echo ""    

    enroot start --root \
                --rw \
                -m $HOME/Megatron-DeepSpeed.240904:/Megatron-DeepSpeed \
                $CONTAINER_NAME \
                bash -c "$ENROOT_SCRIPT \$NODE_RANK $MASTER_ADDR \$hostnode"
EOF
)


srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=$CPUS_PER_TASK \
      -o ./_log2/%j/%N.out \
      -e ./_log2/%j/%N.err \
      bash -c "$SRUN_SCRIPT"

