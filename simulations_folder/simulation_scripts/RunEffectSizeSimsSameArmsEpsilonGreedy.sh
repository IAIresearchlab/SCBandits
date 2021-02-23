#!/bin/bash

numSims=2

timestamp=$(date +%F_%T)

simSetDescriptive="../simulation_saves/EpsilonGreedyNoEffectTest" #Give descriptive name for directory to save set of sims
mkdir -p $simSetDescriptive

effectSizesB=(0.1 0.3 0.5);
#effectSizesB=(0.3);

nsT=(394 64 26);
nsB=(785 88 32);

arrayLength=${#effectSizesB[@]};
#nsB=(32); #hard coded sample sizes corresponding to es

bsProps=(0.25);
bsProps_len=${#bsProps[@]};
armProbs=(0.5); #as default
armProbs_len=${#armProbs[@]};

#for ((a=0; a<$armProbs_len; a++)); do
armProb=${armProbs[0]}
root_armProb=$simSetDescriptive/"num_sims="$numSims"armProb="$armProb
echo $armProb

epsilon=0.3 #0.1, 0.3
for ((i=0; i<$arrayLength; i++)); do
    curN=${nsB[$i]}
    curEffectSize=${effectSizesB[$i]}
    root_armProb_es=$root_armProb/"N="$curN"epsilon="$epsilon

   # for ((j=0; j<$bsProps_len; j++)); do
    curProp=${bsProps[0]}
    
    batch_size_fl=$(awk -v curN="${curN}" -v curProp="${curProp}" 'BEGIN{print (curN*curProp)}')
    batch_size=${batch_size_fl%.*}
    batch_size=1 #using 1 for now
    echo $batch_size
    burn_in_size=$batch_size
   # mkdir -p $root_dir

    #TS---------------
    root_armProb_es_prop_ts=$root_armProb_es"/bbEqualMeansEqualPriorburn_in_size-batch_size="$burn_in_size"-"$batch_size #root for this sim, split by equals for bs #Note, misnamed before (Unequal -> Equal)
  
    directoryName_ts=$root_armProb_es_prop_ts
    echo $directoryName_ts
    mkdir -p "$directoryName_ts"
    
    #python3 run_effect_size_simulations_beta_epsilon_greedy.py \
    #$curEffectSize"-"$armProb $numSims $directoryName_ts "Thompson" $epsilon 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
    #echo $!
   
    #from equalmeans for reference
    python3 run_effect_size_simulations_beta_epsilon_greedy.py \
    0.5,0.5 $numSims $directoryName_ts "Thompson" "armsEqual" $curN 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
    echo $!

    #Uniform------------
    root_armProb_es_prop_unif=$root_armProb_es"/bbEqualMeansUniformburn_in_size-batch_size="$burn_in_size"-"$batch_size;
    directoryName_unif=$root_armProb_es_prop_unif
    mkdir -p "$directoryName_unif"
    #python3 run_effect_size_simulations_beta_epsilon_greedy.py \
   # $curEffectSize"-"$armProb $numSims $directoryName_unif "uniform" 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
    
    #from equalmeans for reference
    python3 run_effect_size_simulations_beta_epsilon_greedy.py \
    0.5,0.5 $numSims $directoryName_unif "uniform" "armsEqual" $curN 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
done
#done



