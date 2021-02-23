#!/bin/bash

numSims=2
#variancesT=(5.0 2.0 1.25); # Note: we're not crossing nsT and variancesT because variancesT is actual variance with ES with that num steps

timestamp=$(date +%F_%T)

#simSetDescriptive=$timestamp"IsEffect_Allcenters" #Give descriptive name for directory to save set of sims
#simSetDescriptive="IsEffect_bs_RL4RLMay12" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/IsEffect_bs_TEST" #Give descriptive name for directory to save set of sims
mkdir -p $simSetDescriptive

#effectSizesB=(0.1 0.3);
effectSizesB=(0.3);
arrayLength=${#effectSizesB[@]};
#nsB=(785 88 32); #hard coded sample sizes corresponding to es
nsB=(88);

bsProps=(0.05 0.10 0.25);
bsProps_len=${#bsProps[@]};
armProbs=(0.2 0.5 0.8); #centres
armProbs=(0.5); 

#armProbs=(0.5); #centres, not passed in currently --old comment?
armProbs_len=${#armProbs[@]};

for ((a=0; a<$armProbs_len; a++)); do
    armProb=${armProbs[$a]}
    root_armProb=$simSetDescriptive/"num_sims="$numSims"armProb="$armProb
    echo $armProb
    for ((i=0; i<$arrayLength; i++)); do
        curN=${nsB[$i]}
        curEffectSize=${effectSizesB[$i]}
        root_armProb_es=$root_armProb/"es="$curEffectSize
        for ((j=0; j<$bsProps_len; j++)); do
            curProp=${bsProps[$j]}
            
            batch_size_fl=$(awk -v curN="${curN}" -v curProp="${curProp}" 'BEGIN{print (curN*curProp)}')
            batch_size=${batch_size_fl%.*}
            
            echo $batch_size
            burn_in_size=$batch_size
           # mkdir -p $root_dir

            #TS---------------
            root_armProb_es_prop_ts=$root_armProb_es"/bbUnEqualMeansEqualPriorburn_in_size-batch_size="$burn_in_size"-"$batch_size #root for this sim, split by equals for bs
          
            directoryName_ts=$root_armProb_es_prop_ts
            echo $directoryName_ts
            mkdir -p $directoryName_ts
            
            python3 run_effect_size_simulations_beta.py \
            $curEffectSize"-"$armProb $numSims $directoryName_ts 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
            echo $!
           
            #from equalmeans for reference
            #python3 louie_experiments/run_effect_size_simulations_beta.py \
            #$armProb,$armProb $numSims $directoryName_ts "Thompson" "armsEqual" $curN 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
            #echo $!

            #Uniform------------
            root_armProb_es_prop_unif=$root_armProb_es"/bbUnEqualMeansUniformburn_in_size-batch_size="$burn_in_size"-"$batch_size;
            directoryName_unif=$root_armProb_es_prop_unif
            mkdir -p $directoryName_unif
            python3 run_effect_size_simulations_beta.py \
            $curEffectSize"-"$armProb $numSims $directoryName_unif "uniform" 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
            
            #from equalmeans for reference
            #python3 louie_experiments/run_effect_size_simulations_beta.py \
            #$armProb,$armProb $numSims $directoryName_unif "uniform" "armsEqual" $curN 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
        done
    done
done

