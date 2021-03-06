#!/bin/bash

numSims=5000
numSims=10000
numSims=2000

simSetDescriptive="../simulation_saves/TSPPDNoEffect_c=0pt1" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDNoEffectTest" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDNoEffectResample" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDNoEffectResampleFastTest" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDNoEffectResampleFastRWCutoff" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDNoEffectResampleFast" #Give descriptive name for directory to save set of sims
simSetDescriptive="../simulation_saves/TSPPDNoEffectResampleFastFullHistory" #Give descriptive name for directory to save set of sims
#simSetDescriptive="../simulation_saves/TSPPDNoEffectResampleFastJeffPrior" #Give descriptive name for directory to save set of sims
mkdir -p $simSetDescriptive

#effectSizesB=(0.1 0.2 0.3 0.5); #switching now to 0.1,0.2,0.3,0.5
effectSizesB=(0.1 0.3 0.5); 
#effectSizesB=(0.3);

nsT=(394 64 26);
nsB=(785 657 88 32); #hard coded sample sizes corresponding to es
#nsB=(657); #hard coded sample sizes corresponding to es
nsB=(785); #All above are subset so only need this if saving fulll history
arrayLength=${#nsB[@]};

bsProps=(0.25);
bsProps_len=${#bsProps[@]};
armProbs=(0.5); #as default
armProbs=(0.25 0.5); 
armProbs=(0.5); #just for full history 
armProbs_len=${#armProbs[@]};

for ((a=0; a<$armProbs_len; a++)); do
    armProb=${armProbs[$a]}
    root_armProb=$simSetDescriptive/"num_sims="$numSims"armProb="$armProb
    echo $armProb

    #0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2

#    c_list=(0.1); #[0.025, 0.05, 0.1, 0.2, 0.3],  [0.08, 0.1, 0.12] 
#    c_list=(0.08 0.1 0.12); #[0.025, 0.05, 0.1, 0.2, 0.3],  [0.08, 0.1, 0.12] 
#    c_list=(0.025 0.05 0.1 0.2 0.3);
#    c_list=(0.025 0.05 0.075 0.1 0.125 0.15 0.2);
    c_list=(0.0);

    c_length=${#c_list[@]};
    for ((i=0; i<$arrayLength; i++)); do
        curN=${nsB[$i]}
        curEffectSize=${effectSizesB[$i]}
        for ((j=0; j<$c_length; j++)); do
        c=${c_list[$j]}

        #    root_armProb_es=$root_armProb/"N="$curN
        root_armProb_es=$root_armProb/"N="$curN"c="$c

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
        python3 run_effect_size_simulations_beta_fast_TSPPD.py \
        0.5,0.5 $numSims $directoryName_ts "Thompson" "armsEqual" $curN 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
        echo $!

        #Uniform------------
    #	root_armProb_es_prop_unif=$root_armProb_es"/bbEqualMeansUniformburn_in_size-batch_size="$burn_in_size"-"$batch_size;
    #	directoryName_unif=$root_armProb_es_prop_unif
    #	mkdir -p "$directoryName_unif"
        #python3 run_effect_size_simulations_beta_epsilon_greedy.py \
           # $curEffectSize"-"$armProb $numSims $directoryName_unif "uniform" 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
        
        #from equalmeans for reference
    #	python3 run_effect_size_simulations_beta_PPD_TS.py \
    #	0.5,0.5 $numSims $directoryName_unif "uniform" "armsEqual" $curN 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
        done
    done
done



