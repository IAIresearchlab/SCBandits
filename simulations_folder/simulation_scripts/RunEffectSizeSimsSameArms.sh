#!/bin/bash
numSims=2
#variancesT=(5.0 2.0 1.25); # Note: we're not crossing nsT and variancesT because variancesT is actual variance with ES with that num steps

timestamp=$(date +%F_%T)

simSetDescriptive="../simulation_saves/TSNoEffectTEST" #Give descriptive name for directory to save set of sims
mkdir -p $simSetDescriptive


nsB=(785 88 32);
arrayLength=${#nsB[@]};
bsProps=(0.05 0.10 0.25);
bsProps_len=${#bsProps[@]};
armProbs=(0.2 0.5 0.8);
armProbs=(0.5);
armProbs_len=${#armProbs[@]};

for ((a=0; a<$armProbs_len; a++)); do
    armProb=${armProbs[$a]}
    root_armProb=$simSetDescriptive/"num_sims="$numSims"armProb="$armProb
    echo $armProb
    for ((i=0; i<$arrayLength; i++)); do
        curN=${nsB[$i]}
        root_armProb_n=$root_armProb/"n="$curN
        for ((j=0; j<$bsProps_len; j++)); do
            curProp=${bsProps[$j]}
            
            batch_size_fl=$(awk -v curN="${curN}" -v curProp="${curProp}" 'BEGIN{print (curN*curProp)}')
            batch_size=${batch_size_fl%.*}
            
            echo $batch_size
            burn_in_size=$batch_size
           # mkdir -p $root_dir

            #TS---------------
            root_armProb_n_prop_ts=$root_armProb_n"/bbEqualMeansEqualPriorburn_in_size-batch_size="$burn_in_size"-"$batch_size #root for this sim, split by equals for bs
          
            directoryName_ts=$root_armProb_n_prop_ts
            echo $directoryName_ts
            mkdir -p $directoryName_ts
            

            python3 run_effect_size_simulations_beta.py \
            $armProb,$armProb $numSims $directoryName_ts "Thompson" "armsEqual" $curN 2> $directoryName_ts"/errorOutput.log" > $directoryName_ts"/output.log" &
            echo $!


            #Uniform------------
            root_armProb_n_prop_unif=$root_armProb_n"/bbEqualMeansUniformburn_in_size-batch_size="$burn_in_size"-"$batch_size;
            directoryName_unif=$root_armProb_n_prop_unif
            mkdir -p $directoryName_unif
            python3 run_effect_size_simulations_beta.py \
            $armProb,$armProb $numSims $directoryName_unif "uniform" "armsEqual" $curN 2> $directoryName_unif"/errorOutput.log" > $directoryName_unif"/output.log" &
        done
    done
done


#Uniform -----------
# for ((a=0; a<$armProbs_len; a++)); do
#     armProb=${armProbs[$a]}
#     for ((i=0; i<$arrayLength; i++)); do
#         curN=${nsB[$i]}
#         for ((j=0; j<$bsProps_len; j++)); do
#             curProp=${bsProps[$j]}
            
#             batch_size_fl=$(awk -v curN="${curN}" -v curProp="${curProp}" 'BEGIN{print (curN*curProp)}')
#             batch_size=${batch_size_fl%.*}
#             burn_in_size=$batch_size
#             directoryName=$root_dir"/bbEqualMeansUniform"$curN"burn_in_size-batch_size="$burn_in_size"-"$batch_size;
#             echo $directoryName
#             mkdir $directoryName
#             python3 louie_experiments/run_effect_size_simulations_beta.py \
#             $armProb,$armProb $numSims $directoryName "uniform" "armsEqual" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#             echo $!
#         done
#     done
# done
# arrayLength=${#variancesT[@]};
# for ((i=0; i<$arrayLength; i++)); do
#     curVariance=${variancesT[$i]}
#     curN=${nsT[$i]}
#     directoryName="ngEqualMeansUniform"$curVariance;
#     echo $directoryName
#     mkdir $directoryName
#     python3 ~/banditalgorithms/src/louie_experiments/run_effect_size_simulations.py 0.0 0.0 $curVariance $numSims $directoryName "uniform" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#     echo $!
# done

# arrayLength=${#variancesT[@]};
# for ((i=0; i<$arrayLength; i++)); do
#     curVariance=${variancesT[$i]}
#     curN=${nsT[$i]}
#     directoryName="ngEqualMeansEqualPrior"$curVariance;
#     echo $directoryName
#     mkdir $directoryName
#     python3 ~/banditalgorithms/src/louie_experiments/run_effect_size_simulations.py 0.0 0.0 $curVariance $numSims $directoryName "Thompson" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#     echo $!
# done


# arrayLength=${#variancesT[@]};
# for ((i=0; i<$arrayLength; i++)); do
#     curVariance=${variancesT[$i]}
#     curN=${nsT[$i]}
#     directoryName="ngEqualMeansArmsHigh"$curVariance;
#     echo $directoryName
#     mkdir $directoryName
#     python3 ~/banditalgorithms/src/louie_experiments/run_effect_size_simulations.py 0.5 0.5 $curVariance $numSims $directoryName "Thompson" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#     echo $!
# done

# arrayLength=${#variancesT[@]};
# for ((i=0; i<$arrayLength; i++)); do
#     curVariance=${variancesT[$i]}
#     curN=${nsT[$i]}
#     directoryName="ngEqualMeansArmsLow"$curVariance;
#     echo $directoryName
#     mkdir $directoryName
#     python3 ~/banditalgorithms/src/louie_experiments/run_effect_size_simulations.py -0.5 -0.5 $curVariance $numSims $directoryName "Thompson" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#     echo $!
# done

# arrayLength=${#nsB[@]};
# for ((i=0; i<$arrayLength; i++)); do
#     curN=${nsB[$i]}
#     directoryName="bbEqualMeansUniform"$curN;
#     echo $directoryName
#     mkdir $directoryName
#     python3 ~/banditalgorithms/src/louie_experiments/run_effect_size_simulations_beta.py 0.5,0.5 $numSims $directoryName "uniform" "armsEqual" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#     echo $!
# done


#/Users/jacobnogas/Documents/School/Summer_2019/IPW\ Sofia\ Villar/banditalgorithms/src/
# arrayLength=${#nsB[@]};
# for ((i=0; i<$arrayLength; i++)); do
#     curN=${nsB[$i]}
#     directoryName="bbEqualMeansArmsHigh"$curN;
#     echo $directoryName
#     mkdir $directoryName
#     python3 ~/banditalgorithms/src/louie_experiments/run_effect_size_simulations_beta.py 0.5,0.5 $numSims $directoryName "Thompson" "armsHigh" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#     echo $!
# done

# arrayLength=${#nsB[@]};
# for ((i=0; i<$arrayLength; i++)); do
#     curN=${nsB[$i]}
#     directoryName="bbEqualMeansArmsLow"$curN;
#     echo $directoryName
#     mkdir $directoryName
#     python3 ~/banditalgorithms/src/louie_experiments/run_effect_size_simulations_beta.py 0.5,0.5 $numSims $directoryName "Thompson" "armsLow" $curN 2> $directoryName"/errorOutput.log" > $directoryName"/output.log" &
#     echo $!
# done



