#!/bin/bash
set -e
# This script requires the repo https://github.com/OpenSourceBrain/SadehEtAl2017-InhibitionStabilizedNetworks 
# to be present in the directory ../SadehEtAl2017-InhibitionStabilizedNetworks

# This path is required for https://github.com/OpenSourceBrain/SadehEtAl2017-InhibitionStabilizedNetworks/blob/master/SpikingSimulationModels/defaultParams.py
export PYTHONPATH=PYTHONPATH:../SadehEtAl2017-InhibitionStabilizedNetworks/SpikingSimulationModels

for i in `seq 100 100 4000`;
do
    echo "==============================="
    echo "Running with seed: "$i
    python ISN.py -AllenCells -hyperA -netpyne -seed:$i
    python to_gdf.py
    python3 ../SadehEtAl2017-InhibitionStabilizedNetworks/PyNN/analysis_perturbation_pynn.py .9 1000 -nogui -average -legend
done 
