
# Generate the point neuron only version of the network
python ISN.py -AllenCells    -neuroml -seed:100  # -perturbation assumed with -neuroml 


# Generate the network with 10 detailed cells
python ISN.py -Detailed_Soma -neuroml -seed:100  # -perturbation assumed with -neuroml
