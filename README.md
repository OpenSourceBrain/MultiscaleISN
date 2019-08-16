# Multiscale ISN

Inhibition Stabilized Networks at multiple scales based on Sadeh et al. 2017. 

## To generate network/run model

The main script to generate the model is **[ISN.py](ISN.py)** and changing the parameters to the main 
generate() function here will create different configurations of the network:
```
def generate(scale_populations = 1,
             percentage_exc_detailed=0,
             exc2_cell = 'SmithEtAl2013/L23_NoHotSpot',
             percentage_inh_detailed=0,
             scalex=1,
             scaley=1,
             scalez=1,
             exc_exc_conn_prob = 0.25,
             exc_inh_conn_prob = 0.25,
             inh_exc_conn_prob = 0.75,
             inh_inh_conn_prob = 0.75,
             ee2_conn_prob = 0,
             ie2_conn_prob = 0,
             Bee = .1,
             Bei = .1,
             Bie = -.2,
             Bii = -.2,
             Bee2 = 1,
             Bie2 = -2,
             Be_bkg = .1,
             Be_stim = .1,
             r_bkg = 0,
             r_bkg_ExtExc=0,
             r_bkg_ExtInh=0,
             r_bkg_ExtExc2=0,
             r_stim = 0,
             fraction_inh_pert=0.75,
             fraction_inh_offset=0,
             inh_offset_amp=0,  # hyperpolarising/depolarising current to inh fraction_inh_offset of cells 
             Ttrans = 500, # transitent time to discard the data (ms)
             Tblank= 1500, # simulation time before perturbation (ms)
             Tstim = 1500, # simulation time of perturbation (ms)
             Tpost = 500, # simulation time after perturbation (ms)
             connections=True,
             connections2=False,
             exc_target_dendrites=False,
             inh_target_dendrites=False,
             duration = 1000,
             dt = 0.025,
             global_delay = .1,
             max_in_pop_to_plot_and_save = 10,
             format='xml',
             suffix='',
             run_in_simulator = None,
             num_processors = 1,
             target_dir='./temp/',
             v_clamp=False,
             simulation_seed=11111):       
```

Generally the defaults work well to generate a spiking network showing ISN properties. 

To generate the 2 main configurations of the network (point neurons only, point neurons + 10 detailed neurons) and save as NeuroML, run:

    ./regenerate_neuroml.sh

To run the 40 network simulations in NetPyNE for the point neuron network, run:

    ./runall.sh

To run the 40 network simulations in NetPyNE for the point neuron network with 10 detailed cells, run:

    ./runall_detailed.sh



[![Build Status](https://travis-ci.org/OpenSourceBrain/MultiscaleISN.svg?branch=master)](https://travis-ci.org/OpenSourceBrain/MultiscaleISN) 
[![DOI](https://www.zenodo.org/badge/136594034.svg)](https://www.zenodo.org/badge/latestdoi/136594034)

### Reusing this model

The code in this repository is provided under the terms of the [software license](LICENSE) included with it. If you use this model in your research, we respectfully ask you to cite the references outlined in the [CITATION](CITATION.md) file.

