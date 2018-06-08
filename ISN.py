# -*- coding: utf-8 -*-

'''
Generates a NeuroML 2 file with many types of cells, populations and inputs
for testing purposes
'''

import opencortex.core as oc
import shutil
import opencortex.utils.color as occ

import numpy as np
import pylab as pl
import os, sys, math, pickle

from pyelectro import analysis
from pyneuroml import pynml

min_pop_size = 1

exc_color = occ.L23_PRINCIPAL_CELL
exc2_color = occ.L23_PRINCIPAL_CELL_2
inh_color = occ.L23_INTERNEURON
inh_color = occ.L23_INTERNEURON_2

exc_color = '1 0 0'
exc2_color = '0 1 0'
inh_color = '0 0 0.9'
inh2_color = '1 0 1'

exc_inh_fraction = .8



# ---
# default values
exc_exc_conn_prob = 0.15 
exc_inh_conn_prob = 0.15 
inh_exc_conn_prob = .5  
inh_inh_conn_prob = .5  

Bee = .6
Bei = .5
Bie = 1 
Bii = 1 

Be_stim = Be_bkg = 0.5

Ttrans = 500.
Tblank = 500. 
Tstim = 500.
Tpost = 500.

#r_bkg_ExtExc=1
#r_bkg_ExtInh=1
r_bkg = 1
dt = 0.025

r_bkg_ExtExc = 4000
r_bkg_ExtInh = 3600
r_stim = -180
r_bkg_ExtExc2 = 20e3

percentage_exc_detailed = 0
exc_target_dendrites = 0
inh_target_dendrites = 0
fraction_inh_pert_rng = [0.5]
ee2_conn_prob = 0
ie2_conn_prob = 0
v_clamp = False
#duration_clamp = 500

def scale_pop_size(baseline, scale):
    return max(min_pop_size, int(baseline*scale))

def generate(scale_populations = 1,
             percentage_exc_detailed=0,
             #exc2_cell = 'SmithEtAl2013/L23_Retuned_477127614',
             exc2_cell = 'SmithEtAl2013/L23_NoHotSpot',
             #exc2_cell = 'BBP/cADpyr229_L23_PC_5ecbf9b163_0_0',
             #exc2_cell = 'BBP/cNAC187_L23_NBC_9d37c4b1f8_0_0',
             #exc2_cell = 'Thalamocortical/L23PyrRS',
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
                 
    reference = ("ISN_net%s"%(suffix)).replace('.','_')
    
    ks = open('kernelseed','w')
    ks.write('%i'%simulation_seed)
    ks.close()
    
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    
    info=('  Generating ISN network: %s\n'%reference)
    info+=('    Duration: %s; dt: %s; scale: %s; simulator: %s (num proc. %s)\n'%(duration, dt, scale_populations, run_in_simulator, num_processors))
    info+=('    Bee: %s; Bei: %s; Bie: %s; Bii: %s\n'%(Bee,Bei,Bie,Bii))
    info+=('    Bkg exc at %sHz\n'%(r_bkg_ExtExc))
    info+=('    Bkg inh at %sHz\n'%(r_bkg_ExtInh))
    info+=('    Be_stim: %s at %sHz (i.e. %sHz for %s perturbed I cells)\n'%(Be_stim,r_stim, r_bkg_ExtInh+r_stim, fraction_inh_pert))
    info+=('    Inh offset: %spA for %s I cells with offset current\n'%(inh_offset_amp, fraction_inh_offset))
    info+=('    Exc detailed: %s%% - Inh detailed %s%%\n'%(percentage_exc_detailed,percentage_inh_detailed))
    info+=('    Seed: %s'%(simulation_seed))
    
    print('-------------------------------------------------')
    print(info)
    print('-------------------------------------------------')
                    

    num_exc = scale_pop_size(np.round(100*exc_inh_fraction),scale_populations)
    num_exc2  = int(math.ceil(num_exc*percentage_exc_detailed/100.0))
    num_exc -= num_exc2
    
    num_inh = scale_pop_size(np.round(100*(1-exc_inh_fraction)),scale_populations)
    num_inh2  = int(math.ceil(num_inh*percentage_inh_detailed/100.0))
    num_inh -= num_inh2
    
    nml_doc, network = oc.generate_network(reference, network_seed=simulation_seed)
    nml_doc.notes=info
    network.notes=info
    
    #exc_cell_id = 'AllenHH_480351780'
    #exc_cell_id = 'AllenHH_477127614'
    #exc_cell_id = 'HH_477127614'
    exc_cell_id = 'HH2_477127614'
    exc_type = exc_cell_id.split('_')[0]
    oc.include_neuroml2_cell_and_channels(nml_doc, 'cells/%s/%s.cell.nml'%(exc_type,exc_cell_id), exc_cell_id)
    
    
    #inh_cell_id = 'AllenHH_485058595'
    #inh_cell_id = 'AllenHH_476686112'
    #inh_cell_id = 'AllenHH_477127614'
    #inh_cell_id = 'HH_476686112'
    inh_cell_id = 'HH2_476686112'
    inh_type = exc_cell_id.split('_')[0]
    oc.include_neuroml2_cell_and_channels(nml_doc, 'cells/%s/%s.cell.nml'%(inh_type,inh_cell_id), inh_cell_id)

    if percentage_exc_detailed>0:
        exc2_cell_id = exc2_cell.split('/')[1]
        exc2_cell_dir = exc2_cell.split('/')[0]
        oc.include_neuroml2_cell_and_channels(nml_doc, 'cells/%s/%s.cell.nml'%(exc2_cell_dir,exc2_cell_id), exc2_cell_id)

    if percentage_inh_detailed>0:
        inh2_cell_id = 'cNAC187_L23_NBC_9d37c4b1f8_0_0'
        oc.include_neuroml2_cell_and_channels(nml_doc, 'cells/BBP/%s.cell.nml'%inh2_cell_id, inh2_cell_id)
    

    xDim = 700*scalex
    yDim = 100*scaley
    yDimExc2 = 50*scaley
    zDim = 700*scalez

    xs = -1*xDim/2
    ys = -1*yDim/2
    zs = -1*zDim/2

    #####   Synapses
    
    synAmpaEE = oc.add_exp_one_syn(nml_doc, id="ampaEE", gbase="%snS"%Bee,
                             erev="0mV", tau_decay="1ms")
    synAmpaEI = oc.add_exp_one_syn(nml_doc, id="ampaEI", gbase="%snS"%Bei,
                             erev="0mV", tau_decay="1ms")

    synGabaIE = oc.add_exp_one_syn(nml_doc, id="gabaIE", gbase="%snS"%abs(Bie),
                             erev="-80mV", tau_decay="2ms")
    synGabaII = oc.add_exp_one_syn(nml_doc, id="gabaII", gbase="%snS"%abs(Bii),
                             erev="-80mV", tau_decay="2ms")

    synAmpaBkg = oc.add_exp_one_syn(nml_doc, id="ampaBkg", gbase="%snS"%Be_bkg,
                             erev="0mV", tau_decay="1ms")
    #synAmpaStim = oc.add_exp_one_syn(nml_doc, id="ampaStim", gbase="%snS"%Be_stim,
    #                         erev="0mV", tau_decay="1ms")

    synAmpaEE2 = oc.add_exp_one_syn(nml_doc, id="ampaEE2", gbase="%snS"%Bee2,
                             erev="0mV", tau_decay="10ms")
    synGabaIE2 = oc.add_exp_one_syn(nml_doc, id="gabaIE2", gbase="%snS"%abs(Bie2),
                             erev="-80mV", tau_decay="10ms")

    #####   Input types

    '''tpfsA = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpsfA",
                                       average_rate="%s Hz"%r_bkg,
                                       delay = '0ms', 
                                       duration = '%sms'%(Ttrans+Tblank),
                                       synapse_id=synAmpaBkg.id)

    tpfsB = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpsfB",
                                       average_rate="%s Hz"%r_bkg,
                                       delay = '%sms'%(Ttrans+Tblank),
                                       duration = '%sms'%(Tstim),
                                       synapse_id=synAmpaBkg.id)

    tpfsC = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpsfC",
                                       average_rate="%s Hz"%(r_bkg+r_stim),
                                       delay = '%sms'%(Ttrans+Tblank),
                                       duration = '%sms'%(Tstim),
                                       synapse_id=synAmpaStim.id)'''

    tpfsExtExc = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpfsExtExc",
                                       average_rate="%s Hz"%r_bkg_ExtExc,
                                       delay = '0ms', 
                                       duration = '%sms'%(Ttrans+Tblank+Tstim+Tpost),
                                       synapse_id=synAmpaBkg.id)
    
    tpfsExtExc2 = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpfsExtExc2",
                                       average_rate="%s Hz"%r_bkg_ExtExc2,
                                       delay = '0ms', 
                                       duration = '%sms'%(Ttrans+Tblank+Tstim+Tpost),
                                       synapse_id=synAmpaBkg.id)

    tpfsExtInh = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpfsExtInh",
                                       average_rate="%s Hz"%r_bkg_ExtInh,
                                       delay = '0ms', 
                                       duration = '%sms'%(Ttrans+Tblank+Tstim+Tpost),
                                       synapse_id=synAmpaBkg.id)
    
    tpfsPertInh_before = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpfsPertInh_before",
                                       average_rate="%s Hz"%r_bkg_ExtInh,
                                       delay = '0ms', 
                                       duration = '%sms'%(Ttrans+Tblank),
                                       synapse_id=synAmpaBkg.id)
    tpfsPertInh_during = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpfsPertInh_during",
                                       average_rate="%s Hz"%(r_bkg_ExtInh+r_stim),
                                       delay = '%sms'%(Ttrans+Tblank), 
                                       duration = '%sms'%(Tstim),
                                       synapse_id=synAmpaBkg.id)
    tpfsPertInh_after = oc.add_transient_poisson_firing_synapse(nml_doc,
                                       id="tpfsPertInh_after",
                                       average_rate="%s Hz"%r_bkg_ExtInh,
                                       delay = '%sms'%(Ttrans+Tblank+Tstim), 
                                       duration = '%sms'%(Tpost),
                                       synapse_id=synAmpaBkg.id)
                                       
    if fraction_inh_offset>0:
        inh_hyper = oc.add_pulse_generator(nml_doc,
                       id="inh_hyper",
                       delay='%sms'%(Ttrans+Tblank),
                       duration='%sms'%(Tstim),
                       amplitude="%spA"%inh_offset_amp)


    #####   Populations

    popExc = oc.add_population_in_rectangular_region(network,
                                                  'popExc',
                                                  exc_cell_id,
                                                  num_exc,
                                                  xs,ys,zs,
                                                  xDim,yDim,zDim,
                                                  color=exc_color)  
    from neuroml import Property
    popExc.properties.append(Property('type','E'))
    allExc = [popExc]

    if num_exc2>0:
        popExc2 = oc.add_population_in_rectangular_region(network,
                                                  'popExc2',
                                                  exc2_cell_id,
                                                  num_exc2,
                                                  xs,yDim/2,zs,
                                                  xDim,yDimExc2,zDim,
                                                  color=exc2_color)
        popExc2.properties.append(Property('type','E'))
                                                  
        allExc.append(popExc2)

    popInh = oc.add_population_in_rectangular_region(network,
                                                  'popInh',
                                                  inh_cell_id,
                                                  num_inh,
                                                  xs,ys,zs,
                                                  xDim,yDim,zDim,
                                                  color=inh_color)     
    popInh.properties.append(Property('type','I'))     
    allInh = [popInh]
    
    if num_inh2>0:
        popInh2 = oc.add_population_in_rectangular_region(network,
                                                  'popInh2',
                                                  inh2_cell_id,
                                                  num_inh2,
                                                  xs,ys,zs,
                                                  xDim,yDim,zDim,
                                                  color=inh2_color)
                                                  
        allInh.append(popInh2)


    #####   Projections

    if connections:
        
        weight_expr = 'abs(normal(1,0.5))'
        
        for popEpr in allExc:
            
            for popEpo in allExc:
                proj = add_projection(network, "projEE",
                                      popEpr, popEpo,
                                      synAmpaEE.id, exc_exc_conn_prob, 
                                      global_delay,
                                      exc_target_dendrites,
                                      weight_expr)
                                                
            for popIpo in allInh:
                proj = add_projection(network, "projEI",
                                      popEpr, popIpo,
                                      synAmpaEI.id, exc_inh_conn_prob, 
                                      global_delay,
                                      exc_target_dendrites,
                                      weight_expr)

            
        for popIpr in allInh:
            
            for popEpo in allExc:
                proj = add_projection(network, "projIE",
                                      popIpr, popEpo,
                                      synGabaIE.id, inh_exc_conn_prob, 
                                      global_delay,
                                      inh_target_dendrites,
                                      weight_expr)
        
            for popIpo in allInh:
                proj = add_projection(network, "projII",
                                      popIpr, popIpo,
                                      synGabaII.id, inh_inh_conn_prob, 
                                      global_delay,
                                      inh_target_dendrites,
                                      weight_expr)

    elif connections2:
        
        weight_expr = 'abs(normal(1,0.5))'
        
        proj = add_projection(network, "projEE",
                                      popExc, popExc,
                                      synAmpaEE.id, exc_exc_conn_prob, 
                                      global_delay,
                                      exc_target_dendrites,
                                      weight_expr)
        proj = add_projection(network, "projEI",
                                      popExc, popInh,
                                      synAmpaEI.id, exc_inh_conn_prob, 
                                      global_delay,
                                      exc_target_dendrites,
                                      weight_expr)
        proj = add_projection(network, "projIE",
                                      popInh, popExc,
                                      synGabaIE.id, inh_exc_conn_prob, 
                                      global_delay,
                                      inh_target_dendrites,
                                      weight_expr)
        proj = add_projection(network, "projII",
                                      popInh, popInh,
                                      synGabaII.id, inh_inh_conn_prob, 
                                      global_delay,
                                      inh_target_dendrites,
                                      weight_expr)

        proj = add_projection(network, "projEE2",
                                      popExc, popExc2,
                                      synAmpaEE2.id, ee2_conn_prob, 
                                      global_delay,
                                      exc_target_dendrites,
                                      weight_expr)   
        proj = add_projection(network, "projIE2",
                                      popInh, popExc2,
                                      synGabaIE2.id, ie2_conn_prob, 
                                      global_delay,
                                      inh_target_dendrites,
                                      weight_expr)      

    #####   Inputs

    oc.add_inputs_to_population(network, "Stim_E",
                                    popExc, tpfsExtExc.id,
                                    all_cells=True)

    if num_exc2>0:
        oc.add_inputs_to_population(network, "Stim_E2",
                                        popExc2, tpfsExtExc2.id,
                                        all_cells=True)

    num_inh_pert = int(popInh.get_size()*fraction_inh_pert)

    oc.add_inputs_to_population(network, "Stim_I_nonpert",
                                    popInh, tpfsExtInh.id,
                                    all_cells=False,
                                    only_cells=range(num_inh_pert,popInh.get_size()))

    oc.add_inputs_to_population(network, "Stim_I_pert_before",
                                    popInh, tpfsPertInh_before.id,
                                    all_cells=False,
                                    only_cells=range(0,num_inh_pert))  
    oc.add_inputs_to_population(network, "Stim_I_pert_during",
                                    popInh, tpfsPertInh_during.id,
                                    all_cells=False,
                                    only_cells=range(0,num_inh_pert))
    oc.add_inputs_to_population(network, "Stim_I_pert_after",
                                    popInh, tpfsPertInh_after.id,
                                    all_cells=False,
                                    only_cells=range(0,num_inh_pert))
                                    
                               
    if fraction_inh_offset>0:              
        num_inh_offset = int(popInh.get_size()*fraction_inh_offset)

        oc.add_inputs_to_population(network, "Stim_I_offset",
                                        popInh, inh_hyper.id,
                                        all_cells=False,
                                        only_cells=range(0,num_inh_offset))

    # injecting noise in the soma of detailed neurons to insert some variability
    '''oc.add_targeted_inputs_to_population(network, 
                                         "PG_noise",
                                         popExc2, 
                                         'noisyCurrentSource1',             # from ../../../NoisyCurrentSource.xml
                                         segment_group='soma_group',
                                         number_per_cell = 1,
                                         all_cells=True)
    
    
    oc.add_inputs_to_population(network, "Stim_pre_ExtExc_%s"%popExc.id,
                                    popExc, tpfsExtExc.id,
                                    all_cells=True)

    for pop in allExc:
        #oc.add_inputs_to_population(network, "Stim_pre_ExtExc_%s"%pop.id,
        #                            pop, tpfsExtExc.id,
        #                            all_cells=True)

        oc.add_inputs_to_population(network, "Stim_pre_%s"%pop.id,
                                    pop, tpfsA.id,
                                    all_cells=True)
        
        oc.add_inputs_to_population(network, "Stim_E_%s"%pop.id,
                                    pop, tpfsB.id,
                                    all_cells=True) 

    for pop in allInh:
        num_inh_pert = int(pop.get_size()*fraction_inh_pert)

        oc.add_inputs_to_population(network, "Stim_pre_ExtInh_%s"%pop.id,
                                    pop, tpfsExtInh.id,
                                    all_cells=True)

        oc.add_inputs_to_population(network, "Stim_pre_%s"%pop.id,
                                    pop, tpfsA.id,
                                    all_cells=True)
        
        oc.add_inputs_to_population(network, "Stim_I_pert_%s"%pop.id,
                                    pop, tpfsC.id,
                                    all_cells=False,
                                    only_cells=range(0,num_inh_pert))   
                                    
        oc.add_inputs_to_population(network, "Stim_I_nonpert_%s"%pop.id,
                                    pop, tpfsB.id,
                                    all_cells=False,
                                    only_cells=range(num_inh_pert,pop.get_size()))  '''
    

    save_v = {}
    plot_v = {}

    # Work in progress...
    # General idea: clamp one (or more) exc cell at rev pot of inh syn and see only exc inputs
    #
    if v_clamp:
        
        levels = {'IPSC': synAmpaEE.erev, 'EPSC':synGabaIE.erev}
        
        for l in levels:
            cell_index = levels.keys().index(l)

            pop = 'popExc2'
            plot = 'IClamp_i_%s'%(l)
            
            for seg_id in [0,2953, 1406]: # 2953: end of axon; 1406 on dendrite
            
                clamp_id = "vclamp_cell%s_seg%s_%s"%(cell_index,seg_id,l)
                v_clamped = levels[l]

                vc = oc.add_voltage_clamp_triple(nml_doc, id=clamp_id, 
                                     delay='0ms', 
                                     duration='%sms'%duration, 
                                     conditioning_voltage=v_clamped,
                                     testing_voltage=v_clamped,
                                     return_voltage=v_clamped, 
                                     simple_series_resistance="1e2ohm",
                                     active = "1")

                vc_dat_file = 'v_clamps_i_seg%s_%s.%s.dat'%(seg_id,l,simulation_seed)
                
                seg_file = '%s_seg%s_%s_v.dat'%(pop,seg_id,l)
                
                save_v[vc_dat_file] = []
                plot_v[plot] = []

                oc.add_inputs_to_population(network, "vclamp_seg%s_%s"%(seg_id, l),
                                            network.get_by_id(pop), vc.id,
                                            all_cells=False,
                                            only_cells=[cell_index],
                                            segment_ids=[seg_id])     

                # record at seg
                q = '%s/%s/%s/%s/%s/i'%(pop, cell_index,network.get_by_id(pop).component,seg_id,clamp_id)
                
                save_v[vc_dat_file].append(q)
                plot_v[plot].append(q)
                
                if seg_id!=0:
                    save_v[seg_file] = []
                    q = '%s/%s/%s/%s/v'%(pop, cell_index,network.get_by_id(pop).component,seg_id)
                    save_v[seg_file].append(q)
                

    #####   Save NeuroML and LEMS Simulation files      
    
    nml_file_name = '%s.net.%s'%(network.id,'nml.h5' if format == 'hdf5' else 'nml')
    oc.save_network(nml_doc, 
                    nml_file_name, 
                    validate=(format=='xml'),
                    format = format,
                    target_dir=target_dir)
        
    print("Saved to: %s"%nml_file_name)
    
    if num_exc>0:
        exc_traces = '%s_%s_v.dat'%(network.id,popExc.id)
        save_v[exc_traces] = []
        plot_v[popExc.id] = []

    if num_inh>0:
        inh_traces = '%s_%s_v.dat'%(network.id,popInh.id)
        save_v[inh_traces] = []
        plot_v[popInh.id] = []

    if num_exc2>0:
        exc2_traces = '%s_%s_v.dat'%(network.id,popExc2.id)
        save_v[exc2_traces] = []
        plot_v[popExc2.id] = []

    if num_inh2>0:
        inh2_traces = '%s_%s_v.dat'%(network.id,popInh2.id)
        save_v[inh2_traces] = []
        plot_v[popInh2.id] = []


    for i in range(min(max_in_pop_to_plot_and_save,num_exc)):
        plot_v[popExc.id].append("%s/%i/%s/v"%(popExc.id,i,popExc.component))
        save_v[exc_traces].append("%s/%i/%s/v"%(popExc.id,i,popExc.component))

    for i in range(min(max_in_pop_to_plot_and_save,num_exc2)):
        plot_v[popExc2.id].append("%s/%i/%s/v"%(popExc2.id,i,popExc2.component))
        save_v[exc2_traces].append("%s/%i/%s/v"%(popExc2.id,i,popExc2.component))

    for i in range(min(max_in_pop_to_plot_and_save,num_inh)):
        plot_v[popInh.id].append("%s/%i/%s/v"%(popInh.id,i,popInh.component))
        save_v[inh_traces].append("%s/%i/%s/v"%(popInh.id,i,popInh.component))

    for i in range(min(max_in_pop_to_plot_and_save,num_inh2)):
        plot_v[popInh2.id].append("%s/%i/%s/v"%(popInh2.id,i,popInh2.component))
        save_v[inh2_traces].append("%s/%i/%s/v"%(popInh2.id,i,popInh2.component))

    gen_spike_saves_for_all_somas = True

    lems_file_name, lems_sim = oc.generate_lems_simulation(nml_doc, network, 
                            target_dir+nml_file_name, 
                            duration =      duration, 
                            dt =            dt,
                            gen_plots_for_all_v = False,
                            gen_plots_for_quantities = plot_v,
                            gen_saves_for_all_v = False,
                            gen_saves_for_quantities = save_v,
                            gen_spike_saves_for_all_somas = gen_spike_saves_for_all_somas,
                            target_dir=target_dir,
                            include_extra_lems_files = ['./NoisyCurrentSource.xml'],
                            report_file_name='report.txt',
                            simulation_seed=simulation_seed)


    if run_in_simulator:

        print ("Running %s for %sms in %s"%(lems_file_name, duration, run_in_simulator))

        traces, events = oc.simulate_network(lems_file_name,
                 run_in_simulator,
                 max_memory='4000M',
                 nogui=True,
                 load_saved_data=True,
                 reload_events=True,
                 plot=False,
                 verbose=True,
                 num_processors=num_processors)


        print("Reloaded traces: %s"%traces.keys())
        #print("Reloaded events: %s"%events.keys())

        use_events_for_rates = False

        exc_rate = 0
        inh_rate = 0

        if use_events_for_rates:
            if (run_in_simulator=='jNeuroML_NetPyNE'):
                raise('Saving of spikes (and so calculation of rates) not yet supported in jNeuroML_NetPyNE')
            for ek in events.keys():
                rate = 1000 * len(events[ek])/float(duration)
                print("Cell %s has a rate %s Hz"%(ek,rate))
                if 'popExc' in ek:
                    exc_rate += rate/num_exc
                if 'popInh' in ek:
                    inh_rate += rate/num_inh

        else:
            tot_exc_rate = 0 
            exc_cells = 0
            tot_inh_rate = 0 
            inh_cells = 0
            tt = [t*1000 for t in traces['t']]
            for tk in traces.keys():
                if tk!='t':
                    rate = get_rate_from_trace(tt,[v*1000 for v in traces[tk]])
                    print("Cell %s has rate %s Hz"%(tk,rate))
                    if 'popExc' in tk:
                        tot_exc_rate += rate
                        exc_cells+=1
                    if 'popInh' in tk:
                        tot_inh_rate += rate
                        inh_cells+=1

            exc_rate = tot_exc_rate/exc_cells
            inh_rate = tot_inh_rate/inh_cells

        print("Run %s: Exc rate: %s Hz; Inh rate %s Hz"%(reference,exc_rate, inh_rate))

        return exc_rate, inh_rate, traces
                        
    return nml_doc, nml_file_name, lems_file_name
                        
                        
def add_projection(network, 
                   proj_id,
                   pop_pre, 
                   pop_post,
                   syn_id,
                   conn_prob,
                   delay,
                   target_dendrites,
                   weight_expr=1):


    if pop_post.size > pop_pre.size:
        num_connections = pop_pre.size * conn_prob
        targeting_mode='convergent'
    else:
        num_connections = pop_post.size * conn_prob
        targeting_mode='divergent'

    post_segment_group = 'soma_group'
    
    if '2' in pop_post.id and target_dendrites:
        post_segment_group = 'dendrite_group'
        

    proj = oc.add_targeted_projection(network,
                                    proj_id,
                                    pop_pre,
                                    pop_post,
                                    targeting_mode=targeting_mode,
                                    synapse_list=[syn_id],
                                    pre_segment_group = 'soma_group',
                                    post_segment_group = post_segment_group,
                                    number_conns_per_cell=num_connections,
                                    delays_dict = {syn_id:delay},
                                    weights_dict = {syn_id:weight_expr})
    return proj
       
def get_rate_from_trace(times, volts):

    analysis_var={'peak_delta':0,'baseline':0,'dvdt_threshold':0, 'peak_threshold':0}

    try:
        analysis_data=analysis.IClampAnalysis(volts,
                                           times,
                                           analysis_var,
                                           start_analysis=0,
                                           end_analysis=times[-1],
                                           smooth_data=False,
                                           show_smoothed_data=False)

        analysed = analysis_data.analyse()

        #pp.pprint(analysed)

        return analysed['mean_spike_frequency']
    
    except:
        return 0

                         
def _plot_(X, E, I, sbplt=111, ttl=[]):
    ax = pl.subplot(sbplt)
    pl.title(ttl)
    pl.imshow(X, origin='lower', interpolation='none')
    pl.xlabel('Exc weight')
    pl.ylabel('Inh weight')
    ax.set_xticks(range(0,len(E))); ax.set_xticklabels(E)
    ax.set_yticks(range(0,len(I))); ax.set_yticklabels(I)
    pl.colorbar()

def run_one(**kwargs):
    
    print('============================================================= \n     run_one: %s'%kwargs)
    
    format = 'hdf5'
    #format = 'xml'

    run_in_simulator = 'jNeuroML_NetPyNE'
    num_processors = 16
    
    simtag = 'AllenCells'

    fraction_inh_pert = 0.9
    scale_populations = 10 #x100 total N

    connections = 1
    # recurrent connection between exc-inh + extra connections from exc and inh pops to exc2
    connections2 = 0
    
    suffix = '';#str(int(fraction_inh_pert*100))
    target_dir = './';
        
    generate(Bee = kwargs['Bee'],
                    Bei = kwargs['Bei'],
                    Bie = kwargs['Bie'],
                    Bii = kwargs['Bii'],
                    Be_bkg = Be_bkg,
                    Be_stim = Be_stim,
                    r_bkg = 0,
                    r_stim = kwargs['r_stim'],
                    r_bkg_ExtExc=kwargs['r_bkg_ExtExc'],
                    r_bkg_ExtInh=kwargs['r_bkg_ExtInh'],
                    r_bkg_ExtExc2=r_bkg_ExtExc2,
                    Ttrans = Ttrans,
                    Tblank= Tblank,
                    Tstim = Tstim,
                    Tpost = Tpost,
                    exc_exc_conn_prob = exc_exc_conn_prob,
                    exc_inh_conn_prob = exc_inh_conn_prob,
                    inh_exc_conn_prob = inh_exc_conn_prob,
                    inh_inh_conn_prob = inh_inh_conn_prob,
                    ee2_conn_prob = ee2_conn_prob,
                    ie2_conn_prob = ie2_conn_prob,
                    connections=connections, connections2=connections2,
                    fraction_inh_pert=fraction_inh_pert,
                    duration = Ttrans+Tblank+Tstim+Tpost,
                    dt = kwargs['dt'],
                    scale_populations=scale_populations,
                    format=format,
                    percentage_exc_detailed=percentage_exc_detailed,
                    target_dir=target_dir,
                    suffix=suffix,
                    run_in_simulator=run_in_simulator,
                    num_processors=num_processors,
                    exc_target_dendrites=exc_target_dendrites,
                    inh_target_dendrites=inh_target_dendrites,
                    v_clamp=v_clamp,
                    simulation_seed=kwargs['simulation_seed'])

if __name__ == '__main__':
    
    run_in_simulator = None
    format = 'hdf5'
    #format = 'xml'

    num_processors = 1
    if '-neuron' in sys.argv: 
        run_in_simulator = 'jNeuroML_NEURON'

    if '-netpyne' in sys.argv: 
        run_in_simulator = 'jNeuroML_NetPyNE'
        num_processors = 10
    
    
    simulation_seed = np.random.randint(1,5555)
    
    for a in sys.argv:
        if a.startswith('-seed:'):
            simulation_seed = int(a[6:])

    if '-test' in sys.argv:  
        simtag = 'test'
        
        r_bkg = 10000.
        r_stim = -200
        
        Be_bkg = 0.5
        Be_stim = Be_bkg

        exc_exc_conn_prob = 0.25
        exc_inh_conn_prob = 0.25
        inh_exc_conn_prob = 0.75
        inh_inh_conn_prob = 0.75
        
        ee2_conn_prob = 4

        scale_populations = .1#0
        
        percentage_exc_detailed = 0#2.5

        Bee = .5e-5
        Bei = .5e-5
        Bie = 1e-5
        Bii = 1e-5
        
        format='xml'
        
        v_clamp= False
        exc_target_dendrites =True
        inh_target_dendrites = 1
        
        Ttrans = 100.
        Tblank= 100.
        Tstim = 100.
        Tpost = 100.
        
        connections = 1
        # recurrent connection between exc-inh + extra connections from exc and inh pops to exc2
        connections2 = 0
    
    elif '-AllenCells' in sys.argv:
        simtag = 'AllenCells'

        fraction_inh_pert_rng = [0.9]
        scale_populations = 10 #x100 total N

        connections = 1
        # recurrent connection between exc-inh + extra connections from exc and inh pops to exc2
        connections2 = 0

    elif '-Detailed_Soma' in sys.argv:
        simtag = 'Detailed_Soma'
        
        v_clamp = False
        v_clamp = True

        fraction_inh_pert_rng = [0.9]
        
        scale_populations = 10
        
        percentage_exc_detailed = 1.25
        exc_target_dendrites = 1
        inh_target_dendrites = 1

        connections = 0
        # recurrent connection between exc-inh + extra connections from exc and inh pops to exc2
        connections2 = 1

        ee2_conn_prob = .2#4
        ie2_conn_prob = 1.5#1

    N = scale_populations*100
    NE = int(exc_inh_fraction*N); NI=N-NE
    NE_detailed = int(percentage_exc_detailed*NE/100)
    NE_point = NE-NE_detailed

    sim_id = simtag+'_N'+str(int(N))

    # --
    if '-rheobase' in sys.argv:
        #rins = np.array([2.5,3,3.5])*1e3
        rins = np.array([0,1,2,3])*300    

        results = {}
        results['r_in'] = rins
        r_out_exc, r_out_exc2, r_out_inh = [], [], []
        for rin in rins:
            suffix = '';#str(int(fraction_inh_pert*100))
            target_dir = './temp/';
            [exc_rate, inh_rate, traces] = generate(Bee = 1e-5,
                                                    Bei = 1e-5,
                                                    Bie = 1e-5,
                                                    Bii = 1e-5,
                                                    Be_bkg = Be_bkg,
                                                    Be_stim = Be_stim,
                                                    r_bkg = 1,
                                                    r_stim = 1,
                                                    r_bkg_ExtExc=rin+14e3,
                                                    r_bkg_ExtInh=rin+5e3,
                                                    Ttrans = Ttrans,
                                                    Tblank= Tblank,
                                                    Tstim = Tstim,
                                                    exc_exc_conn_prob = 0,
                                                    exc_inh_conn_prob = 0,
                                                    inh_exc_conn_prob = 0,
                                                    inh_inh_conn_prob = 0,
                                                    fraction_inh_pert=0,
                                                    duration = Ttrans+Tblank+Tstim,
                                                    dt = dt,
                                                    scale_populations=scale_populations,
                                                    format=format,
                                                    percentage_exc_detailed=percentage_exc_detailed,
                                                    target_dir=target_dir,
                                                    suffix=suffix,
                                                    run_in_simulator=run_in_simulator,
                                                    num_processors=num_processors,
                                                    simulation_seed=simulation_seed)

            #
            T = Ttrans+Tblank+Tstim

            if NE_point != 0:
                exc_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popExc.spikes')
                spt_exc = exc_data[:,1]; spi_exc = exc_data[:,0];
                exc_rate = len(spt_exc[spt_exc>(Ttrans/1e3)]) /((Tblank+Tstim)/1e3) /NE_point 
            else: spt_exc = []; spi_exc = []; exc_rate = []
             
            if NE_detailed != 0:
                exc2_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popExc2.spikes')
                spt_exc2 = exc2_data[:,1]; spi_exc2 = exc2_data[:,0]; 
                exc2_rate = len(spt_exc2[spt_exc2>(Ttrans/1e3)]) /((Tblank+Tstim)/1e3) /NE_detailed
            else: spt_exc2 = []; spi_exc2 = []; exc2_rate = []  

            inh_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popInh.spikes')
            spt_inh = inh_data[:,1]; spi_inh = inh_data[:,0];
            inh_rate = len(spt_inh[spt_inh>(Ttrans/1e3)]) /((Tblank+Tstim)/1e3) /NI

            r_out_exc.append(exc_rate)
            r_out_exc2.append(exc2_rate)
            r_out_inh.append(inh_rate)
        
        results['r_out_exc'] = np.array(r_out_exc)
        results['r_out_exc2'] = np.array(r_out_exc2)
        results['r_out_inh'] = np.array(r_out_inh)

        fl = open(target_dir+'rheobase__'+sim_id+'.res', 'wb'); pickle.dump(results, fl); fl.close()
    
    elif '-gains' in sys.argv:
        #rins = np.array([2.5,3,3.5])*1e3
        rins_exc = np.array([0,1,2,3])*300 +14e3
        rins_inh = np.array([0,1,2,3])*300 +5e3

        results = {}
        results['r_in'] = rins_exc
        results['r_in_inh'] = rins_inh

        r_out_exc, r_out_exc2, r_out_inh = [], [], []
        for ii, rin_exc in enumerate(rins_exc):
            rin_inh = rins_inh[ii]

            suffix = '';#str(int(fraction_inh_pert*100))
            target_dir = './temp/';
            [exc_rate, inh_rate, traces] = generate(Bee = 1e-5,
                                                    Bei = 1e-5,
                                                    Bie = 1e-5,
                                                    Bii = 1e-5,
                                                    Be_bkg = Be_bkg,
                                                    Be_stim = Be_stim,
                                                    r_bkg = 1,
                                                    r_stim = 1,
                                                    r_bkg_ExtExc=rin_exc,
                                                    r_bkg_ExtInh=rin_inh,
                                                    Ttrans = Ttrans,
                                                    Tblank= Tblank,
                                                    Tstim = Tstim,
                                                    exc_exc_conn_prob = 0,
                                                    exc_inh_conn_prob = 0,
                                                    inh_exc_conn_prob = 0,
                                                    inh_inh_conn_prob = 0,
                                                    fraction_inh_pert=0,
                                                    duration = Ttrans+Tblank+Tstim,
                                                    dt = dt,
                                                    scale_populations=scale_populations,
                                                    format=format,
                                                    percentage_exc_detailed=percentage_exc_detailed,
                                                    target_dir=target_dir,
                                                    suffix=suffix,
                                                    run_in_simulator=run_in_simulator,
                                                    num_processors=num_processors,
                                                    simulation_seed=simulation_seed)

            #
            T = Ttrans+Tblank+Tstim

            if NE_point != 0:
                exc_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popExc.spikes')
                spt_exc = exc_data[:,1]; spi_exc = exc_data[:,0];
                exc_rate = len(spt_exc[spt_exc>(Ttrans/1e3)]) /((Tblank+Tstim)/1e3) /NE_point 
            else: spt_exc = []; spi_exc = []; exc_rate = []
             
            if NE_detailed != 0:
                exc2_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popExc2.spikes')
                spt_exc2 = exc2_data[:,1]; spi_exc2 = exc2_data[:,0]; 
                exc2_rate = len(spt_exc2[spt_exc2>(Ttrans/1e3)]) /((Tblank+Tstim)/1e3) /NE_detailed
            else: spt_exc2 = []; spi_exc2 = []; exc2_rate = []  

            inh_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popInh.spikes')
            spt_inh = inh_data[:,1]; spi_inh = inh_data[:,0];
            inh_rate = len(spt_inh[spt_inh>(Ttrans/1e3)]) /((Tblank+Tstim)/1e3) /NI

            r_out_exc.append(exc_rate)
            r_out_exc2.append(exc2_rate)
            r_out_inh.append(inh_rate)
        
        results['r_out_exc'] = np.array(r_out_exc)
        results['r_out_exc2'] = np.array(r_out_exc2)
        results['r_out_inh'] = np.array(r_out_inh)

        fl = open(target_dir+'rheobase__'+sim_id+'.res', 'wb'); pickle.dump(results, fl); fl.close()
        
        
    elif '-hyperA' in sys.argv:
        results = {}
        
        fraction_inh_pert = 0
        fraction_inh_offset = 0.025
        fraction_inh_offset = 0.9
        inh_offset_amp = -4 # good inc
        #inh_offset_amp = 4 # good dec
        v_clamp= False

        suffix = '';#str(int(fraction_inh_pert*100))
        target_dir = './temp/';
        generate(Bee = Bee,
                Bei = Bei,
                Bie = Bie,
                Bii = Bii,
                Be_bkg = Be_bkg,
                Be_stim = Be_stim,
                r_bkg = 0,
                r_stim = r_stim,
                r_bkg_ExtExc=r_bkg_ExtExc,
                r_bkg_ExtInh=r_bkg_ExtInh,
                r_bkg_ExtExc2=r_bkg_ExtExc2,
                Ttrans = Ttrans,
                Tblank= Tblank,
                Tstim = Tstim,
                Tpost = Tpost,
                exc_exc_conn_prob = exc_exc_conn_prob,
                exc_inh_conn_prob = exc_inh_conn_prob,
                inh_exc_conn_prob = inh_exc_conn_prob,
                inh_inh_conn_prob = inh_inh_conn_prob,
                ee2_conn_prob = ee2_conn_prob,
                ie2_conn_prob = ie2_conn_prob,
                connections=connections, connections2=connections2,
                fraction_inh_pert=fraction_inh_pert,
                fraction_inh_offset = fraction_inh_offset,
                inh_offset_amp = inh_offset_amp,
                duration = Ttrans+Tblank+Tstim+Tpost,
                dt = dt,
                scale_populations=scale_populations,
                format=format,
                percentage_exc_detailed=percentage_exc_detailed,
                target_dir=target_dir,
                suffix=suffix,
                run_in_simulator=run_in_simulator,
                num_processors=num_processors,
                exc_target_dendrites=exc_target_dendrites,
                inh_target_dendrites=inh_target_dendrites,
                v_clamp=v_clamp,
                simulation_seed=simulation_seed)

    elif '-perturbation0' in sys.argv:
        
        run_one()
        
    elif '-perturbation' in sys.argv:
        results = {}
        for fraction_inh_pert in fraction_inh_pert_rng:
            suffix = '';#str(int(fraction_inh_pert*100))
            target_dir = './temp/';
            generate(Bee = Bee,
                    Bei = Bei,
                    Bie = Bie,
                    Bii = Bii,
                    Be_bkg = Be_bkg,
                    Be_stim = Be_stim,
                    r_bkg = 0,
                    r_stim = r_stim,
                    r_bkg_ExtExc=r_bkg_ExtExc,
                    r_bkg_ExtInh=r_bkg_ExtInh,
                    r_bkg_ExtExc2=r_bkg_ExtExc2,
                    Ttrans = Ttrans,
                    Tblank= Tblank,
                    Tstim = Tstim,
                    Tpost = Tpost,
                    exc_exc_conn_prob = exc_exc_conn_prob,
                    exc_inh_conn_prob = exc_inh_conn_prob,
                    inh_exc_conn_prob = inh_exc_conn_prob,
                    inh_inh_conn_prob = inh_inh_conn_prob,
                    ee2_conn_prob = ee2_conn_prob,
                    ie2_conn_prob = ie2_conn_prob,
                    connections=connections, connections2=connections2,
                    fraction_inh_pert=fraction_inh_pert,
                    duration = Ttrans+Tblank+Tstim+Tpost,
                    dt = dt,
                    scale_populations=scale_populations,
                    format=format,
                    percentage_exc_detailed=percentage_exc_detailed,
                    target_dir=target_dir,
                    suffix=suffix,
                    run_in_simulator=run_in_simulator,
                    num_processors=num_processors,
                    exc_target_dendrites=exc_target_dendrites,
                    inh_target_dendrites=inh_target_dendrites,
                    v_clamp=v_clamp,
                    simulation_seed=simulation_seed)

            # --
            NI_pert = int(fraction_inh_pert*NI)
            NI_npert = NI-NI_pert

            T = Ttrans+Tblank+Tstim+Tpost
            bw = 100; 

            if run_in_simulator:

                if NE_point != 0:
                    exc_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popExc.spikes')
                    spt_exc = exc_data[:,1]; spi_exc = exc_data[:,0];
                    hst_exc = np.histogram2d(spt_exc, spi_exc, range=((0,T/1e3),(0,NE_point-1)), bins=(T/bw,NE_point))
                else: 
                    spt_exc = []; spi_exc = [];
                    hst_exc = np.histogram2d(spt_exc, spi_exc,range=((0,T/1e3),(0,NE_point)), bins=(T/bw,NE_point+1))

                if NE_detailed != 0:
                    exc2_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popExc2.spikes')
                    spt_exc2 = exc2_data[:,1]; spi_exc2 = exc2_data[:,0]; 
                    hst_exc2 = np.histogram2d(spt_exc2, spi_exc2, range=((0,T/1e3),(0,NE_detailed-1)), bins=(T/bw,NE_detailed))
                else: 
                    spt_exc2 = []; spi_exc2 = [];
                    hst_exc2 = np.histogram2d(spt_exc2, spi_exc2, range=((0,T/1e3),(0,NE_detailed)), bins=(T/bw,NE_detailed+1))

                inh_data = pl.loadtxt(target_dir+'Sim_ISN_net'+suffix+'.popInh.spikes')
                spt_inh = inh_data[:,1]; spi_inh = inh_data[:,0];
                hst_inh = np.histogram2d(spt_inh, spi_inh, range=((0,T/1e3),(0,NI-1)), bins=(T/bw,NI))

                #
                tt = hst_inh[1][0:-1] + np.diff(hst_inh[1])[0]/2

                re_pop_point = np.nanmean(hst_exc[0], 1) /(bw/1e3)
                re_pop_detailed = np.nanmean(hst_exc2[0], 1) /(bw/1e3)
                ri_pop_pert = np.nanmean(hst_inh[0][:,0:NI_pert], 1) /(bw/1e3)
                ri_pop_npert = np.nanmean(hst_inh[0][:,NI_pert:], 1) /(bw/1e3)  

                res = {}

                res['N'] = N; res['NE']=NE; res['NI']=NI
                res['Ttrans'] = Ttrans; res['Tblank'] = Tblank; 
                res['Tstim'] = Tstim; res['Tpost'] = Tpost

                res['spd_exc'] = np.array([spt_exc, spi_exc])
                res['spd_exc2'] = np.array([spt_exc2, spi_exc2])
                res['spd_inh'] = np.array([spt_inh, spi_inh])

                res['tt'] = tt
                res['re_pop'] = re_pop_point
                res['re_pop_detailed'] = re_pop_detailed
                res['ri_pop_pert'] = ri_pop_pert
                res['ri_pop_npert'] = ri_pop_npert

                results[fraction_inh_pert] = res

            fl = open(target_dir+'perturbation__'+sim_id+'.res', 'wb'); pickle.dump(results, fl); fl.close()
        
