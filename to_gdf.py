import os, sys

def convert(base_dir='./temp/', detailed_also = False):
    
    files = ['%sSim_ISN_net.popExc.spikes'%base_dir,'%sSim_ISN_net.popInh.spikes'%base_dir]
    
    if detailed_also:
        files = ['%sSim_ISN_net.popExc.spikes'%base_dir,'%sSim_ISN_net.popInh.spikes'%base_dir, '%sSim_ISN_net.popExc2.spikes'%base_dir]
        

    out_file = 'ISN-nest-EI-0.gdf' ## NEST format gdf file...
    out = open(out_file,'w')

    for fn in files:
        f = open(fn)

        for l in f.readlines():
            w  = l.split('\t')
            i = int(w[0])
            t = float(w[1])*1000
            if 'Inh' in fn:
                i+=800
            if 'Exc2' in fn:
                i+=790

            out.write('%s   %s\n'%(t, i))
            
        print("> Converted: %s"%os.path.abspath(fn))
        
    print("> Saved:     %s"%os.path.abspath(out_file))
        
    out.close()


if __name__ == '__main__':
    
    
    detailed_also = '-detailed' in sys.argv
        
    convert(detailed_also=detailed_also)
        
        
