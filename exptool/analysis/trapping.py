
#  12-08-16: cleanup. needs to be merged with neutrapping.
#  12-23-17: break out bar finding algorithms to the more general pattern.py
'''
.___________..______          ___      .______   .______    __  .__   __.   _______ 
|           ||   _  \        /   \     |   _  \  |   _  \  |  | |  \ |  |  /  _____|
`---|  |----`|  |_)  |      /  ^  \    |  |_)  | |  |_)  | |  | |   \|  | |  |  __  
    |  |     |      /      /  /_\  \   |   ___/  |   ___/  |  | |  . `  | |  | |_ | 
    |  |     |  |\  \----./  _____  \  |  |      |  |      |  | |  |\   | |  |__| | 
    |__|     | _| `._____/__/     \__\ | _|      | _|      |__| |__| \__|  \______| 
trapping.py (part of exptool)



CLASSES:
ApsFinding
ComputeTrapping (under construction)


TODO:

-Current work is on the kmeans implementations for generalize for all simulations



MAIN REFERENCE:
Petersen, Weinberg, & Katz (2016)
[http://adsabs.harvard.edu/abs/2016MNRAS.463.1952P]

Calculate quantities for each orbit. Given a set of aps, find:

0. $\langle \theta_{\rm bar}\rangle_N$, the standard trapping metric that assesses the angular separation from the bar for the clusters. Returned value is the maximum angular separation from the bar for the two clusters. $N$ is the number of aps to use in the average.
1. $\langle X_{\rm aps}\rangle_N$, the average position along the bar axis of the aps, by cluster. Then take the minimum of this value. That is, this is the smallest extent of the aps when clustered.
2. $\sigma_{X_{\rm aps}}$, the variance along the bar major axis of the aps positions in a given cluster.
3. Using the ratio of (1) and (2) as a S/N proxy; $x_1$ orbits, as well as higher-order families, have large values. Note that to find $x_2$ orbits, do (1) and (2) in the y-dimension.
4. $\Omega_r$, the $r$ dimension frequency. Used to calculate orbits that fall below the Nyquist frequency for time sampling.

Two improvements over PWK16:
1. Use the closest $N$ aps in time to the indexed time
2. Set a threshold, $T_{\rm thresh}$, that is some multiple of the bar period $T_{\rm bar}$ in which the $N$ aps must reside.

Some combination of these quantities will define the bar.



'''
from __future__ import absolute_import, division, print_function, unicode_literals


# general imports
import time
import numpy as np
import datetime
import os
from scipy import interpolate

# multiprocessing imports
import itertools
from multiprocessing import Pool, freeze_support
import multiprocessing



# exptool imports
from exptool.io import psp_io
from exptool.utils import kmeans
from exptool.utils import utils
from exptool.analysis import pattern




class ApsFinding():

    #
    # class to compute trapping
    #

    '''
    ApsFinding: a class to find aps

    A standard use would be

    >>> A = trapping.ApsFinding()
    >>> TrappingInstance = A.determine_r_aps(simulation_files,trapping_comp,nout=100000,out_directory=simulation_directory,return_aps=True)

    To read back in a saved aps file:
    
    >>> TrappingInstance = A.read_aps_file(aps_file)
    >>> print TrappingInstance['desc']

    Which will make a dictionary of the orbits.

    '''

    def __init__(self,verbose=0):

        self.verbose = verbose

        return None
    

    def accept_files(self,filelist,verbose=0):
        
        self.slist = filelist
        self.verbose = verbose
        
        ApsFinding.parse_list(self)
        

    def parse_list(self):
        
        f = open(self.slist)
        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

        self.SLIST = np.array(s_list)

        if self.verbose >= 1:
            print('ApsFinding.parse_list: Accepted {0:d} files.'.format(len(self.SLIST)))

    
    def determine_r_aps(self,filelist,comp,nout=10,out_directory='',threedee=False,return_aps=False):

        #
        # need to think of the best way to return this data
        #

        #
        #

        self.slist = filelist


        tstamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d+%H:%M:%S')
           
        f = open(out_directory+'apshold'+tstamp+'.dat','wb+')

        ApsFinding.parse_list(self)
        # returns


        #
        # print descriptor string
        #
        desc = 'apsfile for '+comp+' in '+out_directory+', norbits='+str(nout)+', threedee='+str(threedee)+', using '+filelist
        np.array([desc],dtype='S200').tofile(f)

        Oa = psp_io.Input(self.SLIST[0],comp=comp,verbose=0,nout=nout)
        total_orbits = len(Oa.xpos)
        
        aps_dictionary = {} # make a dictionary for the aps
        for i in range(0,total_orbits): aps_dictionary[i] = []
            

        for i in range(1,len(self.SLIST)-1):

            # open three files to compare
            Oa = psp_io.Input(self.SLIST[i-1],comp=comp,nout=nout,verbose=0)
            Ob = psp_io.Input(self.SLIST[i],comp=comp,nout=nout,verbose=self.verbose)
            Oc = psp_io.Input(self.SLIST[i+1],comp=comp,nout=nout,verbose=0)

            # compute 2d radial positions
            if threedee:
                Oa.R = (Oa.xpos*Oa.xpos + Oa.ypos*Oa.ypos + Oa.zpos*Oa.zpos)**0.5
                Ob.R = (Ob.xpos*Ob.xpos + Ob.ypos*Ob.ypos + Ob.zpos*Ob.zpos)**0.5
                Oc.R = (Oc.xpos*Oc.xpos + Oc.ypos*Oc.ypos + Oc.zpos*Oc.zpos)**0.5

            else:
                Oa.R = (Oa.xpos*Oa.xpos + Oa.ypos*Oa.ypos)**0.5
                Ob.R = (Ob.xpos*Ob.xpos + Ob.ypos*Ob.ypos)**0.5
                Oc.R = (Oc.xpos*Oc.xpos + Oc.ypos*Oc.ypos)**0.5
                
            # use logic to find aps
            aps = np.logical_and( Ob.R > Oa.R, Ob.R > Oc.R )

            
            indx = np.array([i for i in range(0,len(Ob.xpos))])

            x = Ob.xpos[aps]
            y = Ob.ypos[aps]
            z = Ob.zpos[aps]
            numi = indx[aps]

            norb = len(numi)


            for j in range(0,norb):
                aps_dictionary[numi[j]].append([Ob.time,x[j],y[j],z[j]])



        self.napsides = np.zeros([total_orbits,2])


        np.array([total_orbits],dtype='i').tofile(f)

        for j in range(0,total_orbits):

            orbit_aps_array = np.array(aps_dictionary[j])

            if (len(orbit_aps_array) > 0):
                naps = len(orbit_aps_array[:,0])  # this might be better as shape

                np.array([naps],dtype='i').tofile(f)

                self.napsides[j,0] = naps
                self.napsides[j,1] = len(orbit_aps_array.reshape(-1,))

                np.array( orbit_aps_array.reshape(-1,),dtype='f').tofile(f)

            else:
                #np.array([0],dtype='i').tofile(f)
                    
                # guard against zero length

                np.array([1],dtype='i').tofile(f)

                np.array( np.array(([-1.,-1.,-1.,-1.])).reshape(-1,),dtype='f').tofile(f)
                    
                        
        f.close()


        print('trapping.ApsFinding.determine_r_aps: savefile is '+out_directory+'apshold'+tstamp+'.dat')

        if (return_aps):
            ApsDict = ApsFinding.read_aps_file(self,out_directory+'apshold'+tstamp+'.dat')
            return ApsDict


    def read_aps_file(self,aps_file):

        f = open(aps_file,'rb')

        [self.desc] = np.fromfile(f,dtype='S200',count=1)

        [self.norb] = np.fromfile(f,dtype='i',count=1)

        self.aps = {}

        for i in range(self.norb):
            
            [naps] = np.fromfile(f,dtype='i',count=1)
            
            if naps > 0:
    
                aps_array = np.fromfile(f,dtype='f',count=4*naps)
                
                self.aps[i] = aps_array.reshape([naps,4])


        f.close()

        ApsDict = ApsFinding.convert_to_dict(self)
        ApsDict['desc'] = self.desc
        ApsDict['norb'] = self.norb

        return ApsDict


    def convert_to_dict(self):
            
        # remake Aps File as a dictionary
        ApsDict = {}
        
        for indx in range(0,self.norb):
            ApsDict[indx] = self.aps[indx]

        return ApsDict

        
    
#
# some definitions--these are probably not the final resting place for these.
#
        
def get_n_snapshots(simulation_directory):
    #
    # find all snapshots
    #
    dirs = os.listdir( simulation_directory )
    n_snapshots = 0
    for file in dirs:
        if file[0:4] == 'OUT.':
            try:
                if int(file[-5:]) > n_snapshots:
                    n_snapshots = int(file[-5:])
            except:
                n_snapshots = n_snapshots
    return n_snapshots




class ComputeTrapping:

    '''
    Class to be filled out with the trapping dictionary solver once it is out of prototyping.

    '''
    def __init__(self):

        pass


#
# how to read and write arrays
#

def write_trapping_file(array,times,filename,tdtype='i1'):
    f = open(filename,'wb')
    np.array([array.shape[0],array.shape[1]],dtype='i').tofile(f)
    np.array(times,dtype='f').tofile(f)
    np.array(array.reshape(-1,),dtype=tdtype).tofile(f)
    f.close()


def read_trapping_file(t_file,tdtype='i1'):
    f = open(t_file,'rb')
    [norb,ntime] = np.fromfile(f,dtype='i',count=2)
    bar_times = np.fromfile(f,dtype='f',count=ntime)
    trap_tmp = np.fromfile(f,dtype=tdtype,count=norb*ntime)
    trap_array = trap_tmp.reshape([norb,ntime])
    return bar_times,trap_array



def reduce_aps_dictionary(TrappingInstance,norb):
    '''
    sometimes you just don't need all those apsides
    '''
    TrappingInstanceOut = {}
    TrappingInstanceOut['norb'] = norb
    TrappingInstanceOut['desc'] = TrappingInstance['desc']

    for i in range(0,norb):
        TrappingInstanceOut[i] = TrappingInstance[i]

    return TrappingInstanceOut



def process_kmeans(ApsArray,indx=-1,k=2,maxima=False):
    '''
    #
    # robust kmeans implementation
    #
    #    -can be edited for speed
    #    -confined to two dimensions

    inputs
    ----------
    ApsArray         : the array of aps for an individual orbit
    indx             : a designation of the orbit, for use with multiprocessing
    k                : the number of clusters
    maxima           : calculate average (if False) or maximum (if True) quantities


    returns
    ----------
    theta_n          :
    clustermean      :
    clusterstd_x     :
    clusterstd_y     :
    kmeans_plus_flag :




    '''
    kmeans_plus_flag = 0
    K = kmeans.KMeans(k,X=ApsArray)
    K.find_centers()
    
    # find the standard deviation of clusters
    try:
        
        # these may be better served as maxima
        if ~maxima:
            clusterstd_x = np.mean([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
            clusterstd_y = np.mean([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])
        
            # mean cluster center in the x dimension (preferred trapping direction).
            clustermean = np.mean([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

        else:
            clusterstd_x = np.max([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
            clusterstd_y = np.max([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])
        
            # maximum x dimension extent for cluster center (not sensitive to non-bar)
            clustermean = np.max([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

        theta_n = np.max([abs(np.arctan(K.mu[i][1]/K.mu[i][0])) for i in range(0,k)])

    # failure on basic kmeans
    except:
        K = kmeans.KPlusPlus(2,X=ApsArray)
        K.init_centers()
        K.find_centers(method='++')
        kmeans_plus_flag = 1
        
        try:

            if ~maxima:
                clusterstd_x = np.mean([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
                clusterstd_y = np.mean([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])
                clustermean = np.mean([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

            else:
                clusterstd_x = np.max([np.std(np.array(K.clusters[i]),axis=0)[0] for i in range(0,k)])
                clusterstd_y = np.max([np.std(np.array(K.clusters[i]),axis=0)[1] for i in range(0,k)])
                clustermean = np.max([(K.mu[i][0]**2. + K.mu[i][1]**2.)**0.5 for i in range(0,k)])

            theta_n = np.max([abs(np.arctan(K.mu[i][1]/K.mu[i][0])) for i in range(0,k)])

        # failure mode for advanced kmeans
        except:
            
            #
            # would like a more intelligent way to diagnose
            #if indx >= 0:
            #    print 'Orbit %i even failed in Kmeans++!!' %indx
            clusterstd_x = np.nan
            clusterstd_y = np.nan
            clustermean = np.nan
            theta_n = np.nan
            kmeans_plus_flag = 2

    
    
    return theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag




def transform_aps(ApsArray,BarInstance):
    '''
    transform_aps : simple transformation for the aps array, offloaded for clarity.
    
    stuck in one direction, watch out
    '''
    bar_positions = find_barangle(ApsArray[:,0],BarInstance)
    X = np.zeros([len(ApsArray[:,1]),2])
    X[:,0] = ApsArray[:,1]*np.cos(bar_positions) - ApsArray[:,2]*np.sin(bar_positions)
    X[:,1] = -ApsArray[:,1]*np.sin(bar_positions) - ApsArray[:,2]*np.cos(bar_positions)
    return X



def do_single_kmeans_step(TrappingInstanceDict,BarInstance,desired_time,\
                          sbuffer=20,\
                          t_thresh=1.5,\
                          maxima=False,\
                          verbose=1): 
    '''
    do_single_kmeans_step: analyze a desired time in the trapping dictionary


    inputs
    ----------
    TrappingInstanceDict :
    BarInstance          :
    desired_time         :
    sbuffer              : number of closest aps (forward and backward looking) to include in clustering
    t_thresh             :
    maxima               :
    verbose              :

    returns
    ----------
    theta_20
    r_frequency
    x_position
    sigma_x
    sigma_y

    note--all are set to np.nan if unclassifiable for some reason
                          
    '''
    norb = TrappingInstanceDict['norb']

    theta_20 = np.zeros(norb)
    r_frequency = np.zeros(norb)
    x_position = np.zeros(norb)
    sigma_x = np.zeros(norb)
    sigma_y = np.zeros(norb)

    # keep track of base statistics
    skipped_for_aps = 0
    skipped_for_res = 0
    sent_to_kmeans_plus = 0
    failed_kmeans_plus = 0

    t1 = time.time()

    
    for indx in range(0,norb):
        if ((indx % (norb/100)) == 0) & (verbose > 0):  utils.print_progress(indx,norb,'trapping.do_single_kmeans_step')
        #

        # block loop completely if too few aps
        if len(TrappingInstanceDict[indx][:,0]) < sbuffer:
            skipped_for_aps += 1

            theta_20[indx] = np.nan
            r_frequency[indx] = np.nan
            x_positio[indx] = np.nan
            sigma_x[indx] = np.nan
            sigma_y[indx] = np.nan
            
            continue


        # find the closest aps
        relative_aps_time = abs(TrappingInstanceDict[indx][:,0] - desired_time)
        closest_aps = (relative_aps_time).argsort()[0:sbuffer]

        # block loop if furthest aps is above some time threshold
        if relative_aps_time[closest_aps[-1]] > t_thresh:
            skipped_for_res += 1

            theta_20[indx] = np.nan
            r_frequency[indx] = np.nan
            x_positio[indx] = np.nan
            sigma_x[indx] = np.nan
            sigma_y[indx] = np.nan
            
            continue


        # transform to bar frame
        X = transform_aps(TrappingInstanceDict[indx],BarInstance)

        # do k-means
        theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag = process_kmeans(X[closest_aps],maxima=maxima)

        if kmeans_plus_flag == 1: sent_to_kmeans_plus += 1

        if kmeans_plus_flag == 2: failed_kmeans_plus += 1

        # set the values for each orbit
        theta_20[indx] = theta_n

        r_frequency[indx] =  1./(TrappingInstanceDict[indx][closest_aps[0],0] -\
                                 TrappingInstanceDict[indx][closest_aps[0]-1,0])

        x_position[indx] = clustermean
        sigma_x[indx] = clusterstd_x
        sigma_y[indx] = clusterstd_y


    if (verbose > 1): print('K-means took {0:3.2f} seconds ({1:3.2f} ms per orbit)'.format(t2, t2/norb*1000))

    print('skipped_for_aps',skipped_for_aps)
    print('skipped_for_res',skipped_for_res)
    print('sent_to_kmeans_plus',sent_to_kmeans_plus)
    print('failed_kmeans_plus',failed_kmeans_plus)


    return theta_20,r_frequency,x_position,sigma_x,sigma_y




def do_kmeans_dict(TrappingInstanceDict,BarInstance,\
                   sbuffer=20,\
                   opening_angle=np.pi/8.,rfreq_limit=22.5,\
                   sigmax_limit=0.001,t_thresh=1.5,\
                   verbose=0):
    '''
    do_kmeans_dict : single processor version of orbit computation

    inputs
    -----------
    TrappingInstanceDict
    BarInstance
    opening_angle
    rfreq_limi
    sigmax_limit
    t_thresh
    verbose


    returns
    -----------
    np.array([trapping_array_x1,trapping_array_x2])

    
    '''
    #
    norb = TrappingInstanceDict['norb']

    trapping_array_x1 = np.zeros([norb,len(BarInstance.time)],dtype='i1')
    trapping_array_x2 = np.zeros([norb,len(BarInstance.time)],dtype='i1')

    t1 = time.time()
    
    #if (verbose > 0): print('trapping.do_kmeans_dict: opening angle=%4.3f, OmegaR=%3.2f, sigma_x limit=%4.3f, Aps Buffer=%i' %(opening_angle,rfreq_limit,sigmax_limit,sbuffer)
    
    for indx in range(0,norb):
        
        if ((indx % (norb/100)) == 0) & (verbose > 0):  utils.print_progress(indx,norb,'trapping.do_kmeans_dict')
        time_sequence = np.array(TrappingInstanceDict[indx])[:,0]
        
        #
        # guard against total aps range being too small (very important for halo!)
        if time_sequence.size < sbuffer:
            continue
        
        #
        # transform to bar frame
        X = transform_aps(TrappingInstanceDict[indx],BarInstance)
        #
        
        orbit_dist = []
        for midpoint in range(0,len(X)):
            relative_aps_time = time_sequence - time_sequence[midpoint]
            closest_aps = (abs(relative_aps_time)).argsort()[0:sbuffer] # fixed this to include closest aps
            #
            #
            # guard against aps with too large of a timespan (some number of bar periods, preferably)
            if relative_aps_time[closest_aps[-1]] > t_thresh:
                orbit_dist.append([0.0,1.0,1.0])
                orbit_dist.append([np.max(BarInstance.time),1.0,1.0])
                continue
            #
            theta_n,clustermean,clusterstd_x,clusterstd_y,kmeans_plus_flag = process_kmeans(X[closest_aps],indx)
            #
            if midpoint==0:
                #
                orbit_dist.append([0.0,theta_n,clusterstd_x])
            #
            # default action
            orbit_dist.append([time_sequence[midpoint],theta_n,clusterstd_x])
            #
            if midpoint==(len(X)-1):
                orbit_dist.append([np.max(BarInstance.time),theta_n,clusterstd_x])
            #
            #
            
        DD = np.array(orbit_dist) # 0:time, 1:theta_n, 2:sigma_x
        
        #nDD = abs(np.ediff1d(DD[:,1],to_begin=1.0))
        #
        tDD = 1./(abs(np.ediff1d(DD[:,0],to_begin=100.0))+1.e-8)

        
        # make interpolated functions:
        #     1. theta_n vs time
        #     2. r_frequency vs time
        #     3. sigma_x vs time
        #     4. delta(theta_n) vs time (volitility, disabled for speed now)
        theta_func = interpolate.interp1d(DD[:,0],DD[:,1], kind='nearest',fill_value=1.4)      #1
        
        frequency_func = interpolate.interp1d(DD[:,0],tDD,kind='nearest',fill_value=1.4)       #2
        
        sigmax_func = interpolate.interp1d(DD[:,0],abs(DD[:,2]),kind='nearest',fill_value=1.4) #3

        #volfunc = interpolate.interp1d(DD[:,0],nDD, kind='nearest',fill_value=1.4)            #4

        #
        #
        # apply trapping rules
        #
        # set up nyquist frequency limit
        nyquist = 1./(4.*(BarInstance.time[1]-BarInstance.time[0]))

        
        x1 = np.where(   (theta_func(BarInstance.time) < opening_angle) \
                       & (frequency_func(BarInstance.time) < nyquist) \
                       & (sigmax_func(BarInstance.time) < sigmax_limit) )[0]


        x2 = np.where(   (frequency_func(BarInstance.time) > nyquist) )[0]
        #
        # set trapped time regions to true
        trapping_array_x1[indx,x1] = np.ones(len(x1))
        trapping_array_x2[indx,x2] = np.ones(len(x2))
    t2 = time.time()-t1
    if (verbose > 1): print('K-means took {0:3.2f} seconds ({1:3.2f} ms per orbit)'.format(t2, t2/norb*1000))
    return np.array([trapping_array_x1,trapping_array_x2])




#
# interesting strategy with dictionaries here
def redistribute_aps(TrappingInstanceDict,divisions):
    #TrappingInstanceDict = {}
    #for indx in range(0,len(TrappingInstance.aps)):
    #    TrappingInstanceDict[indx] = TrappingInstance.aps[indx]
    npart = np.zeros(divisions,dtype=object)
    holders = [{} for x in range(0,divisions)]
    average_part = int(np.floor(TrappingInstanceDict['norb'])/divisions)
    first_partition = TrappingInstanceDict['norb'] - average_part*(divisions-1)
    print('Each processor has {0:d} particles.'.format(average_part))#, first_partition
    low_particle = 0
    for i in range(0,divisions):
        end_particle = low_particle+average_part
        if i==0: end_particle = low_particle+first_partition
        #print low_particle,end_particle
        for j in range(low_particle,end_particle):
            (holders[i])[j-low_particle] = TrappingInstanceDict[j]
        low_particle = end_particle
        if (i>0): holders[i]['norb'] = average_part
        else: holders[i]['norb'] = first_partition
    return holders





def do_kmeans_dict_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return do_kmeans_dict(*a_b)



def multi_compute_trapping(holding,nprocs,BarInstance,\
                   sbuffer=20,\
                   opening_angle=np.pi/8.,rfreq_limit=22.5,\
                   sigmax_limit=0.001,t_thresh=1.5,\
                   verbose=0):
                   
    pool = Pool(nprocs)
    a_args = [holding[i] for i in range(0,nprocs)]
    second_arg = BarInstance
    third_arg = sbuffer
    fourth_arg = opening_angle
    fifth_arg = rfreq_limit
    sixth_arg = sigmax_limit
    seventh_arg = t_thresh
    
    eighth_arg = [0 for i in range(0,nprocs)]
    eighth_arg[0] = verbose
    
    a_vals = pool.map(do_kmeans_dict_star, itertools.izip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),itertools.repeat(fourth_arg),itertools.repeat(fifth_arg),itertools.repeat(sixth_arg),itertools.repeat(seventh_arg),eighth_arg))
    
    # clean up to exit
    pool.close()
    pool.join()
    
    return a_vals



def do_kmeans_multi(TrappingInstanceDict,BarInstance,\
                   sbuffer=20,\
                   opening_angle=np.pi/8.,rfreq_limit=22.5,\
                   sigmax_limit=0.001,t_thresh=1.5,\
                   verbose=0):
    
    nprocs = multiprocessing.cpu_count()
    holding = redistribute_aps(TrappingInstanceDict,nprocs)
    
    if (verbose > 0):
        print('Beginning kmeans, using {0:d} processors.'.format(nprocs))
    
    t1 = time.time()
    freeze_support()

    trapping_arrays = multi_compute_trapping(holding,nprocs,BarInstance,\
                   sbuffer=sbuffer,\
                   opening_angle=opening_angle,rfreq_limit=rfreq_limit,\
                   sigmax_limit=sigmax_limit,t_thresh=t_thresh,\
                   verbose=verbose)
                   
    print('Total trapping calculation took {0:3.2f} seconds, or {1:3.2f} milliseconds per orbit.'.format(time.time()-t1, 1.e3*(time.time()-t1)/len(TrappingInstanceDict)))

    x1_master = re_form_trapping_arrays(trapping_arrays,0)
    x2_master = re_form_trapping_arrays(trapping_arrays,1)
    
    return x1_master,x2_master



def re_form_trapping_arrays(array,array_number):
    # the arrays are structured as [processor][x1/x2][norb][ntime]
    #print array.shape,len(array)
    #print array[0].shape
    norb_master = 0.0
    for processor in range(0,len(array)): norb_master += array[processor].shape[1]
    #
    # now initialize new blank array? Or should it dictionary?
    net_array = np.zeros([norb_master,array[0].shape[2]],dtype='i2')
    start_index = 0
    for processor in range(0,len(array)):
        end_index = start_index + array[processor].shape[1]
        #print processor,start_index,end_index
        net_array[start_index:end_index] = array[processor][array_number]
        start_index = end_index
    return net_array


