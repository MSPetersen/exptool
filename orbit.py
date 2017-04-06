#
# orbit.py
#
#    part of exptool: orbit input/output from simulations, pure data processing convenience.
#
#    11.12.16 formalized class structure, provided example usage.
#    11.19.16 documented definitions; considered for generic upgrade.
#    11.20.16 do generic upgrade to make a single utility
#                 TODO? consider adding a utility conditioned on memory mapping for individual orbits to speed up process.
#                    not clear if needed, reading back in is very fast.
#
#
#    WISHLIST:
#       orbit interpolation routines
#       orbit plotting routines
#

# future compatibility
from __future__ import print_function


import numpy as np
import psp_io
import trapping
import utils


from scipy.interpolate import UnivariateSpline

'''
# Quick Start Demo:

# which exp-numbered files to read in
tarr = np.arange(0,12,1,dtype='int')

# read in from files and return a dictionary
Orbits = orbit.map_orbits('/path/to/outfile.dat','/simulation/directory','runtag',tarr,comp='dark',dictionary=True, norb=10)


# Orbits is a dictionary with several quantities (see initialize_orbit_dictionary below)

'''

def initialize_orbit_dictionary():
    '''
    make the dictionary that handles orbits
    '''

    OrbitDictionary = {}
    OrbitDictionary['T'] = None
    OrbitDictionary['X'] = None
    OrbitDictionary['Y'] = None
    OrbitDictionary['Z'] = None
    OrbitDictionary['VX'] = None
    OrbitDictionary['VY'] = None
    OrbitDictionary['VZ'] = None
    OrbitDictionary['P'] = None
    OrbitDictionary['M'] = None

    return OrbitDictionary
    
    



def map_orbits(outfile,simulation_directory,runtag,time_array,norb=1,comp='star',verbose=0,**kwargs):
    '''
    make_orbit_map

    : slice across the grain for PSPDumps to track individual particles

    Parameters:
    ----------
    outfile: string, filename
        where to save the mapping to a file, even if just a temporary filename

    simulation_directory: string, filename
        leading directory structure

    runtag: string
        name of the simulation
        
    time_array:  integer array
        array of integers that correspond to simulation files to be queried

    norb: integer
        number of orbits to return

    comp: string, component name
        name of simulation component to retrieve orbits from


    verbose: integer
        verbose keyword to be passed to psp_io

    **kwargs:
        'orblist' : integer array of orbit indices to be returned
        'dictionary' : boolean True/False to return a dictionary

    Returns:
    --------
    None

    -or-

    Orbits : OrbitDictionary-like instance
        see class OrbitDictionary below.
    

    '''

    
    infile_template = simulation_directory+'/OUT.'+runtag+'.'

    if 'dictionary' in kwargs:
        return_dictionary = kwargs['dictionary'] # this needs to be passed as an integer array

    #
    # this writes to file because it is a lot to carry around
    f = open(outfile,'wb')
    #

    # check to see if an orbit list has been set
    if 'orblist' in kwargs:
        orbvals = kwargs['orblist'] # this needs to be passed as an integer array
        norb = np.max(orbvals)+1
        print('orbit.map_orbit: N_orbits accepted {} orbits'.format(len(orbvals)))
    else:
        orbvals = np.arange(0,norb,1,dtype='i')



    # get time array from snapshots

    print('orbit.map_orbit: Making mass template...')
    
    times = []
    bad_times = []
    prev_time = -1.
    for indx,val in enumerate(time_array):
        O = psp_io.Input(infile_template+'%05i' %time_array[indx],nout=1,comp=comp)
        
        if (indx > 0):
            if (O.time <= prev_time):
                print('orbit.map_orbit: Bad file number {}, removing'.format(val))
                bad_times.append(indx)
            
        
            else: times.append(O.time)
        else: times.append(O.time)

        prev_time = O.time

    print('...done.')

    
    # remove any bad times
    time_array = np.delete(time_array,bad_times)


    #
    # print self-describing header to file
    np.array([len(time_array),len(orbvals)],dtype=np.int).tofile(f)
    #


    # print to file
    np.array(times,dtype=np.float).tofile(f)

    # get mass array from snapshot
    #    draws from the first file; consider in future if increasing mass during simulation.
    O = psp_io.Input(infile_template+'%05i' %time_array[0],nout=norb,comp=comp)
    masses = O.mass[orbvals]

    # print to file
    np.array(masses,dtype=np.float).tofile(f)

    # loop through files and extract orbits
    for indx,val in enumerate(time_array):

        O = psp_io.Input(infile_template+'%05i' %time_array[indx],nout=norb,comp=comp,verbose=0)
        # overriding verbose here--otherwise gets crazy?

        #if verbose > 0: print O.time
        if (verbose > 0) & (val < np.max(time_array)): utils.print_progress(val,np.max(time_array),'orbit.map_orbit')

        for star in orbvals:
            np.array([O.xpos[star],O.ypos[star],O.zpos[star],O.xvel[star],O.yvel[star],O.zvel[star],O.pote[star]],dtype=np.float).tofile(f)

    f.close()

    if return_dictionary:
        Orbits = read_orbit_map(outfile)

        return Orbits
    


def read_orbit_map(infile):
    '''
    Reads in orbit map file.

    inputs
    ------
    infile: string
        name of the file printed above


    outputs
    ------
    Orbits: dictionary, OrbitDictionary class
        returns an OrbitDictionary class object
        
    '''

    # open file
    f = open(infile,'rb')

    # read header 
    [ntimes,norb] = np.fromfile(f, dtype=np.int,count=2)

    # read times and masses 
    times = np.fromfile(f,dtype=np.float,count=ntimes)
    mass = np.fromfile(f,dtype=np.float,count=norb)

    #print ntimes,norb

    orb = np.memmap(infile,offset=(16 + 8*ntimes + 8*norb),dtype=np.float,shape=(ntimes,norb,7))

    Orbits = initialize_orbit_dictionary()

    Orbits['T'] = times
    Orbits['M'] = mass

    Orbits['X'] = orb[:,:,0]
    Orbits['Y'] = orb[:,:,1]
    Orbits['Z'] = orb[:,:,2]
    Orbits['VX'] = orb[:,:,3]
    Orbits['VY'] = orb[:,:,4]
    Orbits['VZ'] = orb[:,:,5]
    Orbits['P'] = orb[:,:,6]

    return Orbits



def resample_orbit(OrbDict,orbit,impr=4,sord=0,transform=False,**kwargs):
    '''
    return a single resampled orbit

    what's the best way to extend this to multiple orbits?
    what about adding velocity?

    transform via
    bar=BarInstance

    '''
    newT = np.linspace(np.min(OrbDict['T']),np.max(OrbDict['T']),len(OrbDict['T'])*impr)
    sX = UnivariateSpline(OrbDict['T'],OrbDict['X'][:,orbit],s=sord)
    sY = UnivariateSpline(OrbDict['T'],OrbDict['Y'][:,orbit],s=sord)
    sZ = UnivariateSpline(OrbDict['T'],OrbDict['Z'][:,orbit],s=sord)
    
    ResampledDict = {}
    ResampledDict['T'] = newT
    ResampledDict['X'] = sX(newT)
    ResampledDict['Y'] = sY(newT)
    ResampledDict['Z'] = sZ(newT)

    if transform:
        try:
            BarInstance = kwargs['bar']
        except:
            print('orbit.resample_orbit: bar file reading failed. Input using bar keyword.')

        TDict = orbit_transform(ResampledDict,BarInstance)
        ResampledDict['TX'] = TDict['X']
        ResampledDict['TY'] = TDict['Y']
    
    return ResampledDict



def orbit_transform(InDict,BarInstance,velocity=False):

    bar_angle = trapping.find_barangle(InDict['T'],BarInstance)

    
    OutDict = {}
    OutDict['X'] = InDict['X']*np.cos(bar_angle) - InDict['Y']*np.sin(bar_angle)
    OutDict['Y'] = InDict['X']*np.sin(bar_angle) + InDict['Y']*np.cos(bar_angle)

    if velocity:
        OutDict['VX'] = InDict['VX']*np.cos(bar_angle) - InDict['VY']*np.sin(bar_angle)
        OutDict['VY'] = InDict['VX']*np.sin(bar_angle) + InDict['VY']*np.cos(bar_angle)

    return OutDict



def compute_quantities(OrbitDictionary):

    v2 = OrbitDictionary['VX']*OrbitDictionary['VX'] + OrbitDictionary['VY']*OrbitDictionary['VY'] + OrbitDictionary['VZ']*OrbitDictionary['VZ']
    OrbitDictionary['E'] = v2 + OrbitDictionary['P']

    OrbitDictionary['LZ'] = OrbitDictionary['X']*OrbitDictionary['VY'] - OrbitDictionary['Y']*OrbitDictionary['VX']

    return OrbitDictionary



def resample_orbit_map(OrbDict,impr=4,sord=0,transform=False,**kwargs):
    '''
    return a single resampled orbit

    what's the best way to extend this to multiple orbits?
    what about adding velocity?

    transform via
    bar=BarInstance

    '''
    newT = np.linspace(np.min(OrbDict['T']),np.max(OrbDict['T']),len(OrbDict['T'])*impr)

    # initializte a new dictionary
    ResampledDict = {}
    ResampledDict['T'] = newT
    ResampledDict['M'] = OrbDict['M']

    ResampledDict['X']  = np.zeros([ResampledDict['T'].size,OrbDict['M'].size],dtype='f4')
    ResampledDict['Y']  = np.zeros([ResampledDict['T'].size,OrbDict['M'].size],dtype='f4')
    ResampledDict['Z']  = np.zeros([ResampledDict['T'].size,OrbDict['M'].size],dtype='f4')
    ResampledDict['VX'] = np.zeros([ResampledDict['T'].size,OrbDict['M'].size],dtype='f4')
    ResampledDict['VY'] = np.zeros([ResampledDict['T'].size,OrbDict['M'].size],dtype='f4')
    ResampledDict['VZ'] = np.zeros([ResampledDict['T'].size,OrbDict['M'].size],dtype='f4')



    for orbit in range(0,len(OrbDict['M'])):
        sX = UnivariateSpline(OrbDict['T'],OrbDict['X'][:,orbit],s=sord)
        sY = UnivariateSpline(OrbDict['T'],OrbDict['Y'][:,orbit],s=sord)
        sZ = UnivariateSpline(OrbDict['T'],OrbDict['Z'][:,orbit],s=sord)
        sVX = UnivariateSpline(OrbDict['T'],OrbDict['VX'][:,orbit],s=sord)
        sVY = UnivariateSpline(OrbDict['T'],OrbDict['VY'][:,orbit],s=sord)
        sVZ = UnivariateSpline(OrbDict['T'],OrbDict['VZ'][:,orbit],s=sord)
    
        ResampledDict['X'][:,orbit] = sX(newT)
        ResampledDict['Y'][:,orbit] = sY(newT)
        ResampledDict['Z'][:,orbit] = sZ(newT)
        ResampledDict['VX'][:,orbit] = sVX(newT)
        ResampledDict['VY'][:,orbit] = sVY(newT)
        ResampledDict['VZ'][:,orbit] = sVZ(newT)

    if transform:
        try:
            BarInstance = kwargs['bar']
        except:
            print('orbit.resample_orbit: bar file reading failed. Input using bar keyword.')

        ResampledDict = transform_orbit_map(ResampledDict,BarInstance)
    
    return ResampledDict


#
# there is a np.flipud discrepancy between transform_orbit and transform_orbit_map
# 

def transform_orbit_map(OrbitDictionary,BarInstance):
    '''
    inputs
    -------
    OrbitDictionary, from the defined orbit.initialize_orbit_dictionary
    BarInstance, from trapping.BarDetermine()

    returns
    -------
    OrbitDictionary, with four new attributes:
    TX  : parallel velocity to bar
    TY  : perpendicular position to bar
    VTX : parallel velocity to bar
    VTY : perpendicular velocity to bar

    '''
    bar_positions = trapping.find_barangle(OrbitDictionary['T'],BarInstance)

    # make a tiled version for fast computation
    manybar = np.tile(bar_positions,(OrbitDictionary['M'].shape[0],1)).T

    # transform positions
    OrbitDictionary['TX'] = OrbitDictionary['X']*np.cos(manybar) - OrbitDictionary['Y']*np.sin(manybar)
    OrbitDictionary['TY'] = -OrbitDictionary['X']*np.sin(manybar) - OrbitDictionary['Y']*np.cos(manybar)

    # transform velocities
    OrbitDictionary['VTX'] = OrbitDictionary['VX']*np.cos(manybar) - OrbitDictionary['VY']*np.sin(manybar)
    OrbitDictionary['VTY'] = -OrbitDictionary['VX']*np.sin(manybar) - OrbitDictionary['VY']*np.cos(manybar)

    return OrbitDictionary





def make_orbit_density(infile):
    '''
    Makes density plot of a single orbit

    Parameters
    -----------
    infile: string


    '''

    pass

