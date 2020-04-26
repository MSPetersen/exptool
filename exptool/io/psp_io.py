"""
Python PSP (Phase-Space Protocol) reader

-25 Oct 2014 in original form, Michael S. Petersen, michael.petersen@roe.ac.uk
-03 Dec 2015 committed to exptool
-07 Mar 2016 constructed to theoretically handle niatr/ndatr
-27 Aug 2016 added compatibility for dictionary support, the long-term goal of the reader once I commit to re-engineering everything.
-08 Dec 2016 cleaned up subdividing inputs. needs much more cleaning, particularly eliminating many 'self' items from the Input class. Should also set up dictionary dump by default, could just engineer in at the end?
-11 Mar 2019 set up to read yaml-derived input files. A method to diagnose problems would be amazing--currently written elsewhere.
-14 Aug 2019 handle indexing=True from exp component inputs
-17 Dec 2019 major revision to simplify

PSP is a file format used by the EXP basis function expansion N-body code
written by Martin Weinberg.


TODO
-check translator for old PSP files
-add protection for missing yaml
-add handling for split PSP files
-add handling for multiple components simultaneously
-add _make_dataframe

"""

import numpy as np

# requires yaml support: likely needs to be installed.
import yaml


class Input:
    """Python reader for Phase-Space Protocol (PSP) files used by EXP.

    inputs
    ----------
    filename : str
        The PSP filename to load.

    """

    def __init__(self, filename,comp=None, legacy=True,nbodies=-1,verbose=0,nout=-1):
        """
        inputs
        ------------
        comp    : str
            name of the component to return.
        legacy  : boolean
            if True, adds support for other exptool methods. unneeded if building from scratch.
        nbodies : integer
            reduce the number of bodies that are returned.
        verbose : integer
            levels of verbosity.
        nout    : integer
            deprecated compatibility parameter. use nbodies instead.
        
        """

        
        self.filename = filename
        self.nbodies = nbodies
        
        # deprecated.
        self.infile = filename
        
        # deprecated.
        self.nbodies = int(np.nanmax([nbodies,nout]))
        
        # initial check for file validity
        try:
            f = open(self.filename, 'rb')
            f.close()
        except Exception:
            raise IOError('Failed to load header from file "{}" - are you sure '
                          'this is a PSP file?'.format(filename))

        # test for split PSP files
        # TODO

        # do an initial read of the header
        self.header = self._read_master_header()
        
        _comps = list(self.header.keys())
 

        # if a component is defined, retrieve data
        if comp != None:
            if comp not in _comps:
                raise IOError('The specified component does not exist.')
                
            else:
                self.data = self._read_component_data(self.header[comp])
            
        # if no header is defined, you will get just the master header
        
        if (legacy) & (comp!=None):
            self.header = self.header[comp]
            self.comp = comp
            self._make_backward_compatible()
        elif legacy:
            raise IOError('A component must be specified for legacy usage.')
            

    def _read_component_header(self, f, comp_idx):
        """read in the header for a single component"""
        
        _ = f.tell()  # byte position of this component

        # TODO: if PSP changes, this will have to be altered
        if self._float_len == 4:
            _1,_2, nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(f, dtype=np.uint32, count=6)
            
        else:
            nbodies, nint_attr, nfloat_attr, infostringlen = np.fromfile(f, dtype=np.uint32, count=4)

        # information string from the header
        head = np.fromfile(f, dtype=np.dtype((np.bytes_, infostringlen)),
                           count=1)
        
        # need the backward compatibility here.
        try:
            head_normal = head[0].decode()
            head_dict = yaml.safe_load(head_normal)
        except:
            head_sep = head[0].split('\n')
            head_dict = dict()
            for param in head_sep:
                head_dict[param.split(':')[0].strip()] = param.split(':')[1].strip()

        comp_data_pos = f.tell()  # byte position where component data begins

        # the default fields are (m, x, y, z, vx, vy, vz, p)
        nfields = 8
        comp_length = nbodies * (8 * int(head_dict['parameters']['indexing']) +
                                 self._float_len * nfields +
                                 4 * nint_attr +
                                 self._float_len * nfloat_attr)
        comp_data_end = f.tell() + comp_length  # byte pos. of comp. data end

        data = dict()
        data['index'] = comp_idx
        for k in head_dict:
            data[k] = head_dict[k]
        data['nint_attr'] = nint_attr
        data['nfloat_attr'] = nfloat_attr
        data['nbodies'] = nbodies
        data['data_start'] = comp_data_pos
        data['data_end'] = comp_data_end
        f.seek(comp_data_end)

        return data

    def _read_master_header(self):
        """read the master header of the PSP file"""

        master_header = dict()
        nbodies = 0

        with open(self.filename, 'rb') as f:

            f.seek(16)  # find magic number
            cmagic, = np.fromfile(f, dtype=np.uint32, count=1)

            # check if it is float vs. double
            if cmagic == 2915019716:
                self._float_len = 4
                self._float_str = 'f'
            else:
                self._float_len = 8
                self._float_str = 'd'

            # reset to beginning and read current time
            f.seek(0)
            self.time, = np.fromfile(f, dtype='<f8', count=1)
            self._nbodies_tot, self._ncomp = np.fromfile(f, dtype=np.uint32,
                                                         count=2)

            for i in range(self._ncomp):
                data = self._read_component_header(f, i)
                master_header[data.pop('name')] = data
                nbodies += data['nbodies']

        master_header['nbodies'] = nbodies

        return master_header

    def _read_component_data(self, comp_header):
        """read in all data for component"""

        dtype_str = []
        colnames = []
        if comp_header['parameters']['indexing']:
            # if indexing is on, the 0th column is Long
            dtype_str = dtype_str + ['l']
            colnames = colnames + ['index']

        dtype_str = dtype_str + [self._float_str] * 8
        colnames = colnames + ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'potE']

        dtype_str = dtype_str + ['i'] * comp_header['nint_attr']
        colnames = colnames + ['i_attr{}'.format(i)
                               for i in range(comp_header['nint_attr'])]

        dtype_str = dtype_str + [self._float_str] * comp_header['nfloat_attr']
        colnames = colnames + ['f_attr{}'.format(i)
                               for i in range(comp_header['nfloat_attr'])]

        dtype = np.dtype(','.join(dtype_str))

        out = np.memmap(self.filename,
                        dtype=dtype,
                        shape=(1, comp_header['nbodies']),
                        offset=int(comp_header['data_start']),
                        order='F', mode='r')

        tbl = {}
        for i, name in enumerate(colnames):
            if self.nbodies > 0:
                tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)[0:self.nbodies]
            else:
                tbl[name] = np.array(out['f{}'.format(i)][0], copy=True)

        del out  # close the memmap instance

        return tbl


    def _make_backward_compatible(self):
        """routine to make the dictionary style from above a drop-in replacement for old psp_io"""
        
        if self.nbodies > 0:
            self.mass = self.data['m'][0:self.nbodies]
            self.xpos = self.data['x'][0:self.nbodies]
            self.ypos = self.data['y'][0:self.nbodies]
            self.zpos = self.data['z'][0:self.nbodies]
            self.xvel = self.data['vx'][0:self.nbodies]
            self.yvel = self.data['vy'][0:self.nbodies]
            self.zvel = self.data['vz'][0:self.nbodies]
            self.pote = self.data['potE'][0:self.nbodies]
        
        else:
            self.mass = self.data['m']
            self.xpos = self.data['x']
            self.ypos = self.data['y']
            self.zpos = self.data['z']
            self.xvel = self.data['vx']
            self.yvel = self.data['vy']
            self.zvel = self.data['vz']
            self.pote = self.data['potE']
            
                
        # may also want to delete self.data in this case to save memory
        del self.data
        
    def _make_dataframe(self):
        """routine to make the dictionary style from above pandas dataframe"""
        
        pass
        
        
        

#
# Below here are helper functions to subdivide and combine particles for parallel processes
#

class particle_holder(object):
    '''all the quantities you could ever want to fill in your own PSP-style output.
    '''
    infile = None
    comp = None
    nbodies = None
    time = None
    xpos = None
    ypos = None
    zpos = None
    xvel = None
    yvel = None
    zvel = None
    mass = None
    pote = None





def subdivide_particles_list(ParticleInstance,particle_roi):
    '''fill a new array with particles that meet this criteria
    '''
    holder = particle_holder()
    holder.xpos = ParticleInstance.xpos[particle_roi]
    holder.ypos = ParticleInstance.ypos[particle_roi]
    holder.zpos = ParticleInstance.zpos[particle_roi]
    holder.xvel = ParticleInstance.xvel[particle_roi]
    holder.yvel = ParticleInstance.yvel[particle_roi]
    holder.zvel = ParticleInstance.zvel[particle_roi]
    holder.mass = ParticleInstance.mass[particle_roi]
    holder.infile = ParticleInstance.infile
    holder.comp = ParticleInstance.comp
    holder.nbodies = ParticleInstance.nbodies
    holder.time = ParticleInstance.time
    return holder




def mix_particles(ParticleInstanceArray):
    '''flatten arrays from multiprocessing into one ParticleInstance.
    '''
    n_instances = len(ParticleInstanceArray)
    n_part = 0
    for i in range(0,n_instances):
        n_part += len(ParticleInstanceArray[i].xpos)
    final_holder = particle_holder()
    final_holder.xpos = np.zeros(n_part)
    final_holder.ypos = np.zeros(n_part)
    final_holder.zpos = np.zeros(n_part)
    final_holder.xvel = np.zeros(n_part)
    final_holder.yvel = np.zeros(n_part)
    final_holder.zvel = np.zeros(n_part)
    final_holder.mass = np.zeros(n_part)
    final_holder.pote = np.zeros(n_part)
    #holder.infile = ParticleInstance.infile
    #holder.comp = ParticleInstance.comp
    #holder.nbodies = ParticleInstance.nbodies
    final_holder.time = ParticleInstanceArray[0].time # only uses first time, should be fine?
    #
    #
    first_part = 0
    for i in range(0,n_instances):
        n_instance_part = len(ParticleInstanceArray[i].xpos)
        final_holder.xpos[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].xpos
        final_holder.ypos[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].ypos
        final_holder.zpos[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].zpos
        final_holder.xvel[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].xvel
        final_holder.yvel[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].yvel
        final_holder.zvel[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].zvel
        final_holder.mass[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].mass
        final_holder.pote[first_part:first_part+n_instance_part] = ParticleInstanceArray[i].pote
        first_part += n_instance_part
    return final_holder


