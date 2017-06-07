####################################
#
# Python PSP reader
#
#    MSP 10.25.14
#    Added to exptool 12.3.15
#    Constructed to theoretically handle niatr/ndatr 3.7.16
#    niatr/ndatr verified 5.26.16
#
#    08-27-2016 added compatibility for dictionary support, the long-term goal of the reader once I commit to re-engineering everything.
#
#    12-08-2016 cleaned up subdividing inputs. needs much more cleaning, particularly eliminating many 'self' items from the Input class.
#                  should also set up dictionary dump by default, could just engineer in at the end?


'''
.______     _______..______       __    ______   
|   _  \   /       ||   _  \     |  |  /  __  \  
|  |_)  | |   (----`|  |_)  |    |  | |  |  |  | 
|   ___/   \   \    |   ___/     |  | |  |  |  | 
|  |   .----)   |   |  |         |  | |  `--'  | 
| _|   |_______/    | _|    _____|__|  \______/  
                           |______|
psp_io
      input and output of Martin Weinberg's PSP files




'''

import time
import numpy as np
import os

#from exptool.analysis import trapping


class Input():
    '''
    #!
    #! input class to read PSP files
    #!

    #! INPUT fields:
    #!    infile                                      :    string               :    filename to be read
    #!  [optional inputs]
    #!    comp                                        :    string               :    selected component to be read  (default: None)
    #!    nout                                        :    integer              :    how many bodies to return      (default: all)
    #!    verbose                                     :    integer              :    reporting mode, flags below    (default: 0)
    #!    orbit_list                                  :    string               :    filename of orbitlist          (ascii, one integer per line)
    #!    infile_list                                 :    string               :    filename of infilelist         (ascii, one filename per line)
    #!    validate                                    :    boolean              :    validate file and exit         (defalt: False)
    
    #! OUTPUT fields:
    #!    mass,xpos,ypos,zpos,xvel,yvel,zvel,pote     :    float/double         :    particle arrays
    #!       ---OR if multiple timesteps---
    #!    MASS,XPOS,YPOS,ZPOS,XVEL,YVEL,ZVEL,POTE,TIME:    float/double         :    particle arrays [particle, time]

    #! VERBOSITY FLAGS:
    #!  0: Silent
    #!  1: Report Component data
    #!  2: Report Timing data     
    #!  4: Report Debug data      (in progress)

    #! WISHLIST:
    #!     
    #!     return N components, if desired
    #!     continued optimization
    #!     add ability to pull single orbit timeseries from specified files
    #!    

    #! EXAMPLE USAGE
    #! 
    #! import psp_io
    #! O = psp_io.Input(infile)
    #! 
    #!

    # member definitions:
    #
    # __init__
    # psp_full_read
    # master_header_read
    # component_header_read
    # break_info_string
    # component_read
    # orbit_map
    # orbit_resolve
    '''

    def __init__(self, infile, comp=None, nout=None, verbose=0, orbit_list=None, infile_list=None, validate=True):

        #
        # set input parameters
        # 
        
        self.infile = infile
        self.comp = comp
        self.nout = nout
        self.verbose = verbose

        # is there an orbit list?
        self.orbit_list = orbit_list
        self.OLIST = None

        # is there an infile list?
        self.infile_list = infile_list
        self.ILIST = None

        # override validate flag if component
        if self.comp != None: validate=False

        #
        # do the components have niatr/ndatr? to be deprecated once I figure out how to write them properly
            
        #
        # set mode based on inputs
        #
        # 0: failure mode
        # 1: read in orbits from single file 
        # 2: read in orbits from multiple files
        # 3: check validity of PSP file

        
        try:
            self.f = open(infile,'rb')

            #
            # default to single file reading
            #
            mode = 1

            #
            # is this an orbit trace?
            #
            if (self.infile_list): 
                mode = 2

                #
                # check specific cases
                #
                if not (self.orbit_list):
                    mode = 0
                    # mode where orbit_list is not defined, but infile is. Read a single orbit
                    #self.singular_orbit = raw_input("Orbit to probe? ")

                if not (self.comp):
                    mode = 0
                    print 'psp_io.Input: Component must be defined to proceed with orbit resolution.'

            if validate==True:

                # set to a new mode
                mode = 3

                Input.psp_read_headers(self)
        
                if self.verbose>=1:
                    print 'psp_io.Input: The time is %3.3f, with %i components and %i total bodies.' %(self.time,self.ncomp,self.ntot)

                    if self.verbose >= 2:

                        comp_num = 0
                        while comp_num < self.ncomp:
                            
                            print 'psp_io.Input: Component %s, using %s force calculation.' %(self.comp_titles[comp_num],self.comp_expansions[comp_num])

                            comp_num += 1
            
            
                
                    
        except:
            print 'psp_io.Input: The master infile is not defined (or does not exist). Master infile required to proceed.'
            mode = 0



        

        
        #
        # single-file mode
        #
        if mode == 1:

            #
            # drop into full reader routine
            #
            Input.psp_full_read(self)

            self.f.close()

            
        #
        # multi-time mode
        #
        if mode == 2:

            if self.verbose >= 1:
                print 'psp_io.Input: Orbit Resolution Initialized...'
      
            #
            # drop into orbit retrieval mode
            #
            Input.orbit_resolve(self)


        if mode == 0:
            print 'psp_io.Input: Exiting with error.'
            # would be great to put some error code handling in here

            
    def psp_full_read(self):
        master_time = time.time()

        #
        # do cursory header read
        #
        Input.psp_read_headers(self)
        
        if self.verbose>=1:
            print 'psp_io.psp_full_read: The time is %3.3f, with %i components and %i total bodies.' %(self.time,self.ncomp,self.ntot)

        #
        # select component to output
        #
        Input.select_component(self)

        #
        # if the component is found proceed.
        #
        if (self.which_comp >= 0): 

            #
            # how many bodies to return? (overridden if orbit_list)
            #
            if (self.nout):
                self.nbodies = self.nout
            else:
                self.nbodies = self.comp_nbodies[self.which_comp]
                

            #
            # only return a specified list of orbits?
            #          
            if (self.orbit_list):
                Input.orbit_map(self)
                self.nbodies = len(self.OLIST)


            #
            # if returning a single dump, drop into the full component_read loop
            #
            if self.verbose >= 1:
                Input.break_info_string(self)

            #
            # 
            #
            Input.component_read(self)

            
            if self.verbose >= 2:
                print 'psp_io.psp_full_read: PSP file read in %3.2f seconds' %(time.time()-master_time)



        

    def psp_read_headers(self):

        #
        # helper class to read the basic data from multiple headers
        #
        
        # read the master header        
        Input.master_header_read(self)

        #
        # inspect component headers
        #
        present_comp = 0
        while present_comp < self.ncomp:
            
            if self.verbose >= 4:
                print 'psp_io.psp_read_headers: Examining component %i' %(present_comp)
                
            # read the component header
            Input.component_header_read(self,present_comp)

            self.f.seek(self.comp_data_end[present_comp])

            present_comp += 1


    def select_component(self):

        #
        # decide which component, if any, to retrieve
        #
        
        if (self.comp):
            try:
                self.which_comp = np.where(np.array(self.comp_titles) == self.comp)[0][0]
            except:
                print 'psp_io.select_component: No matching component!'
                self.which_comp = None
        else:
            self.which_comp = None

            if self.verbose > 0:
                print 'psp_io.select_component: Proceeding without selecting component.'




    def master_header_read(self):
        
        #
        # read the master header
        #
        #    Allocate arrays of necessary component 
        #
        self.f.seek(16) # find magic number
        [cmagic] = np.fromfile(self.f, dtype=np.uint32,count=1)

        # check if it is a float
        # reasonably certain the endianness doesn't affect us here, but verify?
        if cmagic == 2915019716:
            self.floatl = 4
            self.dyt = 'f'
        else:
            self.floatl = 8
            self.dyt = 'd'
            
        # reset to beginning and proceed
        self.f.seek(0)
        [self.time] = np.fromfile(self.f, dtype='<f8',count=1)
        [self.ntot,self.ncomp] = np.fromfile(self.f, dtype=np.uint32,count=2)

        self.comp_pos = np.zeros(self.ncomp,dtype=np.uint64)                  # byte position of COMPONENT HEADER for returning easily
        self.comp_pos_data = np.zeros(self.ncomp,dtype=np.uint64)             # byte position of COMPONENT DATA for returning easily
        self.comp_data_end = np.zeros(self.ncomp,dtype=np.uint64)             # byte position of COMPONENT DATA END for returning easily

        # generic PSP items worth making accessible
        self.comp_titles = ['' for i in range(0,self.ncomp)]
        self.comp_expansions = ['' for i in range(0,self.ncomp)]
        self.comp_basis = ['' for i in range(0,self.ncomp)]
        self.comp_niatr = np.zeros(self.ncomp,dtype=np.uint64)                # each component's number of integer attributes
        self.comp_ndatr = np.zeros(self.ncomp,dtype=np.uint64)                # each component's number of double attributes
        self.comp_string = ['' for i in range(0,self.ncomp)]
        self.comp_nbodies = np.zeros(self.ncomp,dtype=np.uint64)              # each component's number of bodies

        
    def component_header_read(self,present_comp):

        self.comp_pos[present_comp] = self.f.tell()
        
        # if PSP changes, this will have to be altered, or I need to figure out a more future-looking version
        if self.floatl==4:
            [cmagic,deadbit,nbodies,niatr,ndatr,infostringlen] = np.fromfile(self.f, dtype=np.uint32,count=6)
        else: 
            [nbodies,niatr,ndatr,infostringlen] = np.fromfile(self.f, dtype=np.uint32,count=4)

        # information string from the header
        head = np.fromfile(self.f, dtype='a'+str(infostringlen),count=1)
        [comptitle,expansion,EJinfo,basisinfo] = [q for q in head[0].split(':')]

        self.comp_pos_data[present_comp] = self.f.tell()            # save where the data actually begins

        # 8 is the number of fields (m,x,y,z,vx,vy,vz,p)
        comp_length = nbodies*(self.floatl*8 + 4*niatr + self.floatl*ndatr)
        self.comp_data_end[present_comp] = self.f.tell() + comp_length                         # where does the data from this component end?
        
        self.comp_titles[present_comp] = comptitle.strip()
        self.comp_expansions[present_comp] = expansion.strip()
        self.comp_basis[present_comp] = basisinfo
        self.comp_niatr[present_comp] = niatr
        self.comp_ndatr[present_comp] = ndatr
        self.comp_string[present_comp] = str(head)
        self.comp_nbodies[present_comp] = nbodies


    def create_particle_buffer(self):
        #
        # routine to return the unique data array based on the particle class
        #

        if self.floatl==4:
            fstring = 'f,f,f,f,f,f,f,f'
            for i in range(0,self.comp_niatr[self.which_comp]): fstring += ',i'
            for i in range(0,self.comp_ndatr[self.which_comp]): fstring += ',f'

        else:
            fstring = 'd,d,d,d,d,d,d,d'
            for i in range(0,self.comp_niatr[self.which_comp]): fstring += ',i'
            for i in range(0,self.comp_ndatr[self.which_comp]): fstring += ',d'

            
        self.readtype = np.dtype(fstring)


    def component_read(self):

        #
        # define particle data type
        #      which defines self.readtype
        #

        self.create_particle_buffer()


        #
        # gather data field
        #
        if not (self.orbit_list):

            out = np.memmap(self.infile,dtype=self.readtype,shape=(1,int(self.nbodies)),offset=int(self.comp_pos_data[self.which_comp]),order='F',mode='r')


            #
            # populate known attributes
            #
            self.mass = out['f0'][0]
            self.xpos = out['f1'][0]
            self.ypos = out['f2'][0]
            self.zpos = out['f3'][0]
            self.xvel = out['f4'][0]
            self.yvel = out['f5'][0]
            self.zvel = out['f6'][0]
            self.pote = out['f7'][0]

            
            #
            # treat niatr, ndatr
            #
            for int_attr in range(0,self.comp_niatr[self.which_comp]): 

                setattr(self, 'i'+str(int_attr), out['f'+str(8+int_attr)][0])


            for dbl_attr in range(0,self.comp_ndatr[self.which_comp]): 

                setattr(self, 'd'+str(dbl_attr), out['f'+str(int(8 + self.comp_niatr[self.which_comp] + dbl_attr))][0])


        #        
        # mode in which only specific orbits are returned
        #
        if (self.orbit_list):

            #
            # read in all orbits, then obtain specific orbits
            #     (okay because low overhead)
            #
            out = np.memmap(self.infile,dtype=self.readtype,shape=(1,int(self.comp_nbodies[self.which_comp])),offset=int(self.comp_pos_data[self.which_comp]),order='F',mode='r')

            #print np.array(out['f0'][0])[self.OLIST]
            #print out['f0'][0][self.OLIST]

            self.mass = out['f0'][0][self.OLIST]
            self.xpos = out['f1'][0][self.OLIST]
            self.ypos = out['f2'][0][self.OLIST]
            self.zpos = out['f3'][0][self.OLIST]
            self.xvel = out['f4'][0][self.OLIST]
            self.yvel = out['f5'][0][self.OLIST]
            self.zvel = out['f6'][0][self.OLIST]
            self.pote = out['f7'][0][self.OLIST]

            #
            # treat niatr, ndatr
            #
            for int_attr in range(0,self.comp_niatr[self.which_comp]): # + self.comp_ndatr[self.which_comp]

                setattr(self, 'i'+str(int_attr), out['f'+str(8+int_attr)][0][self.OLIST])


            for dbl_attr in range(0,self.comp_ndatr[self.which_comp]): # + self.comp_ndatr[self.which_comp]

                setattr(self, 'd'+str(dbl_attr), out['f'+str(int(8 + self.comp_niatr[self.which_comp] + dbl_attr))][0][self.OLIST])



    def break_info_string(self):

        #
        # break the info string to be human-readable
        #
        
        head = self.comp_string[self.which_comp]
        [comptitle,expansion,EJinfo,basisinfo] = [q for q in head.split(':')]

        print 'component: ',self.comp_titles[self.which_comp]
        print 'bodies: ',self.comp_nbodies[self.which_comp]
        print 'expansion: ',expansion.strip()
        print 'ej info: ',EJinfo
        print 'basis info: ',basisinfo

        #
        # could develop a more user-friendly output for these
        #

    def orbit_map(self):

        #
        # read in the orbit list and convert to an array
        #
        
        g = open(self.orbit_list)
        olist = []
        for line in g:
            d = [q for q in line.split()]
            # no safeguards here yet
            if len(d)==1: olist.append(int(d[0]))

        g.close()

        self.OLIST = np.array(olist)

        #
        # override number of bodies to return to match orbit list
        #
        self.nbodies = len(self.OLIST)

        if self.verbose >= 1:
            print 'psp_io.orbit_map: Orbit map accepted with %i bodies.' %self.nbodies

    def timestep_map(self):

        #
        # read in file list and convert to an array
        #

        g = open(self.infile_list)
        ilist = []
        for line in g:
            d = [q for q in line.split()]
            if len(d)==1: ilist.append(d[0])

        g.close()

        self.ILIST = np.array(ilist)

        if self.verbose >= 1:
            print 'psp_io.timestep_map: Filename map accepted with %i files (timesteps).' %len(self.ILIST)

        
    def orbit_resolve(self):

        #
        # wrapper to cycle through different files (timesteps) and return orbits
        #
        if self.verbose >= 2:
            res_time_initial = time.time()

        #
        # read a first array to seek_list
        #
        Input.psp_read_headers(self)
        self.f.close()

        if self.verbose>=1:
            print 'psp_io.orbit_resolve: The time is %3.3f, with %i components and %i total bodies.' %(self.time,self.ncomp,self.ntot)

        #
        # select component to output
        #
        Input.select_component(self)

        #
        # select orbits and files to map
        #
        Input.orbit_map(self)
        Input.timestep_map(self)
        
        #
        # allocate particle arrays
        #
        self.ntimesteps = len(self.ILIST)
        
        self.TIME = np.zeros([self.ntimesteps])
        self.XPOS = np.zeros([self.nbodies,self.ntimesteps])
        self.YPOS = np.zeros([self.nbodies,self.ntimesteps])
        self.ZPOS = np.zeros([self.nbodies,self.ntimesteps])
        self.XVEL = np.zeros([self.nbodies,self.ntimesteps])
        self.YVEL = np.zeros([self.nbodies,self.ntimesteps])
        self.ZVEL = np.zeros([self.nbodies,self.ntimesteps])
        self.POTE = np.zeros([self.nbodies,self.ntimesteps])

        #
        # cycle through files
        #
        for i,file in enumerate(self.ILIST):

            #
            # open the next file
            #
            self.f = open(file,'rb')

            [time] = np.fromfile(self.f, dtype='<f8',count=1)

            if self.verbose>=4:
                print 'Time: %3.3f' %(time)

            #
            # read and stuff arrays
            #
            self.infile = file
            Input.component_read(self)

            # set mass once, which is unchanging (for now!)
            if i==0: self.MASS = self.mass

            
            self.TIME[i] = time
            #self.MASS[:,i] = self.mass
            self.XPOS[:,i] = self.xpos
            self.YPOS[:,i] = self.ypos
            self.ZPOS[:,i] = self.zpos
            self.XVEL[:,i] = self.xvel
            self.YVEL[:,i] = self.yvel
            self.ZVEL[:,i] = self.zvel
            self.POTE[:,i] = self.pote

            #
            # close file for cleanliness
            #
            self.f.close()

            #
            # delete the individual instances
            #
            del self.mass
            del self.xpos
            del self.ypos
            del self.zpos
            del self.xvel
            del self.yvel
            del self.zvel
            del self.pote

        if self.verbose >= 2:
                    print 'psp_io.orbit_resolve: Orbit(s) resolved in %3.2f seconds' %(time.time()-res_time_initial)




class PSPDump():
    '''

    class to wrap the Input class in order to allow for easier manipulation

    '''

    def __init__(self, infile, comp=None, nout=None, verbose=0, orbit_list=None, infile_list=None, validate=False):


        DUMP = Input(infile,comp=comp,nout=nout,verbose=verbose)

        self.xpos = DUMP.xpos
        self.ypos = DUMP.ypos
        self.zpos = DUMP.zpos
        self.xvel = DUMP.xvel
        self.yvel = DUMP.yvel
        self.zvel = DUMP.zvel
        self.pote = DUMP.pote

    def add_quantities(self):

        self.rtwo = (self.xpos*self.xpos + self.ypos*self.ypos)**0.5
        self.rthree = (self.xpos*self.xpos + self.ypos*self.ypos + self.zpos*self.zpos)**0.5
        self.v2 = (self.xvel*self.xvel + self.yvel*self.yvel + self.zvel*self.zvel)
        self.E = self.v2 + self.pote

        

#
# Below here are helper functions to subdivide and combine particles.
#

class particle_holder(object):
    #
    # all the quantities you could ever want to fill in your own dump.
    #
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




def convert_to_dict(ParticleInstance):
    ParticleInstanceDict = {}
    ParticleInstanceDict['xpos'] = ParticleInstance.xpos
    ParticleInstanceDict['ypos'] = ParticleInstance.ypos
    ParticleInstanceDict['zpos'] = ParticleInstance.zpos
    ParticleInstanceDict['xvel'] = ParticleInstance.xvel
    ParticleInstanceDict['yvel'] = ParticleInstance.yvel
    ParticleInstanceDict['zvel'] = ParticleInstance.zvel
    ParticleInstanceDict['pote'] = ParticleInstance.pote
    ParticleInstanceDict['mass'] = ParticleInstance.mass
    return convert_to_dict

    

#
# this really shouldn't even be an option anymore.
def subdivide_particles(ParticleInstance,loR=0.,hiR=1.0,zcut=1.0,loT=-np.pi,hiT=np.pi,transform=False,bar_angle=None):
    #
    # if transform=True, requires ParticleInstance.xbar to be defined
    #
    R = (ParticleInstance.xpos*ParticleInstance.xpos + ParticleInstance.ypos*ParticleInstance.ypos)**0.5
    if transform==False:
        particle_roi = np.where( (R > loR) & (R < hiR) & (abs(ParticleInstance.zpos) < zcut))[0]
    #if transform==True:
    #    # compute the bar lag
    #    ParticleInstanceTransformed = trapping.BarTransform(ParticleInstance,bar_angle=bar_angle)
    #    BL = ( (np.arctan2(ParticleInstanceTransformed.ypos,ParticleInstanceTransformed.xpos) + np.pi/2.) % np.pi) - np.pi/2.
    #    # look for particles in the wedge relative to bar angle
    #    particle_roi = np.where( (R > loR) & (R < hiR) & (abs(ParticleInstance.zpos) < zcut) & (BL > loT) & (BL < hiT))[0]
    #
    # fill a new array with particles that meet this criteria
    #
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
    return holder


def subdivide_particles_list(ParticleInstance,particle_roi):
    #
    # fill a new array with particles that meet this criteria
    #
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



#
# can this get infile, etc?
#
def mix_particles(ParticleInstanceArray):
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



###############################################################################

# manipulate file lists

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






def map_simulation_files(outfile,simulation_directory,simulation_name):
    #
    # simple definition to sort through a directory and make a list of all the dumps, guarding for bad files
    #
    if not os.path.isfile(outfile): # check to see if map already exists before making
        #
        f = open(outfile,'w')
        #
        n_snapshots = get_n_snapshots(simulation_directory)
        current_time = -1.
        for i in range(0,n_snapshots+1):
            try:
                PSPDump = Input(simulation_directory+'OUT.'+simulation_name+'.%05i' %i)
                t = PSPDump.time
                del PSPDump

            except:
                print 'Bad file: ',simulation_directory+'OUT.'+simulation_name+'.%05i' %i
                t = current_time
                
            if t > current_time:
                print >>f,simulation_directory+'OUT.'+simulation_name+'.%05i' %i
                current_time = t
                print current_time
                
        #
        f.close()
    #
    else:
        print('psp_io.map_simulation_files: file already exists.')

