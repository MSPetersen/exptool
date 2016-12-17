#
# extract global quantities from simulations
#

#  08-29-16: added maximum radius capabilities to bar_fourier_compute

#  10-25-16: some redundancies noticed (bar_fourier_compute) and should be unified
#  12-08-16: cleanup. needs to be merged with neutrapping.

import time
import numpy as np
import psp_io
import datetime
import kmeans



def compute_bar_lag(ParticleInstance,rcut=0.01):
    #
    # simple fourier method to calculate where the particles are in relation to the bar
    #
    R = (ParticleInstance.xpos*ParticleInstance.xpos + ParticleInstance.ypos*ParticleInstance.ypos)**0.5
    TH = np.arctan2(ParticleInstance.ypos,ParticleInstance.xpos)
    loR = np.where( R < rcut)[0]
    A2 = np.sum(ParticleInstance.mass[loR] * np.cos(2.*TH[loR]))
    B2 = np.sum(ParticleInstance.mass[loR] * np.sin(2.*TH[loR]))
    bar_angle = 0.5*np.arctan2(B2,A2)
    print 'Position angle is %4.3f . . .' %bar_angle
    #
    # two steps:
    #   1. rotate theta so that the bar is aligned at 0,2pi
    #   2. fold onto 0,pi to compute the lag
    #
    tTH = (TH - bar_angle + np.pi/2.) % np.pi  # compute lag with bar at pi/2
    #
    # verification plot
    #plt.scatter( R[0:10000]*np.cos(tTH[0:10000]-np.pi/2.),R[0:10000]*np.sin(tTH[0:10000]-np.pi/2.),color='black',s=0.5)
    return tTH - np.pi/2. # retransform to bar at 0



    

def find_barangle(time,BarInstance):
    #
    # use a bar instance to match the output time to a bar position
    #
    try:
        indx_barpos = np.zeros([len(time)])
        for indx,timeval in enumerate(time):
            indx_barpos[indx] = -BarInstance.pos[ abs(timeval-BarInstance.time).argmin()]
    except:
        indx_barpos = -BarInstance.pos[ abs(time-BarInstance.time).argmin()]
    return indx_barpos


    


class BarTransform():

    def __init__(self,ParticleInstanceIn,bar_angle=None):

        self.ParticleInstanceIn = ParticleInstanceIn

        self.bar_angle = bar_angle
        
        self.calculate_transform_and_return()
        
        #return None

    def calculate_transform_and_return(self,maxr=1.):


        if self.bar_angle == None:
            self.bar_angle = -1.*BarTransform.bar_fourier_compute(self,self.ParticleInstanceIn.xpos,self.ParticleInstanceIn.ypos,maxr=maxr)
        
        transformed_x = self.ParticleInstanceIn.xpos*np.cos(self.bar_angle) - self.ParticleInstanceIn.ypos*np.sin(self.bar_angle)
        transformed_y = -self.ParticleInstanceIn.xpos*np.sin(self.bar_angle) - self.ParticleInstanceIn.ypos*np.cos(self.bar_angle)

        transformed_vx = self.ParticleInstanceIn.xvel*np.cos(self.bar_angle) - self.ParticleInstanceIn.yvel*np.sin(self.bar_angle)
        transformed_vy = -self.ParticleInstanceIn.xvel*np.sin(self.bar_angle) - self.ParticleInstanceIn.yvel*np.cos(self.bar_angle)


        self.xpos = transformed_x
        self.ypos = transformed_y
        self.zpos = self.ParticleInstanceIn.zpos

        self.xvel = transformed_vx
        self.yvel = transformed_vy
        self.zvel = self.ParticleInstanceIn.zvel

        self.mass = self.ParticleInstanceIn.mass
        self.pote = self.ParticleInstanceIn.pote

        

    
    def bar_fourier_compute(self,posx,posy,maxr=1.):

        #
        # use x and y positions tom compute the m=2 power, and find phase angle
        #
        w = np.where( (posx*posx + posy*posy)**0.5 < maxr )[0]
        
        aval = np.sum( np.cos( 2.*np.arctan2(posy[w],posx[w]) ) )
        bval = np.sum( np.sin( 2.*np.arctan2(posy[w],posx[w]) ) )

        return np.arctan2(bval,aval)/2.



    


class BarDetermine():

    #
    # class to find the bar
    #

    def __init__(self):
        return None
    
    def track_bar(self,filelist,verbose=0,maxr=1.):

        self.slist = filelist
        self.verbose = verbose
        self.maxr = maxr
        
        BarDetermine.cycle_files(self)

        BarDetermine.unwrap_bar_position(self)

        BarDetermine.frequency_and_derivative(self)

    def parse_list(self):
        f = open(self.slist)
        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

        self.SLIST = np.array(s_list)

        if self.verbose >= 1:
            print 'BarDetermine.parse_list: Accepted %i files.' %len(self.SLIST)

    def cycle_files(self):

        if self.verbose >= 2:
                t1 = time.time()

        BarDetermine.parse_list(self)

        self.time = np.zeros(len(self.SLIST))
        self.pos = np.zeros(len(self.SLIST))

        for i in range(0,len(self.SLIST)):
                O = psp_io.Input(self.SLIST[i],comp='star',verbose=self.verbose)
                self.time[i] = O.time
                self.pos[i] = BarDetermine.bar_fourier_compute(self,O.xpos,O.ypos,maxr=self.maxr)


        if self.verbose >= 2:
                print 'Computed %i steps in %3.2f minutes, for an average of %3.2f seconds per step.' %( len(self.SLIST),(time.time()-t1)/60.,(time.time()-t1)/float(len(self.SLIST)) )


    def bar_doctor_print(self):

        #
        # wrap the bar file
        #
        BarDetermine.unwrap_bar_position(self)

        BarDetermine.frequency_and_derivative(self)

        BarDetermine.print_bar(self,outfile)

        

    def unwrap_bar_position(self,jbuffer=-1.,smooth=False):
    

        #
        # modify the bar position to smooth and unwrap
        #
        jnum = 0
        jset = np.zeros_like(self.pos)
        
        for i in range(1,len(self.pos)):
            
            if (self.pos[i]-self.pos[i-1]) < jbuffer:   jnum += 1

            jset[i] = jnum

        unwrapped_pos = self.pos + jset*np.pi

        if (smooth):
            unwrapped_pos = helpers.savitzky_golay(unwrapped_pos,7,3)

        # to unwrap on twopi, simply do:
        #B.bar_upos%(2.*np.pi)

        self.pos = unwrapped_pos

        #
        # this implementation is not particularly robust, could revisit in future

    def frequency_and_derivative(self,smth_order=None,fft_order=None):

        

        if smth_order or fft_order:
            print 'Cannot assure proper functionality of both order smoothing and low pass filtering.'

        self.deriv = np.zeros_like(self.pos)
        for i in range(1,len(self.pos)):
            self.deriv[i] = (self.pos[i]-self.pos[i-1])/(self.time[i]-self.time[i-1])

            
        if (smth_order):
            smth_params = np.polyfit(self.time, self.deriv, smth_order)
            pos_func = np.poly1d(smth_params)
            self.deriv = pos_func(self.time)

        if (fft_order):
            self.deriv = self.deriv
            
    def bar_fourier_compute(self,posx,posy,maxr=1.):

        #
        # use x and y positions tom compute the m=2 power, and find phase angle
        #
        w = np.where( (posx*posx + posy*posy)**0.5 < maxr )[0]
        
        aval = np.sum( np.cos( 2.*np.arctan2(posy[w],posx[w]) ) )
        bval = np.sum( np.sin( 2.*np.arctan2(posy[w],posx[w]) ) )

        return np.arctan2(bval,aval)/2.



    def print_bar(self,outfile):

        #
        # print the barfile to file
        #

        f = open(outfile,'w')

        for i in range(0,len(self.time)):
            print >>f,self.time[i],self.pos[i],self.deriv[i]

        f.close()
 
    def place_ellipse(self):

        return None

    def read_bar(self,infile):

        #
        # read a printed bar file
        #

        f = open(infile)

        time = []
        pos = []
        deriv = []
        for line in f:
            q = [float(d) for d in line.split()]
            time.append(q[0])
            pos.append(q[1])
            try:
                deriv.append(q[2])
            except:
                pass

        self.time = np.array(time)
        self.pos = np.array(pos)
        self.deriv = np.array(deriv)

        if len(self.deriv < 1):

            BarDetermine.frequency_and_derivative(self)

    def find_barangle(self,time,bartime,barpos):

        #
        # helper class to find the position of the bar at specified times
        #
        
        try:
            tmp = self.pos[0]

            try:
                indx_barpos = np.zeros([len(time)])
                for indx,timeval in enumerate(time):
                    indx_barpos[indx] = -self.pos[ abs(timeval-self.time).argmin()]
            except:
                indx_barpos = -self.pos[ abs(time-self.time).argmin()]

            return indx_barpos

            
        except:
            print 'BarDetermine.find_barangle: Requires BarDetermine.read_bar or BarDetermine.detect_bar to run.'
        





            


class ApsFinding():

    #
    # class to compute trapping
    #

    '''

    A standard use would be

    >>> A = trapping.Trapping()
    >>> A.accept_files('/scratch/mpetersen/Disk013/run013pfiles_min.dat',verbose=2)
    >>> A.determine_r_aps(component,to_file=True,transform=False)

    if using transform, the bar position is computed from fourier methodology, which should be robust to false transformation at T>0.4

    Can also read the files back in

    >>> A.read_apshold_one(apshold_file)
    >>> A.read_apshold_two(apshold_file)

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
            print 'ApsFinding.parse_list: Accepted %i files.' %len(self.SLIST)

    
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


        print 'trapping.ApsFinding.determine_r_aps: savefile is '+out_directory+'apshold'+tstamp+'.dat'

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



