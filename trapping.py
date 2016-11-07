#
# extract global quantities from simulations
#

#  08-29-16: added maximum radius capabilities to bar_fourier_compute

#  10-25-16: some redundancies noticed (bar_fourier_compute) and should be unified

import time
import numpy as np
import psp_io
import datetime



'''
# try a quick aps-finding exercise


import psp_io
import potential
import helpers
import trapping

B = trapping.BarDetermine()
B.read_bar('/scratch/mpetersen/disk064abar.dat')



#B = trapping.BarDetermine('/scratch/mpetersen/Disk007/testfiles3.dat',verbose=2)


D = trapping.BarDetermine()
D.accept_inputs('/scratch/mpetersen/Disk013/run013pfiles.dat',verbose=2)
D.accept_inputs('/scratch/mpetersen/Disk013/run013pfiles.dat',verbose=2)


D.unwrap_bar_position()
D.frequency_and_derivative(smth_order=0) # go ahead and print the whole thing, can always re-run
D.print_bar('/scratch/mpetersen/disk013pbar.dat')



trapping.BarDetermine.frequency_and_derivative(B,smth_order=2)
trapping.BarDetermine.frequency_and_derivative(C,smth_order=2)


EK = potential.EnergyKappa(Od)


'''

class particle_holder(object):
    ctime = None
    xpos = None
    ypos = None
    zpos = None
    xvel = None
    yvel = None
    zvel = None
    mass = None
    pote = None



def find_barangle(time,BarInstance):
    try:
        indx_barpos = np.zeros([len(time)])
        for indx,timeval in enumerate(time):
            indx_barpos[indx] = -BarInstance.bar_pos[ abs(timeval-BarInstance.bar_time).argmin()]
    except:
        indx_barpos = -BarInstance.bar_pos[ abs(time-BarInstance.bar_time).argmin()]
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

        self.bar_time = np.zeros(len(self.SLIST))
        self.bar_pos = np.zeros(len(self.SLIST))

        for i in range(0,len(self.SLIST)):
                O = psp_io.Input(self.SLIST[i],comp='star',verbose=self.verbose)
                self.bar_time[i] = O.ctime
                self.bar_pos[i] = BarDetermine.bar_fourier_compute(self,O.xpos,O.ypos,maxr=self.maxr)


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
        jset = np.zeros_like(self.bar_pos)
        
        for i in range(1,len(self.bar_pos)):
            
            if (self.bar_pos[i]-self.bar_pos[i-1]) < jbuffer:   jnum += 1

            jset[i] = jnum

        self.bar_upos = self.bar_pos + jset*np.pi

        if (smooth):
            self.bar_upos = helpers.savitzky_golay(self.bar_upos,7,3)

        # to unwrap on twopi, simply do:
        #B.bar_upos%(2.*np.pi)

        #
        # this implementation is not particularly robust, could revisit in future

    def frequency_and_derivative(self,smth_order=None,fft_order=None):

        

        if smth_order or fft_order:
            print 'Cannot assure proper functionality of both order smoothing and low pass filtering.'

        self.bar_deriv = np.zeros_like(self.bar_upos)
        for i in range(1,len(self.bar_upos)):
            self.bar_deriv[i] = (self.bar_upos[i]-self.bar_upos[i-1])/(self.bar_time[i]-self.bar_time[i-1])

            
        if (smth_order):
            smth_params = np.polyfit(self.bar_time, self.bar_deriv, smth_order)
            bar_pos_func = np.poly1d(smth_params)
            self.bar_deriv = bar_pos_func(self.bar_time)

        if (fft_order):
            self.bar_deriv = self.bar_deriv
            
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

        for i in range(0,len(self.bar_time)):
            print >>f,self.bar_time[i],self.bar_upos[i],self.bar_deriv[i]

        f.close()
 
    def place_ellipse(self):

        return None

    def read_bar(self,infile):

        #
        # read a printed bar file
        #

        f = open(infile)

        bar_time = []
        bar_pos = []
        bar_deriv = []
        for line in f:
            q = [float(d) for d in line.split()]
            bar_time.append(q[0])
            bar_pos.append(q[1])
            try:
                bar_deriv.append(q[2])
            except:
                pass

        self.bar_time = np.array(bar_time)
        self.bar_pos = np.array(bar_pos)
        self.bar_upos = np.array(bar_pos)
        self.bar_deriv = np.array(bar_deriv)

        if len(self.bar_deriv < 1):

            BarDetermine.frequency_and_derivative(self)

    def find_barangle(self,time,bartime,barpos):

        #
        # helper class to find the position of the bar at specified times
        #
        
        try:
            tmp = self.bar_pos[0]

            try:
                indx_barpos = np.zeros([len(time)])
                for indx,timeval in enumerate(time):
                    indx_barpos[indx] = -self.bar_pos[ abs(timeval-self.bar_time).argmin()]
            except:
                indx_barpos = -self.bar_pos[ abs(time-self.bar_time).argmin()]

            return indx_barpos

            
        except:
            print 'BarDetermine.find_barangle: Requires BarDetermine.read_bar or BarDetermine.detect_bar to run.'
        







#
# reading in arrays also possible!
#
def read_trapping_file(t_file):
    f = open(t_file,'rb')
    [norb,ntime] = np.fromfile(f,dtype='i',count=2)
    bar_times = np.fromfile(f,dtype='f',count=ntime)
    #tarr = np.arange(tbegin,tend,dt)
    #
    trap_tmp = np.fromfile(f,dtype='i2',count=norb*ntime)
    trap_array = trap_tmp.reshape([norb,ntime])
    return bar_times,trap_array






            


class Trapping():

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
        
        Trapping.parse_list(self)
        

    def parse_list(self):
        
        f = open(self.slist)
        s_list = []
        for line in f:
            d = [q for q in line.split()]
            s_list.append(d[0])

        self.SLIST = np.array(s_list)

        if self.verbose >= 1:
            print 'Trapping.parse_list: Accepted %i files.' %len(self.SLIST)

    
    def determine_r_aps(self,filelist,comp,nout=10,to_file=1,transform=False,out_directory='',threedee=False):

        #
        # need to think of the best way to return this data
        #

        #
        # two separate modes, to_file=1,2
        #

        self.slist = filelist


        if (to_file > 0):
            tstamp = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d+%H:%M:%S')
            f = open(out_directory+'apshold'+tstamp+'.dat','wb+')

        Trapping.parse_list(self)

        if (to_file==2):
            
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

            if transform:
                B = BarDetermine()
                bpos = B.bar_fourier_compute(Ob.xpos,Ob.ypos)
                Ob.tX = Ob.xpos*np.cos(bpos) - Ob.ypos*np.sin(bpos)
                Ob.tY = Ob.xpos*np.sin(bpos) + Ob.ypos*np.cos(bpos)

            
            indx = np.array([i for i in range(0,len(Ob.xpos))])

            if transform:
                x = Ob.tX[aps]
                y = Ob.tY[aps]
            else:
                x = Ob.xpos[aps]
                y = Ob.ypos[aps]
                
            z = Ob.zpos[aps]
            numi = indx[aps]

            norb = len(numi)



            if (to_file==1):
                np.array( [Ob.ctime],dtype='f').tofile(f)
                np.array( [norb],dtype='i').tofile(f)
                for j in range(0,norb):
                    q = np.array([numi[j],x[j],y[j],z[j]],dtype='f')
                    #q = np.array([ [i[j],x[j],y[j],z[j]] for j in range(0,norb)],dtype='f')
                    q.tofile(f)

            if (to_file==2):

                for j in range(0,norb):
                    aps_dictionary[numi[j]].append([Ob.ctime,x[j],y[j],z[j]])


                
            else:

                print 'Trapping.determine_r_aps: Are you sure you want to dump to memory?'

        if (to_file==1): f.close()

        self.napsides = np.zeros([total_orbits,2])

        if (to_file==2):

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

    def read_apshold_two(self,apshold_file):

        f = open(apshold_file,'rb')

        [norb] = np.fromfile(f,dtype='i',count=1)

        self.aps = {}

        for i in range(norb):
            
            [naps] = np.fromfile(f,dtype='i',count=1)
            
            if naps > 0:
    
                aps_array = np.fromfile(f,dtype='f',count=4*naps)
                
                self.aps[i] = aps_array.reshape([naps,4])


        f.close()

        

    def read_apshold_one(self,apshold_file,comp):

        # read in the apshold to a dictionary

        self.aps = {}

        Oa = psp_io.Input(self.SLIST[0],comp=comp,verbose=self.verbose)
        norb = len(Oa.xpos)

        for i in range(norb):
            self.aps[i] = []


        f = open(apshold_file,'rb')

        more = 1

        
        while more == 1:

            try:
                [ct] = np.fromfile(f,dtype='f',count=1)
                print 'Read succesful line at %4.3f' %ct
                [naps] = np.fromfile(f,dtype='i',count=1)

                for n in range(0,naps):
                    [tmpi,tmpx,tmpy,tmpz] = np.fromfile(f,dtype='f',count=4)
                    self.aps[tmpi].append([ct,tmpx,tmpy,tmpz])

            except:

                more = 0

        # would also be cool to print a per orbit self-describing format

        
    def find_bar_coherance(self,BarDetermineInstance,N=20):

        #
        # two paths to go--trapping for all orbits as a function of time
        #   or individual time, is the orbit trapped. but want a faster way to handle the aps file in this case
        '''
        bposes = find_barangle(A.aps[indx][:,0],B.bar_time,B.bar_pos)
        txa = A.aps[indx][:,1]*np.cos(bposes) - A.aps[indx][:,2]*np.sin(bposes)
        tya = -A.aps[indx][:,1]*np.sin(bposes) - A.aps[indx][:,2]*np.cos(bposes)

        relpos2 = np.arctan(tya/txa)
        '''
            


