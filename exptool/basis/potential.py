#############################################
#
#.______     ______   .___________. _______ .__   __. .___________. __       ___       __      
#|   _  \   /  __  \  |           ||   ____||  \ |  | |           ||  |     /   \     |  |     
#|  |_)  | |  |  |  | `---|  |----`|  |__   |   \|  | `---|  |----`|  |    /  ^  \    |  |     
#|   ___/  |  |  |  |     |  |     |   __|  |  . `  |     |  |     |  |   /  /_\  \   |  |     
#|  |      |  `--'  |     |  |     |  |____ |  |\   |     |  |     |  |  /  _____  \  |  `----.
#| _|       \______/      |__|     |_______||__| \__|     |__|     |__| /__/     \__\ |_______|
#                                                                                              
#  potential.py
#    An exptool utility to handle energy and kappa calculations
#
#    MSP 10.1.2015
#
#
# http://patorjk.com/software/taag/#p=display&f=Star%20Wars&t=potential
#
'''
.______     ______   .___________. _______ .__   __. .___________. __       ___       __      
|   _  \   /  __  \  |           ||   ____||  \ |  | |           ||  |     /   \     |  |     
|  |_)  | |  |  |  | `---|  |----`|  |__   |   \|  | `---|  |----`|  |    /  ^  \    |  |     
|   ___/  |  |  |  |     |  |     |   __|  |  . `  |     |  |     |  |   /  /_\  \   |  |     
|  |      |  `--'  |     |  |     |  |____ |  |\   |     |  |     |  |  /  _____  \  |  `----.
| _|       \______/      |__|     |_______||__| \__|     |__|     |__| /__/     \__\ |_______|
potential (part of exptool.basis)
    construct instances that are combinations of different components



TODO:
1. add support for energy/kappa dimensionality conversion



'''

# python3 compatibility...
from __future__ import absolute_import, division, print_function, unicode_literals



# standard libraries
import numpy as np
import time

# exptool classes
from exptool.utils import utils
from exptool.io import psp_io
from exptool.analysis import pattern
from exptool.basis import eof
from exptool.basis import spheresl
from exptool.utils import halo_methods

# for interpolation
from scipy.interpolate import UnivariateSpline



class Fields():
    '''
    class to accumulate (all) particles from a dump and return the field quantities

    UNFORTUNATELY, this only works on halo+disk systems right now. Should offload the ability to plug in multiple components.

    '''
    def __init__(self,infile,eof_file,sph_file,model_file,nhalo=1000000,transform=False,no_odd=False,centering=False,mutual_center=False,verbose=1):
        '''
        __init__


        inputs
        --------------------
        infile
        eof_file
        sph_file
        model_file
        nhalo=1000000
        transform=False
        no_odd=False
        centering=False
        mutual_center=False
        verbose=1


        returns
        -------------------
        self, now set up with basic parameters

        '''

        self.infile = infile
        self.eof_file = eof_file
        self.sph_file = sph_file
        self.model_file = model_file
        self.nhalo = nhalo
        self.transform = transform
        self.no_odd = no_odd
        self.centering = centering
        self.mutual_center = mutual_center

        self.verbose = verbose

        

        # do the total coefficient calculation?
        #Fields.total_coefficients(self)

    def total_coefficients(self):
        '''
        total_coefficients


        inputs
        -----------------
        self


        returns
        -----------------
        self
            time
            


        '''


        # read in the files
        #
        PSPDumpDisk = psp_io.Input(self.infile,comp='star')

        # add for ability to tabulate
        self.time = PSPDumpDisk.time
        
        if self.transform:
            PSPDumpDiskTransformed = pattern.BarTransform(PSPDumpDisk)
            
            if self.verbose > 1:
                print('potential.Fields.total_coefficients: Using bar_angle {0:4.3f}'.format(PSPDumpDiskTransformed.bar_angle))
                
        else:
            # let's reread for safety
            PSPDumpDiskTransformed = psp_io.Input(self.infile,comp='star')



        # read in both partial and full halo to figure out the halofactor
        PSPDumpHaloT = psp_io.Input(self.infile,comp='dark')

        PSPDumpHalo = psp_io.Input(self.infile,comp='dark',nout=self.nhalo)


        self.halofac = float(PSPDumpHaloT.nbodies)/float(PSPDumpHalo.nbodies)

        
        if self.transform:
            PSPDumpHaloTransformed = pattern.BarTransform(PSPDumpHalo,bar_angle=PSPDumpDiskTransformed.bar_angle)
            
        else:
            PSPDumpHaloTransformed = PSPDumpHalo = psp_io.Input(self.infile,comp='dark',nout=self.nhalo)

        #
        # do centering
        if self.centering:

            print('potential.Fields.total_coefficients: Computing centering (centering=True)')

            # this should be adaptable at some point
            ncenter = 10000

            # rank order particles
            rrank = (PSPDumpDiskTransformed.xpos*PSPDumpDiskTransformed.xpos + \
                     PSPDumpDiskTransformed.ypos*PSPDumpDiskTransformed.ypos + \
                     PSPDumpDiskTransformed.zpos*PSPDumpDiskTransformed.zpos)**0.5

            cparticles = rrank.argsort()[0:ncenter]

            # use the specified particles to calculate the center of mass in each dimension
            self.xcen_disk = np.sum(PSPDumpDiskTransformed.xpos[cparticles]*PSPDumpDiskTransformed.mass[cparticles])/np.sum(PSPDumpDiskTransformed.mass[cparticles])
            self.ycen_disk = np.sum(PSPDumpDiskTransformed.ypos[cparticles]*PSPDumpDiskTransformed.mass[cparticles])/np.sum(PSPDumpDiskTransformed.mass[cparticles])
            self.zcen_disk = np.sum(PSPDumpDiskTransformed.zpos[cparticles]*PSPDumpDiskTransformed.mass[cparticles])/np.sum(PSPDumpDiskTransformed.mass[cparticles])

            # pinned both components to same position?
            if self.mutual_center:

                print('potential.Fields.total_coefficients: Using computed disk center for halo (mutual_center=True)')

                self.xcen_halo = self.xcen_disk
                self.ycen_halo = self.ycen_disk
                self.zcen_halo = self.zcen_disk



            else:
                

                # rank order particles
                rrank = (PSPDumpDiskTransformed.xpos*PSPDumpDiskTransformed.xpos + \
                     PSPDumpDiskTransformed.ypos*PSPDumpDiskTransformed.ypos + \
                     PSPDumpDiskTransformed.zpos*PSPDumpDiskTransformed.zpos)**0.5

                cparticles = rrank.argsort()[0:ncenter]

                self.xcen_halo = np.sum(PSPDumpHaloTransformed.xpos[cparticles]*PSPDumpHaloTransformed.mass[cparticles])/np.sum(PSPDumpHaloTransformed.mass[cparticles])
                self.ycen_halo = np.sum(PSPDumpHaloTransformed.ypos[cparticles]*PSPDumpHaloTransformed.mass[cparticles])/np.sum(PSPDumpHaloTransformed.mass[cparticles])
                self.zcen_halo = np.sum(PSPDumpHaloTransformed.zpos[cparticles]*PSPDumpHaloTransformed.mass[cparticles])/np.sum(PSPDumpHaloTransformed.mass[cparticles])


            print('potential.Fields.total_coefficients: (x,y,z) = {0:6.5f},{1:6.5f},{2:6.5f}'\
                  .format(float(self.xcen_disk),float(self.ycen_disk),float(self.zcen_disk)))

            PSPDumpDiskTransformed.xpos = PSPDumpDiskTransformed.xpos - self.xcen_disk
            PSPDumpDiskTransformed.ypos = PSPDumpDiskTransformed.ypos - self.ycen_disk
            PSPDumpDiskTransformed.zpos = PSPDumpDiskTransformed.zpos - self.zcen_disk
            
            PSPDumpHaloTransformed.xpos = PSPDumpHaloTransformed.xpos - self.xcen_halo
            PSPDumpHaloTransformed.ypos = PSPDumpHaloTransformed.ypos - self.ycen_halo
            PSPDumpHaloTransformed.zpos = PSPDumpHaloTransformed.zpos - self.zcen_halo


        else:
            self.xcen_disk = 0.
            self.ycen_disk = 0.
            self.zcen_disk = 0.

            self.xcen_halo = 0.
            self.ycen_halo = 0.
            self.zcen_halo = 0.
                     
        #
        # compute coefficients
        #
        self.EOF = eof.compute_coefficients(PSPDumpDiskTransformed,self.eof_file,verbose=self.verbose,no_odd=self.no_odd)

        
        self.SL = spheresl.compute_coefficients(PSPDumpHaloTransformed,self.sph_file,self.model_file,verbose=self.verbose,no_odd=self.no_odd)


        

    def prep_tables(self):
        '''
        prep_tables
            reads the cached files to set up tables for accumulation


        '''

        try:
            x = self.EOF.eof_file

        except:
            print('potential.Fields.prep_tables: must first call total_coefficients.')

        # build disk tables
        self.potC,self.rforceC,self.zforceC,self.densC,\
          self.potS,self.rforceS,self.zforceS,self.densS \
          = eof.parse_eof(self.EOF.eof_file)
          
        self.rmindisk,self.rmaxdisk,self.numx,self.numy,self.mmax,self.norder,self.ascale,self.hscale,self.cmapdisk,self.densdisk \
          = eof.eof_params(self.EOF.eof_file)
          
        self.XMIN,self.XMAX,self.dX,self.YMIN,self.YMAX,self.dY \
          = eof.set_table_params(RMAX=self.rmaxdisk,RMIN=self.rmindisk,ASCALE=self.ascale,HSCALE=self.hscale,NUMX=self.numx,NUMY=self.numy,CMAP=self.cmapdisk)

        self.disk_use_m = self.mmax
        self.disk_use_n = self.norder

        # build halo tables
        self.lmaxhalo,self.nmaxhalo,self.numrhalo,self.cmaphalo,\
          self.rminhalo,self.rmaxhalo,self.scalehalo,self.ltablehalo,self.evtablehalo,self.eftablehalo \
          = halo_methods.read_cached_table(self.SL.sph_file)
          
        self.xihalo,self.rarrhalo,self.p0halo,self.d0halo \
          = halo_methods.init_table(self.SL.model_file,self.numrhalo,self.rminhalo,self.rmaxhalo,cmap=self.cmaphalo,scale=self.scalehalo)  

        self.halo_use_l = self.lmaxhalo
        self.halo_use_n = self.nmaxhalo

    def return_density(self,xval,yval,zval):
        '''
        definition to return the density for the monopole term and total separately, and for the two components

        wrapped elsewhere to some end

        '''
        
        try:
            x = self.EOF.eof_file
            y = self.potC

        except:
            print('potential.Fields.return_density: must first call total_coefficients and prep_tables.')

        r2val = (xval*xval + yval*yval)**0.5  + 1.e-10
        r3val = (r2val*r2val + zval*zval)**0.5  + 1.e-10
        costh = zval/r3val
        phival = np.arctan2(yval,xval)

            
        # disk evaluation call
        diskp0,diskp,diskfr,diskfp,diskfz,diskden0,diskden1 = eof.accumulated_eval(r2val, zval, phival,\
                                              self.EOF.cos, self.EOF.sin,\
                                              self.potC, self.rforceC, self.zforceC, self.densC,\
                                              self.potS, self.rforceS, self.zforceS, self.densS,\
                                              rmin=self.XMIN,dR=self.dX,zmin=self.YMIN,dZ=self.dY,numx=self.numx,numy=self.numy,fac = 1.0,\
                                              MMAX=self.mmax,NMAX=self.norder,\
                                              ASCALE=self.ascale,HSCALE=self.hscale,CMAP=self.cmapdisk,no_odd=self.no_odd)
        #
        # halo evaluation call
        haloden0,haloden1,halopot0,halopot1,halopotr,halopott,halopotp = spheresl.all_eval(r3val, costh, phival,\
                                                                                           self.halofac*self.SL.expcoef,\
                                                                                           self.xihalo,self.p0halo,self.d0halo,\
                                                                                           self.cmaphalo,self.scalehalo,\
                                                                                           self.lmaxhalo,self.nmaxhalo,\
                                                                                           self.evtablehalo,self.eftablehalo,no_odd=self.no_odd)

        return haloden0,haloden1,diskden0,diskden1


    def density_calculate(self,rvals=np.linspace(0.,0.1,100)):
        '''
        routine to compute in-plane major axis density values for looking at profile change over time

        '''

        # cheap to just do this here and set everything up as a just-in-case
        Fields.prep_tables(self)

        if not self.densdisk:
            print('Fields.density_calculate: no density terms in basis!')
            return None

        halodens_mono = np.zeros_like(rvals)
        diskdens_mono = np.zeros_like(rvals)
        halodens_total = np.zeros_like(rvals)
        diskdens_total = np.zeros_like(rvals)

        for indx,rval in enumerate(rvals):
            halodens_mono[indx],halodens_total[indx],diskdens_mono[indx],diskdens_total[indx] = Fields.return_density(self,rval,0.0,0.0)

        self.rvals = rvals
        self.halodens_mono = halodens_mono
        self.diskdens_mono = diskdens_mono
        self.halodens_total = halodens_total
        self.diskdens_total = diskdens_total


    def set_field_parameters(self,no_odd=False,halo_l=-1,halo_n=-1,disk_m=-1,disk_n=-1):
        '''
        in preparation for other definitions, specify


        '''
        
        self.no_odd = no_odd

        if halo_l > -1: self.halo_use_l = halo_l
        if halo_n > -1: self.halo_use_n = halo_n
        if disk_m > -1: self.disk_use_m = disk_m
        if disk_n > -1: self.disk_use_n = disk_n


    def reset_field_parameters(self):

        self.no_odd = False
        self.halo_use_l = self.lmaxhalo
        self.halo_use_n = self.nmaxhalo
        self.disk_use_m = self.mmax
        self.disk_use_n = self.norder
    

    def return_forces_cyl(self,xval,yval,zval,rotpos=0.0):
        
        try:
            x = self.no_odd

        except:
            print('Fields.return_forces_cart: applying default potential parameters.')
            Fields.set_field_parameters(self)


        r2val = (xval*xval + yval*yval)**0.5  + 1.e-10
        r3val = (r2val*r2val + zval*zval)**0.5  + 1.e-10
        costh = zval/r3val
        phival = np.arctan2(yval,xval)


        #
        # disk force call
        diskfr,diskfp,diskfz,diskp,diskp0 = eof.force_eval(r2val, zval, phival + rotpos, \
                                                      self.EOF.cos, self.EOF.sin, \
                                                      self.potC, self.rforceC, self.zforceC,\
                                                      self.potS, self.rforceS, self.zforceS,\
                                                      rmin=self.XMIN,dR=self.dX,zmin=self.YMIN,dZ=self.dY,numx=self.numx,numy=self.numy,fac = 1.0,\
                                                      MMAX=self.disk_use_m,NMAX=self.disk_use_n,\
                                                      #MMAX=self.mmax,NMAX=self.norder,\
                                                      ASCALE=self.ascale,HSCALE=self.hscale,CMAP=self.cmapdisk,no_odd=self.no_odd,perturb=False)
        #
        # halo force call
        halofr,haloft,halofp,halop,halop0 = spheresl.force_eval(r3val, costh, phival + rotpos, \
                                                   self.halofac*self.SL.expcoef,\
                                                   self.xihalo,self.p0halo,self.d0halo,self.cmaphalo,self.scalehalo,\
                                                   self.halo_use_l,self.halo_use_n,\
                                                   #self.lmaxhalo,self.nmaxhalo,\
                                                   self.evtablehalo,self.eftablehalo,no_odd=self.no_odd)
                                                   
        # recommended guards against bizarre phi forces

        # do we need any other guards?
        if r3val < np.min(self.xihalo):
            halofp = 0.
            diskfp = 0.

        # convert halo to cylindrical coordinates
        frhalo = -1.*(r2val*halofr + zval*haloft)/r3val
                
        fzhalo = -1.*(zval*halofr - r2val*haloft)/r3val

        # this is now returning the total potential in both disk and halo case
        return diskfr,frhalo,diskfp,-1.*halofp,diskfz,fzhalo,diskp,(halop + halop0)


            

    def return_forces_cart(self,xval,yval,zval,rotpos=0.0):
        
        try:
            x = self.no_odd

        except:
            print('Fields.return_forces_cart: applying default potential parameters.')
            Fields.set_field_parameters(self)


        r2val = (xval*xval + yval*yval)**0.5  + 1.e-10
        r3val = (r2val*r2val + zval*zval)**0.5  + 1.e-10
        costh = zval/r3val
        phival = np.arctan2(yval,xval)


        #
        # disk force call
        diskfr,diskfp,diskfz,diskp,diskp0 = eof.force_eval(r2val, zval, phival + rotpos, \
                                                      self.EOF.cos, self.EOF.sin, \
                                                      self.potC, self.rforceC, self.zforceC,\
                                                      self.potS, self.rforceS, self.zforceS,\
                                                      rmin=self.XMIN,dR=self.dX,zmin=self.YMIN,dZ=self.dY,numx=self.numx,numy=self.numy,fac = 1.0,\
                                                      MMAX=self.disk_use_m,NMAX=self.disk_use_n,\
                                                      #MMAX=self.mmax,NMAX=self.norder,\
                                                      ASCALE=self.ascale,HSCALE=self.hscale,CMAP=self.cmapdisk,no_odd=self.no_odd,perturb=False)
        #
        # halo force call
        halofr,haloft,halofp,halop,halop0 = spheresl.force_eval(r3val, costh, phival + rotpos, \
                                                   self.halofac*self.SL.expcoef,\
                                                   self.xihalo,self.p0halo,self.d0halo,self.cmaphalo,self.scalehalo,\
                                                   self.halo_use_l,self.halo_use_n,\
                                                   #self.lmaxhalo,self.nmaxhalo,\
                                                   self.evtablehalo,self.eftablehalo,no_odd=self.no_odd)
                                                   
        # recommended guards against bizarre phi forces

        # do we need any other guards?
        if r3val < np.min(self.xihalo):
            halofp = 0.
            diskfp = 0.
        
        fxdisk = (diskfr*(xval/r2val) - diskfp*(yval/(r2val*r2val)) )
        fxhalo = -1.* ( halofr*(xval/r3val) - haloft*(xval*zval/(r3val*r3val*r3val))) + halofp*(yval/(r2val*r2val))
        
        fydisk = (diskfr*(yval/r2val) + diskfp*(xval/(r2val*r2val)) )
        fyhalo = -1.* ( halofr*(yval/r3val) - haloft*(yval*zval/(r3val*r3val*r3val))) - halofp*(xval/(r2val*r2val))
        
        fzdisk = diskfz
        fzhalo = -1.* ( halofr*(zval/r3val) + haloft*( (r2val*r2val)/(r3val*r3val*r3val)) )

        # this is now returning the total potential in both disk and halo case
        return fxdisk,fxhalo,fydisk,fyhalo,fzdisk,fzhalo,diskp,(halop + halop0)

    
    def rotation_curve(self,rvals=np.linspace(0.0001,0.1,100),mono=False,angle=0.):
        '''
        returns the rotation curve alone the x axis for quick and dirty viewing. rotate potential first if desired!


        inputs
        --------------
        self           :                            the Field instance
        rvals          : (default=sampling to 0.1)  what rvalues to evaluate
        mono           : (default=False)            use only the monopole?

        returns
        --------------
        additions to Field instance--

        disk_rotation  :                            disk contribution to the rotation curve
        halo_rotation  :                            halo contribution to the rotation curve
        total_rotation :                            the total rotation curve

        
        '''

        try:
            x = self.no_odd

        except:
            print('Fields.return_forces_cart: applying default potential parameters.')
            Fields.set_field_parameters(self)



        if mono == True:
            tmp_halo = self.halo_use_l
            tmp_disk = self.disk_use_m

            # zero out to get monopole only
            self.halo_use_l = 0
            self.disk_use_m = 0



        
        disk_force = np.zeros_like(rvals)
        halo_force = np.zeros_like(rvals)

        for indx,rval in enumerate(rvals):
            disk_force[indx],halo_force[indx],a,b,c,d,e,f = Fields.return_forces_cyl(self,rval,angle,0.0)

        self.rvals = rvals
        self.disk_dpdr = -1.*disk_force
        self.halo_dpdr = -1.*halo_force
        self.total_dpdr = -1.*disk_force + -1.*halo_force
        self.disk_rotation = (rvals*abs(disk_force))**0.5
        self.halo_rotation = (rvals*abs(halo_force))**0.5
        self.total_rotation = (rvals*(abs(halo_force)+abs(disk_force)))**0.5


        if mono == True:
            # reset values
            self.halo_use_l = tmp_halo
            self.disk_use_m = tmp_disk



    def resonance_positions(self,rvals=np.linspace(0.0001,0.1,100),mono=False):
        '''
        calculate simple resonance lines for modeling purposes

        inputs
        --------------
        self   :                            the Field instance
        rvals  : (default=sampling to 0.1)  what rvalues to evaluate
        mono   : (default=False)            use only the monopole?

        returns
        --------------
        additions to Field instance--

        rvals
        omega
        kappa

        '''

        try:
            x = self.no_odd

        except:
            print('Fields.return_forces_cart: applying default potential parameters.')
            Fields.set_field_parameters(self)



        if mono == True:
            tmp_halo = self.halo_use_l
            tmp_disk = self.disk_use_m

            # zero out to get monopole only
            self.halo_use_l = 0
            self.disk_use_m = 0


        disk_force = np.zeros_like(rvals)
        halo_force = np.zeros_like(rvals)

        for indx,rval in enumerate(rvals):
            disk_force[indx],halo_force[indx],a,b,c,d,e,f = Fields.return_forces_cyl(self,rval,0.0,0.0)

        self.rvals = rvals

        # circular frequency
        self.omega = ((abs(halo_force)+abs(disk_force))/rvals)**0.5

        # radial frequency
        # do the derivative
        spl = UnivariateSpline(rvals, self.omega, k=3, s=0)
        ddphi = spl.derivative()(rvals)

        self.kappa = ( 3.*self.omega**2. + ddphi)**0.5

        if mono == True:
            # reset values
            self.halo_use_l = tmp_halo
            self.disk_use_m = tmp_disk



    def compute_axis_potential(self,rvals=np.linspace(0.,0.1,100)):
        '''
        returns the potential along the major axis

        this is kind of dumb and needs a revamp. Throwing away tons of information.

        '''

        disk_pot = np.zeros_like(rvals)
        halo_pot = np.zeros_like(rvals)

        for indx,rval in enumerate(rvals):
            a,b,c,d,e,f,disk_pot[indx],halo_pot[indx] = Fields.return_forces_cart(self,rval,0.0,0.0)

        self.rvals = rvals
        self.disk_pot = disk_pot
        self.halo_pot = halo_pot
        self.total_pot = disk_pot+halo_pot


    def make_force_grid(self,rline = np.linspace(0.00022,0.1,100),thline = np.linspace(0.00022,2.*np.pi,50)):
        '''
        make_force_grid: evaluate a simple grid of points along an axis

        inputs
        ---------
        self   : Fields instance
        rline  : spacing in radius to probe
        thline : spacing in theta to probe
        
        returns
        ---------
        wake   : dictionary with the following keys
           R   : radius grid
           T   : theta grid
           P   : potential grid
           D   : density grid
           tfR : total radial force grid
           dfR : DISK  radial force grid
           hfR : HALO  radial force grid

        '''

        
        rgrid,thgrid = np.meshgrid(rline,thline)
        
        
        P = psp_io.particle_holder()
        P.xpos = (rgrid*np.cos(thgrid)).reshape(-1,)
        P.ypos = (rgrid*np.sin(thgrid)).reshape(-1,)
        P.zpos = np.zeros(rgrid.size)
        P.mass = np.zeros(rgrid.size)

        # the only way to do even-only calculation with these is to wipe out the odd terms from the coefficients (do-able)

        # for disk
        cos_coefs_in = np.copy(self.EOF.cos)
        sin_coefs_in = np.copy(self.EOF.sin)
        #
        if self.no_odd:
            for i in range(1,self.EOF.mmax,2):
                cos_coefs_in[i] = np.zeros(self.EOF.nmax)
                sin_coefs_in[i] = np.zeros(self.EOF.nmax)


        # if using a restored file, potC etc may already exist. checking...
        try:
            p0,p,d0,d,fr,fp,fz,R = eof.accumulated_eval_particles(P, cos_coefs_in, sin_coefs_in ,\
                                                                potC=self.potC, rforceC=self.rforceC, zforceC=self.zforceC, \
                                                                potS=self.potS, rforceS=self.rforceS, zforceS=self.zforceS, \
                                                                rmin=self.rmindisk,dR=0,zmin=self.zmindisk,dZ=0,numx=0,numy=0,MMAX=6,NMAX=18,\
                                                                ASCALE=0.0,HSCALE=0.0,CMAP=0,m1=0,m2=1000,verbose=0,density=False,eof_file='')

        except:
            p0,p,d0,d,fr,fp,fz,R = eof.accumulated_eval_particles(P, cos_coefs_in, sin_coefs_in ,m1=0,m2=self.disk_use_m,eof_file=self.EOF.eof_file,density=True)


        den0,den1,pot0,pot1,potr,pott,potp,rr = spheresl.eval_particles(P,self.halofac*self.SL.expcoef,self.SL.sph_file,self.SL.model_file,l1=0,l2=self.halo_use_l)

        halo_rforce = ( rr*potr + P.zpos*pott )/( rr**2. + P.zpos**2.)**0.5

        wake = {}
        wake['R'] = rgrid
        wake['T'] = thgrid

        wake['P'] = (p+pot1+p0+pot0).reshape([thline.shape[0],rline.shape[0]])
        wake['P1'] = (p+pot1).reshape([thline.shape[0],rline.shape[0]])
        wake['D'] = (d+den0+den1).reshape([thline.shape[0],rline.shape[0]])
        
        wake['tfR'] = (-1.*fr+halo_rforce).reshape([thline.shape[0],rline.shape[0]])
        wake['dfR'] = (-1.*fr).reshape([thline.shape[0],rline.shape[0]])
        wake['hfR'] = halo_rforce.reshape([thline.shape[0],rline.shape[0]])

        wake['tfP'] = (fp+potp).reshape([thline.shape[0],rline.shape[0]])
        wake['dfP'] = fp.reshape([thline.shape[0],rline.shape[0]])
        wake['hfP'] = potp.reshape([thline.shape[0],rline.shape[0]])

        wake['tfZ'] = fz.reshape([thline.shape[0],rline.shape[0]])
        wake['dfZ'] = fz.reshape([thline.shape[0],rline.shape[0]])

        wake['Rline'] = rline
        wake['Tline'] = thline
        
        self.wake = wake



    def save_field(self,filename=''):
        '''
        save_field
        ----------------
        print field quantities to file to restore quickly

        inputs
        ----------------
        self       : the Field instance
        filename   : default to add file number


        outputs
        ---------------
        printed field file, to be read with potential.restore_field(filename)

        '''

        if filename=='':
            print('potential.Fields.save_field: No filename specified.')
            
        f = open(filename,'wb')
        
        #####################################################
        # global parameters
        #self.infile
        np.array([self.infile],dtype='S100').tofile(f)

        #self.eof_file
        np.array([self.eof_file],dtype='S100').tofile(f)

        #self.sph_file
        np.array([self.sph_file],dtype='S100').tofile(f)

        #self.model_file
        np.array([self.model_file],dtype='S100').tofile(f)

        #[infile,eof_file,sph_file,model_file] = np.fromfile(f,dtype='S100',count=4)

        #self.nhalo
        np.array([self.nhalo],dtype='i4').tofile(f)

        #self.transform
        np.array([self.transform],dtype='i4').tofile(f)

        #self.no_odd
        np.array([self.no_odd],dtype='i4').tofile(f)

        #self.centering
        np.array([self.centering],dtype='i4').tofile(f)

        #self.mutual_center
        np.array([self.mutual_center],dtype='i4').tofile(f)

        #self.verbose
        np.array([self.verbose],dtype='i4').tofile(f)

        #[nhalo,transform,no_odd,centering,mutual_center,verbose] = np.fromfile(f,dtype='i4',count=6)
        
        #self.time
        np.array([self.time],dtype='f4').tofile(f)

        #[time] = np.fromfile(f,dtype='f4',count=1)

        
        ####################################################
        # EOF parameters

        #self.numx
        np.array([self.numx],dtype='i4').tofile(f)
     
        #self.numy
        np.array([self.numy],dtype='i4').tofile(f)

        #self.mmax
        np.array([self.mmax],dtype='i4').tofile(f)

        #self.norder
        np.array([self.norder],dtype='i4').tofile(f)

        #self.cmapdisk
        np.array([self.cmapdisk],dtype='i4').tofile(f)

        #self.densdisk
        np.array([self.densdisk],dtype='i4').tofile(f)

        
        #[numx,numy,mmax,norder,cmapdisk,densdisk] = np.fromfile(f,dtype='i4',count=6)

        #self.rmindisk
        np.array([self.rmindisk],dtype='f4').tofile(f)

        #self.rmaxdisk
        np.array([self.rmaxdisk],dtype='f4').tofile(f)

        #self.ascale
        np.array([self.ascale],dtype='f4').tofile(f)

        #self.hscale
        np.array([self.hscale],dtype='f4').tofile(f)

        #self.XMIN
        np.array([self.XMIN],dtype='f4').tofile(f)

        #self.dX
        np.array([self.dX],dtype='f4').tofile(f)

        #self.YMIN
        np.array([self.YMIN],dtype='f4').tofile(f)

        #self.dY
        np.array([self.dY],dtype='f4').tofile(f)

        #self.xcen_disk = 0.
        np.array([self.xcen_disk],dtype='f4').tofile(f)
        
        #self.ycen_disk = 0.
        np.array([self.ycen_disk],dtype='f4').tofile(f)

        #self.zcen_disk = 0.
        np.array([self.zcen_disk],dtype='f4').tofile(f)

        #[rmindisk,rmaxdisk,ascale,hscale,XMIN,dX,YMIN,dY,xcen_disk,ycen_disk,zcen_disk] = np.fromfile(f,dtype='f4',count=11)

        #self.EOF.cos
        #self.EOF.sin
        np.array(self.EOF.cos.reshape(-1,),dtype='f8').tofile(f)
        np.array(self.EOF.sin.reshape(-1,),dtype='f8').tofile(f)
        # 8 bytes X 2 arrays x (m+1) x n = 16(m+1)n bytes

        #EOF.cos = (np.fromfile(f,dtype='f8',count=(mmax+1)*norder)).reshape([(mmax+1),norder])
        #EOF.sin = (np.fromfile(f,dtype='f8',count=(mmax+1)*norder)).reshape([(mmax+1),norder])
   

        #self.potC
        np.array(self.potC.reshape(-1,),dtype='f8').tofile(f)
        # 8 bytes x (numx+1) x (numy+1) = 8(numx+1)(numy+1) bytes
        #potC = (np.fromfile(f,dtype='f8',count=(mmax+1)*norder)).reshape([(mmax+1),norder])
    
        #self.rforceC
        np.array(self.rforceC.reshape(-1,),dtype='f8').tofile(f)

        #self.zforceC
        np.array(self.zforceC.reshape(-1,),dtype='f8').tofile(f)

        #self.densC
        np.array(self.densC.reshape(-1,),dtype='f8').tofile(f)

        #self.potS
        np.array(self.potS.reshape(-1,),dtype='f8').tofile(f)

        #self.rforceS
        np.array(self.rforceS.reshape(-1,),dtype='f8').tofile(f)

        #self.zforceS
        np.array(self.zforceS.reshape(-1,),dtype='f8').tofile(f)

        #self.densS
        np.array(self.densS.reshape(-1,),dtype='f8').tofile(f)


        #########################################
        # SL parameters
        #self.halofac
        np.array([self.halofac],dtype='f4').tofile(f)

        #self.rminhalo
        np.array([self.rminhalo],dtype='f4').tofile(f)

        #self.rmaxhalo
        np.array([self.rmaxhalo],dtype='f4').tofile(f)

        #self.scalehalo
        np.array([self.scalehalo],dtype='f4').tofile(f)

        #self.xcen_halo = 0.
        np.array([self.xcen_halo],dtype='f4').tofile(f)
        
        #self.ycen_halo = 0.
        np.array([self.ycen_halo],dtype='f4').tofile(f)

        #self.zcen_halo = 0.
        np.array([self.zcen_halo],dtype='f4').tofile(f)

        #[halofac,rminhalo,rmaxhalo,scalehalo,xcen_halo,ycen_halo,zcen_halo] = np.fromfile(f,dtype='f4',count=7)
    
        #self.numrhalo
        np.array([self.numrhalo],dtype='i4').tofile(f)

        #self.cmaphalo
        np.array([self.cmaphalo],dtype='i4').tofile(f)
               
        #self.lmaxhalo
        np.array([self.lmaxhalo],dtype='i4').tofile(f)

        #self.nmaxhalo
        np.array([self.nmaxhalo],dtype='i4').tofile(f)

        #[numrhalo,cmaphalo,lmaxhalo,nmaxhalo] = np.fromfile(f,dtype='i4',count=4)

        #self.xihalo
        np.array(self.xihalo.reshape(-1,),dtype='f8').tofile(f)
        #xihalo = (np.fromfile(f,dtype='f8',count=numrhalo))
        
        #self.p0halo
        np.array(self.p0halo.reshape(-1,),dtype='f8').tofile(f)
        
        #self.d0halo
        np.array(self.d0halo.reshape(-1,),dtype='f8').tofile(f)

        #self.ltablehalo
        np.array(self.ltablehalo.reshape(-1,),dtype='f8').tofile(f)
        
        #self.evtablehalo
        np.array(self.evtablehalo.reshape(-1,),dtype='f8').tofile(f)
        
        #self.eftablehalo
        np.array(self.eftablehalo.reshape(-1,),dtype='f8').tofile(f)
    
        #self.SL.expcoef
        np.array(self.SL.expcoef.reshape(-1,),dtype='f8').tofile(f)
        # 8 bytes X 2 arrays x (m+1) x n = 16(m+1)n bytes to end of array
    


        f.close()

        


def restore_field(filename=''):
    '''
    restore_field
    ----------------
    read in a Fields instance


    '''
    f = open(filename,'rb')

    ###########################
    # global block
    [infile,eof_file,sph_file,model_file] = np.fromfile(f,dtype='S100',count=4)
    [nhalo,transform,no_odd,centering,mutual_center,verbose] = np.fromfile(f,dtype='i4',count=6)
    [time] = np.fromfile(f,dtype='f4',count=1)

    F = Fields(infile,eof_file,sph_file,model_file,nhalo=nhalo,transform=transform,no_odd=no_odd,centering=centering,mutual_center=mutual_center,verbose=verbose)

    # somehow this doesn't get carried over normally?
    F.time = time

    ###########################
    # EOF block
    [F.numx,F.numy,F.mmax,F.norder,F.cmapdisk,F.densdisk] = np.fromfile(f,dtype='i4',count=6)
    [F.rmindisk,F.rmaxdisk,F.ascale,F.hscale,F.XMIN,F.dX,F.YMIN,F.dY,F.xcen_disk,F.ycen_disk,F.zcen_disk] = np.fromfile(f,dtype='f4',count=11)

    F.EOF = eof.EOF_Object()
    F.EOF.cos = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder)).reshape([(F.mmax+1),F.norder])
    F.EOF.sin = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder)).reshape([(F.mmax+1),F.norder])

    F.potC = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])
    F.rforceC = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])
    F.zforceC = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])
    F.densC = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])

    F.potS = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])
    F.rforceS = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])
    F.zforceS = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])
    F.densS = (np.fromfile(f,dtype='f8',count=(F.mmax+1)*F.norder*(F.numx+1)*(F.numy+1))).reshape([(F.mmax+1),F.norder,(F.numx+1),(F.numy+1)])

    #############################
    # SL block
    [F.halofac,F.rminhalo,F.rmaxhalo,F.scalehalo,F.xcen_halo,F.ycen_halo,F.zcen_halo] = np.fromfile(f,dtype='f4',count=7)

    [F.numrhalo,F.cmaphalo,F.lmaxhalo,F.nmaxhalo] = np.fromfile(f,dtype='i4',count=4)

    F.xihalo = (np.fromfile(f,dtype='f8',count=F.numrhalo))
    F.p0halo = (np.fromfile(f,dtype='f8',count=F.numrhalo))
    F.d0halo = (np.fromfile(f,dtype='f8',count=F.numrhalo))
    F.ltable = (np.fromfile(f,dtype='f8',count=(F.lmaxhalo+1)))

    F.evtablehalo = (np.fromfile(f,dtype='f8',count=(F.lmaxhalo+1)*(F.nmaxhalo+1))).reshape([(F.lmaxhalo+1),(F.nmaxhalo+1)])
    F.eftablehalo = (np.fromfile(f,dtype='f8',count=(F.lmaxhalo+1)*(F.nmaxhalo+1)*(F.numrhalo))).reshape([(F.lmaxhalo+1),(F.nmaxhalo+1),(F.numrhalo)])

    F.SL = spheresl.SL_Object()
    F.SL.expcoef = (np.fromfile(f,dtype='f8',count=(F.lmaxhalo+1)*(F.lmaxhalo+1)*(F.nmaxhalo+1))).reshape([(F.lmaxhalo+1)*(F.lmaxhalo+1),(F.nmaxhalo+1)])

    F.disk_use_m = F.mmax
    F.disk_use_n = F.norder

    F.halo_use_l = F.lmaxhalo
    F.halo_use_n = F.nmaxhalo

    # should restore to point just after F.prep_tables()
    f.close()

    # compatibility doubling
    F.SL.model_file = F.model_file
    F.SL.sph_file = F.sph_file
    F.EOF.eof_file = F.eof_file
    F.EOF.mmax = F.mmax

    return F

        

class EnergyKappa():

    #
    # class to look at energy-kappa mapping
    #
    def __init__(self,ParticleArray,nbins=200,map_file=None,eres=80,percen=99.5,spline_order=3):


        self.PA = PArray()

        # check to see if this is single or multi-timestep
        try:
            self.PA.MASS = ParticleArray.MASS
            self.PA.XPOS = ParticleArray.XPOS
            self.PA.YPOS = ParticleArray.YPOS
            self.PA.ZPOS = ParticleArray.ZPOS
            self.PA.XVEL = ParticleArray.XVEL
            self.PA.YVEL = ParticleArray.YVEL
            self.PA.ZVEL = ParticleArray.ZVEL
            self.PA.POTE = ParticleArray.POTE
            self.multitime = True
        except:
            self.PA.TIME = ParticleArray.time
            self.PA.MASS = ParticleArray.mass
            self.PA.XPOS = ParticleArray.xpos
            self.PA.YPOS = ParticleArray.ypos
            self.PA.ZPOS = ParticleArray.zpos
            self.PA.XVEL = ParticleArray.xvel
            self.PA.YVEL = ParticleArray.yvel
            self.PA.ZVEL = ParticleArray.zvel
            self.PA.POTE = ParticleArray.pote
            self.multitime = False

        self.nbins = nbins

        

        EnergyKappa.map_ekappa(self,percen=percen,eres=eres,spline_order=spline_order)

        if map_file:
            EnergyKappa.output_map(self,map_file)


    def map_ekappa(self,percen=99.9,eres=80,twodee=False,spline_order=3,smethod='sg'):

        # enables plotting of
        # self.
        #      Energy    :    the index
        #      Kappa     :    LZ/LZ_max for each orbit
        #      E         :    energy for each orbit
        #      maxLZ     :    relation to Energy for a circular (planar) orbit
        #      maxL      :    relation to Energy for a spherical orbit
        #      maxR      :    relation to Energy for radius in spherical bins

        # squared velocity
        V2 = (self.PA.XVEL*self.PA.XVEL + self.PA.YVEL*self.PA.YVEL + self.PA.ZVEL*self.PA.ZVEL)

        # angular momentum evaluation
        LX = self.PA.YPOS*self.PA.ZVEL - self.PA.ZPOS*self.PA.YVEL
        LY = self.PA.ZPOS*self.PA.XVEL - self.PA.XPOS*self.PA.ZVEL
        LZ = self.PA.XPOS*self.PA.YVEL - self.PA.YPOS*self.PA.XVEL
        L = (LX*LX + LY*LY + LZ*LZ)**0.5

        # total energy (to be made accessible)
        self.Energy = 0.5*V2 + self.PA.POTE

        #
        # should think about 2d vs 3d utility
        #
        if twodee:
            R = (self.PA.XPOS*self.PA.XPOS + self.PA.YPOS*self.PA.YPOS)**0.5
                
        else:
            R = (self.PA.XPOS*self.PA.XPOS + self.PA.YPOS*self.PA.YPOS + self.PA.ZPOS*self.PA.ZPOS)**0.5

        # partition particles into Energy bins
        self.Ebins = np.linspace(0.999*np.min(self.Energy),np.min([-2.5,1.001*np.max(self.Energy)]),eres)
        eindx = np.digitize(self.Energy,self.Ebins)

        # allocate arrays
        self.maxLz = np.zeros_like(self.Ebins)
        self.maxR = np.zeros_like(self.Ebins)
        self.maxL = np.zeros_like(self.Ebins)
        self.circR = np.zeros_like(self.Ebins)

        
        for i,energy in enumerate(self.Ebins):
            energy_range = np.where( eindx==i+1)[0]
            
            if len(energy_range) > 1:
                # reduce operations for speed
                #maxLx[i] = np.percentile(LX[yese],percen)
                #maxLy[i] = np.percentile(LY[yese],percen)
                self.maxLz[i] = np.percentile(LZ[energy_range],percen) #np.max(LZ[yese])

                # take median of top 100 Lz for guiding center radius
                #   (that is, radius of a circular orbit)
                lzarg = energy_range[LZ[energy_range].argsort()]
                self.circR[i] = np.median( R[lzarg[-100:-1]] )
                
                self.maxR[i] = np.percentile(R[energy_range],percen)
                self.maxL[i] = np.percentile(L[energy_range],percen)
                
            else: # guard for empty bins
                #maxLx[i] = maxLx[i-1]
                #maxLy[i] = maxLy[i-1]
                self.maxLz[i] = self.maxLz[i-1]
                self.maxR[i] = self.maxR[i-1]
                self.maxL[i] = self.maxL[i-1]
                self.circR[i] = self.circR[i-1]

        # smooth discontinuities from bin choice
        if smethod == 'sg':
            smthLz = helpers.savitzky_golay(self.maxLz,7,3) # could figure out an adaptive smooth?
            smthL = helpers.savitzky_golay(self.maxL,7,3)
            smthR = helpers.savitzky_golay(self.circR,7,3)
        else:
            smthLzf = UnivariateSpline(self.Ebins,self.maxLz,k=spline_order)
            smthLf  = UnivariateSpline(self.Ebins,self.maxL,k=spline_order)
            smthRf  = UnivariateSpline(self.Ebins,self.circR,k=spline_order)
            smthLz = smthLzf(self.Ebins)
            smthL  = smthLf(self.Ebins)
            smthR  = smthRf(self.Ebins)

        
        # return energy and kappa for all orbits
        if smethod == 'sg':
            self.Kappa = LZ/smthLz[eindx-1]
            self.Beta = L/smthL[eindx-1]
            self.cR = smthR[eindx-1]
        else:
            self.Kappa = LZ/smthLzf(self.Energy)
            self.Beta = L/smthLf(self.Energy)
            self.cR = smthRf(self.Energy)
        self.LZ = LZ
        self.L = L

    def clear_output_map_file(self):
        return None

    def output_map(self,file):

        #
        # helper class to 
        #
        f = open(file,'w+')
        #print >>f,PA.TIME,len(self.Ebins),self.Ebins,self.maxLz,self.maxL,self.maxR,self.circR

        f.close()


    def ek_grid(self,eres=80,kres=80,set_ebins=True,ebins_in=None):

        self.Kbins = np.linspace(-1.,1.,kres)
        self.Ebins = self.Ebins

        if not (set_ebins):
            self.Ebins = ebins_in

        self.Eindx = np.digitize(self.Energy,self.Ebins)
        self.Kindx = np.digitize(self.Kappa,self.Kbins)


    def sum_ek_values(self,sumval):

        # sumval is an input of the same lengths as self.Eindx


            
        # has ek_grid already been run?
        #     if not, run it.
        try:
            x = self.Kindx[0]
        except:
            print('exptool.potential.sum_ek_values: Making Grid...')
            EnergyKappa.ek_grid(self)

        if len(sumval)!=len(self.Eindx):
            print('exptool.potential.sum_ek_values: Input array must have values for all particles.')
            #break

        ebmax = len(self.Ebins)
        kbmax = len(self.Kbins)

        self.EKarray = np.zeros([ebmax,kbmax])
        self.Egrid,self.Kgrid = np.meshgrid(self.Ebins,self.Kbins)

        for i in range(0,len(self.Eindx)):
            if (self.Eindx[i] < ebmax) and (self.Kindx[i] < kbmax):
                self.EKarray[self.Eindx[i],self.Kindx[i]] += sumval[i]






#
# this is EXCLUSIVELY temporary until a better format is decided on
#
def get_fields(simulation_directory,simulation_name,intime,eof_file,sph_file,model_file,bar_bonus='',nhalo=1000000,transform=False,bar_file=''):
    '''
    input
    -----------------------------------
    simulation_directory     :
    simulation_name          :
    intime                   :
    eof_file                 :
    sph_file                 :
    model_file               :
    bar_bonus=''             :
    nhalo=1000000            :

    returns
    ----------------------------------
    F                        :
    pattern                  :
    rotfreq                  :

    '''
    infile = simulation_directory+'OUT.'+simulation_name+'.%05i' %intime
    BarInstance = pattern.BarDetermine()

    if transform:
        BarInstance.read_bar(bar_file)
            
    	# reset the derivative
        BarInstance.frequency_and_derivative(spline_derivative=2)
    
        PSPDump = psp_io.Input(infile,legacy=False)
	#validate=True)
    
        patt = pattern.find_barpattern(PSPDump.time,BarInstance,smth_order=None)
    
        rotfreq = patt/(2.*np.pi)

    else:
        patt = 0.
        rotfreq = 0.
    
    F = Fields(infile,eof_file,sph_file,model_file,nhalo=nhalo,transform=transform,no_odd=False,centering=True,mutual_center=True)

    F.total_coefficients()
    F.prep_tables()

    return F,patt,rotfreq



def make_rotation(simulation_directory,simulation_name,intime):
    '''
    make_rotation


    '''

    # this restores to the orientation of the saving potential! could be dangerous.
    F = restore_field(simulation_directory+'/potential.'+str(intime)+'.dat')

    F.EOF.eof_file = simulation_directory+'/.eof.cache.file'
    F.SL.model_file = simulation_directory+'/SLGridSph.model'
    F.SL.sph_file = simulation_directory+'/SLGridSph.cache.'+simulation_name
    
    F.set_field_parameters(no_odd=True,halo_l=-1,halo_n=-1,disk_m=-1,disk_n=-1)
    F.make_force_grid()
    F.rotation_curve()

    
    return F


