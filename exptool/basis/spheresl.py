###########################################################################################
#
#  spheresl.py
#     Compute cofficients and forces in a spherical basis. Part of exptool.
#
#     Designed to be a counterpart to eof.py (for cylindrical basis coefficient and force calculation), slowly rounding into form.
#
# 08-30-16: print_progress and verbosity keys added
# 12-03-16: major revisions
# 06-14-18: fixed evaluation bugs, added documentation, checked Python3 compatibility
#
#
'''  _______..______    __    __   _______ .______       _______     _______. __      
    /       ||   _  \  |  |  |  | |   ____||   _  \     |   ____|   /       ||  |     
   |   (----`|  |_)  | |  |__|  | |  |__   |  |_)  |    |  |__     |   (----`|  |     
    \   \    |   ___/  |   __   | |   __|  |      /     |   __|     \   \    |  |     
.----)   |   |  |      |  |  |  | |  |____ |  |\  \----.|  |____.----)   |   |  `----.
|_______/    | _|      |__|  |__| |_______|| _| `._____||_______|_______/    |_______|
spheresl (part of exptool.basis)
    Implementation of Martin Weinberg's SphereSL routines for EXP simulation analysis

see:
Basis.H/.cc
SphereSL.H/.cc
SphericalBasis.H/.cc
Sphere.H/.cc


'''

# python2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

# general python imports
import numpy as np
import time
import sys
import os


# exptool imports
from exptool.utils import utils
from exptool.utils import halo_methods
from exptool.io import psp_io



# (try to) pull in C routines
try:
    from exptool.basis._accumulate_c import r_to_xi,xi_to_r,d_xi_to_r
except:
    from exptool.basis.compatibility import r_to_xi,xi_to_r,d_xi_to_r


# special math imports
from scipy.special import gammaln

# multiprocessing imports
import multiprocessing
import itertools
from multiprocessing import Pool, freeze_support

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

###############################################################################
'''

Table manipulation block

Collection of methods to interface with the tabulated potential-density pairs

get_halo_dens_pot_force
get_halo_dens
get_halo_force
get_halo_force_pot
get_halo_pot_matrix
get_halo_pot

'''
###############################################################################


class SPHTable(object):
    '''
    class handling Spherical Model Grid tables and associated cache files


    '''
    def __init__(self):

        # sizes
        self.lmax = None
        self.nmax = None

        # descriptors
        self.cmap = None
        self.scale = None
        
        # harmonic functions
        self.evtable = None
        self.eftable = None

        # spherical table
        self.xi = None
        self.d0 = None
        


def get_halo_dens_pot_force(x, lmax, nmax, evtable, eftable, xi, d0, p0, cmap, scale):
    #
    # needs the potential table to be defined
    #
    numr = d0.shape[0]
    dens_mat = np.zeros([lmax+1,nmax])
    pot_mat = np.zeros([lmax+1,nmax])
    force_mat = np.zeros([lmax+1,nmax])

    x = r_to_xi(x,cmap,scale);
    
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    
    dxi = xi[1]-xi[0]
    
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
    
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    
    fac = d_xi_to_r(x,cmap,scale)/dxi;
    
    for l in range(0,lmax+1):

        for n in range(0,nmax):
        
            dens_mat[l][n] = (x1*eftable[l][n][indx] + x2*eftable[l,n,indx+1])*np.sqrt(evtable[l,n]) * (x1*d0[indx] + x2*d0[indx+1]);

            if indx == 0:
                # do a forced advance of the indx by one if running into the edge
                # 01-05-16: fixes a bug where the center of the determination fell apart
                force_mat[l][n] = fac * ((x2 - 0.5)*eftable[l,n,0]*p0[0] - 2.0*x2*eftable[l,n,1]*p0[1] + (x2 + 0.5)*eftable[l,n,2]*p0[2]) / np.sqrt(evtable[l][n]);
            else:
                force_mat[l][n] = fac * ((x2 - 0.5)*eftable[l,n,indx-1]*p0[indx-1] - 2.0*x2*eftable[l,n,indx]*p0[indx] + (x2 + 0.5)*eftable[l,n,indx+1]*p0[indx+1]) / np.sqrt(evtable[l][n]);

            pot_mat[l][n] = (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])/\
                          np.sqrt(evtable[l,n]) * (x1*p0[indx] + x2*p0[indx+1]);

    return dens_mat,force_mat,pot_mat



def get_halo_dens(x, lmax, nmax, evtable, eftable, xi, d0, cmap, scale):#, int which):
    #
    # needs the potential table to be defined
    #
    numr = len(d0)
    mat = np.zeros([lmax+1,nmax])
    #if (which || !cmap):
    x = r_to_xi(x,cmap,scale);
    #
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    dxi = xi[1]-xi[0]
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    for l in range(0,lmax+1): #(int l=0; l<=lmax; l++) {
        #print x1,x2
        for n in range(0,nmax): #(int n=1; n<=nmax; n++) {
            mat[l][n] = (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])*np.sqrt(evtable[l,n]) * (x1*d0[indx] + x2*d0[indx+1]);
    return mat



    
def get_halo_force(x, lmax, nmax, evtable, eftable, xi, p0, cmap=0, scale=1.0):
    '''
    get_halo_force

    inputs
    --------

    returns
    --------
    mat     :   (lmax+1,nmax) matrix of halo forces

    '''
    numr = p0.shape[0]
    mat = np.zeros([lmax+1,nmax])
    
    x = r_to_xi(x,cmap,scale);
    
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
            
    dxi = xi[1]-xi[0]
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    
    if (indx < 1): indx = 1;
    if (indx > (numr-2) ): indx = numr - 2;
        
    p = (x - xi[indx])/dxi;
        
    fac = d_xi_to_r(x,cmap,scale)/dxi;
    
    for l in range(0,lmax+1):
        
        for n in range(0,nmax):
            
            mat[l][n] = fac * ((p - 0.5)*eftable[l,n,indx-1]*p0[indx-1] - 2.0*p*eftable[l,n,indx]*p0[indx] + (p + 0.5)*eftable[l,n,indx+1]*p0[indx+1]) / np.sqrt(evtable[l,n]);

    return mat



def get_halo_force_pot(x_in, lmax, nmax, evtable, eftable, xi, p0, cmap, scale):#, int which):
    #
    # needs the potential table to be defined
    #

    # truncate the table if not doing a full tabulation
    '''
    lmax_check = evtable.shape[0] - 1
    nmax_check = evtable.shape[1] - 1

    if lmax < lmax_check:
      evtable = evtable[0:lmax+1,:]
      eftable = eftable[0:lmax+1,:,:]

    if nmax < nmax_check:
      evtable = evtable[:,0:nmax+1]
      eftable = eftable[:,0:nmax+1]      
    '''

    numr = p0.shape[0]
    
    x = r_to_xi(x_in,cmap,scale);
    
    #print x
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
            
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    
    dxi = xi[1]-xi[0]

    fac = d_xi_to_r(x,cmap,scale)/dxi;

    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #print indx
    
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
        
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    
    pot_mat = np.zeros([lmax+1,nmax])
    force_mat = np.zeros([lmax+1,nmax])

    # trapezoidal rule solve for the potential
    pot_mat = (x1*eftable[:,:,indx] + x2*eftable[:,:,indx+1])/np.sqrt(evtable[:,:]) * (x1*p0[indx] + x2*p0[indx+1]);

    if indx == 0:
        # do a forced advance of the indx by one if running into the edge
        force_mat = fac * ((x2 - 0.5)*eftable[:,:,0]*p0[0] - 2.0*x2*eftable[:,:,1]*p0[1] + (x2 + 0.5)*eftable[:,:,2]*p0[2]) / np.sqrt(evtable);

    else:
        force_mat = fac * ((x2 - 0.5)*eftable[:,:,indx-1]*p0[indx-1] - 2.0*x2*eftable[:,:,indx]*p0[indx] + (x2 + 0.5)*eftable[:,:,indx+1]*p0[indx+1]) / np.sqrt(evtable)

    
    return pot_mat,force_mat



def get_halo_pot_matrix(x_in, lmax, nmax, evtable, eftable, xi, p0, cmap, scale):
    #
    # needs the potential table to be defined
    #

    numr = p0.shape[0]
    
    x = r_to_xi(x_in,cmap,scale);
    
    #print x
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
            
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    
    dxi = xi[1]-xi[0]

    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #print indx
    
    if (indx < 0):      indx = 0;
    if (indx > numr-2): indx = numr - 2;
        
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x -   xi[indx])/(dxi);
    
    mat = np.zeros([lmax+1,nmax])

    mat = (x1*eftable[:,:,indx] + x2*eftable[:,:,indx+1])/np.sqrt(evtable) * (x1*p0[indx] + x2*p0[indx+1]);
    

    return mat




def get_halo_pot(x, l, n, evtable, eftable, xi, p0, cmap, scale):#, int which):
    #
    # needs the potential table to be defined
    #
    #if (which || !cmap):
    numr = len(p0)
    x = r_to_xi(x,cmap,scale);
    #print x
    if (cmap==1):
        if (x<-1.0): x=-1.0;
        if (x>=1.0): x=1.0-1.0e-08;
    if (cmap==2):
        if (x<xmin): x=xmin;
        if (x>xmax): x=xmax;
    dxi = xi[1]-xi[0]
    #print x-np.min(xi), (x-np.min(xi))/(dxi)
    indx = int(np.floor( (x-np.min(xi))/(dxi) ))
    #print indx
    if (indx<0): indx = 0;
    if (indx>numr-2): indx = numr - 2;
    x1 = (xi[indx+1] - x)/(dxi);
    x2 = (x - xi[indx])/(dxi);
    
    return (x1*eftable[l,n,indx] + x2*eftable[l,n,indx+1])/np.sqrt(evtable[l,n]) * (x1*p0[indx] + x2*p0[indx+1]);



###############################################################################
'''

Multiprocessing block

Collection of methods to wrap for multiprocessing

Culminates in

compute_coefficients
eval_particles

'''
###############################################################################


def redistribute_particles(ParticleInstance,divisions):
    npart = np.zeros(divisions,dtype=object)
    holders = [psp_io.particle_holder() for x in range(0,divisions)]
    average_part = int(np.floor(len(ParticleInstance.xpos)/divisions))
    first_partition = len(ParticleInstance.xpos) - average_part*(divisions-1)
    low_particle = 0
    for i in range(0,divisions):
        end_particle = low_particle+average_part
        if i==0: end_particle = low_particle+first_partition
        #print low_particle,end_particle
        holders[i].xpos = ParticleInstance.xpos[low_particle:end_particle]
        holders[i].ypos = ParticleInstance.ypos[low_particle:end_particle]
        holders[i].zpos = ParticleInstance.zpos[low_particle:end_particle]
        holders[i].mass = ParticleInstance.mass[low_particle:end_particle]
        low_particle = end_particle
    return holders



def compute_coefficients_solitary_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return compute_coefficients_solitary(*a_b)

def multi_compute_coefficients(holding,nprocs,sph_file,mod_file,verbose=1,no_odd=False):
    pool = Pool(nprocs)
    a_args = [holding[i] for i in range(0,nprocs)]
    second_arg = sph_file
    third_arg = mod_file
    fourth_arg = [0 for i in range(0,nprocs)]
    fourth_arg[0] = verbose
    fifth_arg = no_odd

    # hack our way to python 2/3 compatibility
    try:
        a_coeffs = pool.map(compute_coefficients_solitary_star, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),\
                                                                           fourth_arg,itertools.repeat(fifth_arg)))
    except:
        a_coeffs = pool.map(compute_coefficients_solitary_star, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg),\
                                                                           fourth_arg,itertools.repeat(fifth_arg)))
                                                                           
                                                                           
    pool.close()
    pool.join()
    return a_coeffs



# NOTE:
# if python2/python3 compatibility is desired here, need to have itertools.izip AND zip versions
# see http://www.diveintopython3.net/porting-code-to-python-3-with-2to3.html

def compute_coefficients(PSPInput,sph_file,mod_file,verbose=1,no_odd=False):

    SL_Out = SL_Object()
    SL_Out.time = PSPInput.time
    SL_Out.dump = PSPInput.infile
    SL_Out.comp = PSPInput.comp
    SL_Out.nbodies = PSPInput.mass.size # in case we aren't using the full total; how many were input?
    SL_Out.sph_file = sph_file
    SL_Out.model_file = mod_file

    # get information for SL_Out
    lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
    SL_Out.lmax = lmax
    SL_Out.nmax = nmax
    
    nprocs = multiprocessing.cpu_count()

    holding = redistribute_particles(PSPInput,nprocs)

    if (verbose):
            print('sl.compute_coefficients: {0:d} processors, {1:d} particles each.'.format(nprocs,len(holding[0].mass)))
    
    t1 = time.time()
    freeze_support()
    
    a_coeffs = multi_compute_coefficients(holding,nprocs,sph_file,mod_file,verbose=verbose,no_odd=no_odd)
    
    if (verbose > 0): print('spheresl.compute_coefficients: accumulation took {0:3.2f} seconds, or {1:4.2f} microseconds per orbit.'.format(time.time()-t1, 1.e6*(time.time()-t1)/len(PSPInput.mass)))

    # sum over processes
    summed_coefs = np.sum(np.array(a_coeffs,dtype=object),axis=0)

    SL_Out.expcoef = summed_coefs
    
    return SL_Out




def all_eval_particles_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return all_eval_particles(*a_b)

def multi_all_eval_particles(holding,nprocs,expcoef,sph_file,mod_file,verbose):
    pool = Pool(nprocs)
    a_args = [holding[i] for i in range(0,nprocs)]
    second_arg = expcoef
    third_arg = sph_file
    fourth_arg = mod_file
    fifth_arg = [0 for i in range(0,nprocs)]
    fifth_arg[0] = verbose
    
    a_vals = pool.map(all_eval_particles_star, zip(a_args, itertools.repeat(second_arg),itertools.repeat(third_arg)\
                                                              ,itertools.repeat(fourth_arg),fifth_arg))
                                                              
    pool.close()
    pool.join()
    return a_vals



def eval_particles(ParticleInstance,expcoef,sph_file,mod_file,nprocs=-1,verbose=1,l1=0,l2=1000):
    if nprocs == -1:
        nprocs = multiprocessing.cpu_count()

    if nprocs > 1:
        holding = redistribute_particles(ParticleInstance,nprocs)
        t1 = time.time()
        freeze_support()
        #
        if (verbose):
            print('sl.eval_particles: {0:d} processors, {1:d} particles each.'.format(nprocs,len(holding[0].mass)))

        # this doesn't handle l1/l2
        a_vals = multi_all_eval_particles(holding,nprocs,expcoef,sph_file,mod_file,verbose)
        #
        
        if (verbose):
            print('spheresl.eval_particles: particle Evaluation took {0:3.2f} seconds, or {1:4.2f} microseconds per orbit.'.format(time.time()-t1, 1.e6*(time.time()-t1)/len(ParticleInstance.mass)))
        # sum over processes
        den0,den1,pot0,pot1,potr,pott,potp,rr = mix_outputs_sph(np.array(a_vals))
        

    else:
        den0,den1,pot0,pot1,potr,pott,potp,rr = all_eval_particles(ParticleInstance, expcoef, sph_file, mod_file,verbose,L1=l1,L2=l2)

        
    return den0,den1,pot0,pot1,potr,pott,potp,rr



def mix_outputs_sph(MultiOutput):
    n_instances = len(MultiOutput)
    n_part = 0
    for i in range(0,n_instances):
        n_part += len(MultiOutput[i][0])
    full_den0 = np.zeros(n_part)
    full_den1 = np.zeros(n_part)
    full_pot0 = np.zeros(n_part)
    full_pot1 = np.zeros(n_part)
    full_potr = np.zeros(n_part)
    full_pott = np.zeros(n_part)
    full_potp = np.zeros(n_part)
    full_rr = np.zeros(n_part)
    #
    #
    first_part = 0
    for i in range(0,n_instances):
        n_instance_part = len(MultiOutput[i][0])
        full_den0[first_part:first_part+n_instance_part] = MultiOutput[i][0]
        full_den1[first_part:first_part+n_instance_part] = MultiOutput[i][1]
        full_pot0[first_part:first_part+n_instance_part] = MultiOutput[i][2]
        full_pot1[first_part:first_part+n_instance_part] = MultiOutput[i][3]
        full_potr[first_part:first_part+n_instance_part] = MultiOutput[i][4]
        full_pott[first_part:first_part+n_instance_part] = MultiOutput[i][5]
        full_potp[first_part:first_part+n_instance_part] = MultiOutput[i][6]
        full_rr[first_part:first_part+n_instance_part] = MultiOutput[i][7]
        first_part += n_instance_part
    return full_den0,full_den1,full_pot0,full_pot1,full_potr,full_pott,full_potp,full_rr







def compute_coefficients_solitary(ParticleInstance,sph_file,model_file,verbose=0,no_odd=False):
    '''
    
     follows compute_coefficients from EXP's SphereSL.cc (see also SphericalBasis.cc)
    
        this is the workhorse that does the accumulation.
     

    inputs
    -----------------



    returns
    ----------------
    expcoef               : (matrix, (lmax)*(lmax+2)+1 x nmax) coefficient matrix
    
    '''
    
    fac0 = -4.0*np.pi;
    lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
    xi,r,p0,d0 = halo_methods.init_table(model_file,numr,rmin,rmax,cmap,scale)

    
    expcoef = np.zeros([lmax*(lmax+2)+1,nmax])

    
    factorial = factorial_return(lmax)
    
    norb = len(ParticleInstance.mass)

    for p in range(0,norb): 

        if (verbose > 0) & ( ((float(p)+1.) % 1000. == 0.0) | (p==0)): utils.print_progress(p,norb,'spheresl.compute_coefficients')

        xx = ParticleInstance.xpos[p]
        yy = ParticleInstance.ypos[p]
        zz = ParticleInstance.zpos[p]
        mass = ParticleInstance.mass[p];

        #
        # compute spherical coordinates
        #
        r2 = xx*xx + yy*yy + zz*zz
        r = np.nanmax([np.sqrt(r2),1.0e-10]);
        costh = zz/r;
        phi = np.arctan2(yy,xx);
        
        #
        # now compute the legendre polynomial value tables
        #
        legs = legendre_R(lmax, costh)

        #
        # go through loops
        #
        potd = get_halo_pot_matrix(r, lmax, nmax, evtable, eftable, xi, p0, cmap, scale)
           # returns potd, (lmax+1,nmax)
        
        loffset = 0
        for l in range(0,lmax+1):

            # skip odd azimuthal terms
            if ( (l % 2) != 0) & (no_odd):
                loffset += (2*l+1)  # advance expcoef position anyway
                continue
            
            moffset = 0
            for m in range(0,l+1):

                fac = factorial[l][m] * legs[l][m]

                fac4 = potd[l]*fac*fac0; 
                
                if (m==0):
                     
                    expcoef[loffset+moffset] += fac4 * mass
                    
                    moffset += 1  # advance expcoef position
                    
                else:
                  
                    expcoef[loffset+moffset  ] += np.cos( phi * m ) * fac4 * mass
                    expcoef[loffset+moffset+1] += np.sin( phi * m ) * fac4 * mass
                    
                    moffset += 2  # advance expcoef position
                    
            loffset += (2*l+1)  # advance expcoef position
            
    return expcoef







def legendre_R(lmax, x):
    '''
    Compute Associated Legendre Polynomials

    return an (lmax+1,lmax+1) element array of the legendre polynomials for the l and m spherical harmonic orders.

    see equivalent function in Basis.cc

    '''
    p = np.zeros([lmax+1,lmax+1])
    
    p[0][0] = 1.0
    pll = 1.0
    
    if (lmax > 0):
        somx2 = np.sqrt( (1.0 - x)*(1.0 + x) );
        fact = 1.0;
        for m in range(1,lmax+1):
            pll *= -fact*somx2;
            p[m][m] = pll;
            fact += 2.0;
    
    
    for m in range(0,lmax):#(m=0; m<lmax; m++) {
        pl2 = p[m][m];
        p[m+1][m] = x*(2*m+1)*pl2
        pl1 = x*(2*m+1)*pl2
        
        for l in range(m+2,lmax+1):
            p[l][m] = (x*(2*l-1)*pl1-(l+m-1)*pl2)/(l-m)
            pll = (x*(2*l-1)*pl1-(l+m-1)*pl2)/(l-m)
            pl2 = pl1
            pl1 = pll
    
    
    return p





def dlegendre_R(lmax, x):
    '''
    Compute Associated Legendre Polynomials and derivitives

    return an (lmax+1,lmax+1) element array of the legendre polynomials for the l and m spherical harmonic orders.

    AND

    return an (lmax+1,lmax+1) element of the legendre derivatives.

    (for use in computing forces)

    see comparable call in Basis.cc

    '''
    p = np.zeros([lmax+1,lmax+1])
    dp = np.zeros([lmax+1,lmax+1])
    
    p[0][0] = 1.0
    pll = 1.0
    
    if (lmax > 0):
        somx2 = np.sqrt( (1.0 - x)*(1.0 + x) );
        fact = 1.0;
        for m in range(1,lmax+1):#(m=1; m<=lmax; m++) {
            pll *= -fact*somx2;
            p[m][m] = pll;
            fact += 2.0;
    
    
    for m in range(0,lmax):#(m=0; m<lmax; m++) {
        pl2 = p[m][m];
        p[m+1][m] = x*(2*m+1)*pl2
        pl1 = x*(2*m+1)*pl2
        
        for l in range(m+2,lmax+1):
            p[l][m] = (x*(2*l-1)*pl1-(l+m-1)*pl2)/(l-m)
            pll = (x*(2*l-1)*pl1-(l+m-1)*pl2)/(l-m)
            pl2 = pl1
            pl1 = pll
    
    
    MINEPS=1.e-8
    if (1.0-np.abs(x) < MINEPS):
        if (x>0): x =   1.0 - MINEPS;
        else    : x = -(1.0 - MINEPS);
    #
    #
    somx2 = 1.0/(x*x - 1.0);
    dp[0][0] = 0.0;
    for l in range(1,lmax+1):# (l=1; l<=lmax; l++):
        for m in range(0,l):#(m=0; m<l; m++):
            #print l
            dp[l][m] = somx2*(x*l*p[l][m] - (l+m)*p[l-1][m]);
            dp[l][l] = somx2*x*l*p[l][l];
            
    return p,dp


def sinecosine_R(mmax, phi):
    '''
    Compute vectors of sines and cosines by recursion


    '''

    c = np.zeros([mmax+1])
    s = np.zeros([mmax+1])
    
    c[0] = 1.0;
    s[0] = 0.0;

    c[1] = cos(phi);
    s[1] = sin(phi);

    for m in range(2,mmax+1):#m=2; m<=mmax; m++) {
        c[m] = 2.0*c[1]*c[m-1] - c[m-2];
        s[m] = 2.0*c[1]*s[m-1] - s[m-2];

    return c,s



def factorial_return_new(lmax):
    '''
    factorial from SphericalBasis.cc


    '''
    factorial = np.zeros([lmax+1,lmax+1])

    for l in range(0,lmax+1):
        for m in range(0,l+1):
            factorial[l][m] = np.math.factorial(l-m)/np.math.factoria(l+m);

    return factorial





def factorial_return(lmax):
    '''
    return an (lmax+1,lmax+1) element array of factorial terms for spherical harmonics.

    see functional form of spherical harmonics.

    '''
    factorial = np.zeros([lmax+1,lmax+1])
    for l in range(0,lmax+1):
        
        for m in range(0,l+1):
            
            factorial[l][m] = np.sqrt( (0.5*l+0.25)/np.pi * np.exp(gammaln(1.0+l-m) - gammaln(1.0+l+m)) );
            
            if (m != 0):
                factorial[l][m] *= np.sqrt(2.)

    return factorial



#
#
# these definitions may be useful to get the matrix formulation of spheresl running (01.17.17)
def get_pot_coefs(l, l_coef, l_potd, l_dpot):
    nmax = len(l_coef)
    pp = 0.0
    dpp = 0.0;
    for n in range(0,nmax):
        
        pp  += potd[i] * coef[i];
        dpp += dpot[i] * coef[i];
        
    return -1.*pp, -1.*dpp


def get_dens_coefs(l, l_coef, dend):
    nmax = len(l_coef)
    # pass this the l array of coefficients
    # and d array of coefficients
    pp = 0.0;
    for n in range(0,nmax):
        
        pp  += dend[n] * l_coef[n];
        
    return pp




def all_eval_table(r, costh, phi, expcoef, sph_file, mod_file,L1=0,L2=-1):
  '''
  all_eval_table
     version of all_eval that reads in cached tables (slower, but standalone)



  '''
  lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
  xi,rarr,p0,d0 = halo_methods.init_table(mod_file,numr,rmin,rmax,cmap,scale)
  if (L2 == -1): L2 = lmax+1
  # compute factorial array
  factorial = factorial_return(lmax)
  #
  # begin function
  sinth = -np.sqrt(np.abs(1.0 - costh*costh));
  fac1 = factorial[0][0];
  #
  # use the basis to get the density,potential,force arrays (which have already been read in somewhere?)
  dend = get_halo_dens(r, lmax, nmax, evtable, eftable, xi, d0, cmap, scale)
  potd = get_halo_pot_matrix(r, lmax, nmax, evtable, eftable, xi, p0, cmap, scale)
  dpot = get_halo_force(r, lmax, nmax, evtable, eftable, xi, p0, cmap, scale)
  #
  #
  legs,dlegs = dlegendre_R(lmax,costh)
  #
  den0 = np.sum(fac1 * expcoef[0]*dend[0]);
  pot0 = np.sum(fac1 * expcoef[0]*potd[0]);
  potr = np.sum(fac1 * expcoef[0]*dpot[0]);
  den1 = 0.0;
  pot1 = 0.0;
  pott = 0.0;
  potp = 0.0;
  #
  # L loop
  #
  loffset = 1
  for l in range(1,lmax+1):
    
    if (l>(L2+1)) | (l<(L1+1)):
      loffset+=(2*l+1)
      continue
    #
    # M loop
    moffset = 0
    for m in range(0,l+1):#(int m=0, moffset=0; m<=l; m++) {
      #print l,m,loffset,moffset
      fac1 = factorial[l][m];
      if (m==0):
            #den1 += fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]);
            #pot1 += fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]);
            #dpot += fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]);
            #pott += fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]);
            den1 += np.sum(fac1*legs[1][m] * (expcoef[loffset+moffset] * dend[l]));
            pot1 += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * potd[l]));
            #dpot += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
            potr += np.sum(fac1*legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
            pott += np.sum(fac1*dlegs[l][m]* (expcoef[loffset+moffset] * potd[l]));
            moffset+=1;
      else:
            cosm = np.cos(phi*m);
            sinm = np.sin(phi*m);
            #den1 += fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm );
            #pot1 += fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm );
            #dpot += fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm );
            #pott += fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm );
            #potp += fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm );
            den1 += np.sum(fac1*legs[l][m]*( expcoef[loffset+moffset]   * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm ));
            pot1 += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm ));
            #dpot += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm ));
            potr += np.sum(fac1*legs[l][m]* ( expcoef[loffset+moffset]   * dpot[l]*cosm +    expcoef[loffset+moffset+1] * dpot[l]*sinm ));
            pott += np.sum(fac1*dlegs[l][m]* ( expcoef[loffset+moffset]   * potd[l]*cosm +   expcoef[loffset+moffset+1] * potd[l]*sinm ));
            potp += np.sum(fac1*legs[l][m] * m * (-expcoef[loffset+moffset]   * potd[l]*sinm +   expcoef[loffset+moffset+1] * potd[l]*cosm ));
            moffset +=2;
    loffset+=(2*l+1)
  #
  #
  #
  #densfac = 1.0/(scale*scale*scale) * 0.25/np.pi;
  #potlfac = 1.0/scale;
  #den0  *= densfac;
  #den1  *= densfac;
  #pot0  *= potlfac;
  #pot1  *= potlfac;
  #potr  *= potlfac/scale;
  #pott  *= potlfac*sinth;
  #potp  *= potlfac;
  return den0,den1,pot0,pot1,potr,pott,potp



def all_eval(r, costh, phi, expcoef,\
             xi,p0,d0,cmap,scale,\
             lmax,nmax,\
             evtable,eftable,\
             no_odd=False,verbose=0):
    '''
    all_eval: simple workhorse to evaluate the spherical basis
        this version loads everything up first: lots of inputs, but no need to reread anything.
        pros, fast; cons, needs a number of inputs

    inputs
    -------
    r is 3-dimensional
    
    

    '''

    # check to see if the lmax, nmax, and evtable, eftable, expcoef agree
    #   this allows for truncation of lmax,nmax in a simple way
    # truncate the table if not doing a full tabulation
    lmax_check = evtable.shape[0] - 1
    nmax_check = evtable.shape[1] - 1

    if lmax < lmax_check:
      if verbose >=4: print('spheresl.all_eval: reducing lmax.')
      evtable = evtable[0:lmax+1,:]
      eftable = eftable[0:lmax+1,:,:]
      expcoef = expcoef[0:lmax+1,:]

    if nmax < nmax_check:
      if verbose >=4: print('spheresl.all_eval: reducing nmax.')
      evtable = evtable[:,0:nmax]
      eftable = eftable[:,0:nmax]
      expcoef = expcoef[:,0,nmax]   



    # compute factorial array
    factorial = factorial_return(lmax)
    fac1 = factorial[0][0];

    # begin function
    #sinth = -np.sqrt(np.abs(1.0 - costh*costh));
    
    #
    # use the basis to get the density,potential,force arrays
    #
    # these three need to be stuffed together into one call to save loops
    dend,dpot,potd = get_halo_dens_pot_force(r, lmax, nmax, evtable, eftable, xi, d0, p0, cmap, scale)
    
    #
    legs,dlegs = dlegendre_R(lmax,costh)
    #
    den0 = np.sum(fac1 * expcoef[0]*dend[0]);
    pot0 = np.sum(fac1 * expcoef[0]*potd[0]);
    potr = np.sum(fac1 * expcoef[0]*dpot[0]);

    # converted den1 to be the total density (not just perturbing density, NOTE CHANGE!) (01.20.2017)
    den1 = np.sum(fac1 * expcoef[0]*dend[0]);
    
    pot1 = 0.0;
    pott = 0.0;
    potp = 0.0;
    #
    # L loop
    #
    loffset = 1
    for l in range(1,lmax+1):

      # skip odd terms if desired
      if ( (l % 2) != 0) & (no_odd):
                loffset+=(2*l+1)
                continue
        
      # M loop
      moffset = 0
      
      for m in range(0,l+1):
        
        fac1 = factorial[l][m];
        
        if (m==0):
              den1 += np.sum(fac1* legs[l][m] * (expcoef[loffset+moffset] * dend[l]));
              pot1 += np.sum(fac1* legs[l][m] * (expcoef[loffset+moffset] * potd[l]));
              potr += np.sum(fac1* legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
              pott += np.sum(fac1*dlegs[l][m] * (expcoef[loffset+moffset] * potd[l]));
              moffset+=1;
              
        else:
              cosm = np.cos(phi*m);
              sinm = np.sin(phi*m);
              den1 += np.sum(fac1* legs[l][m] *     ( expcoef[loffset+moffset] * dend[l]*cosm + expcoef[loffset+moffset+1] * dend[l]*sinm ));
              pot1 += np.sum(fac1* legs[l][m] *     ( expcoef[loffset+moffset] * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm ));
              potr += np.sum(fac1* legs[l][m] *     ( expcoef[loffset+moffset] * dpot[l]*cosm + expcoef[loffset+moffset+1] * dpot[l]*sinm ));
              pott += np.sum(fac1*dlegs[l][m] *     ( expcoef[loffset+moffset] * potd[l]*cosm + expcoef[loffset+moffset+1] * potd[l]*sinm ));
              potp += np.sum(fac1* legs[l][m] * m * (-expcoef[loffset+moffset] * potd[l]*sinm + expcoef[loffset+moffset+1] * potd[l]*cosm ));
              moffset +=2;
              
      loffset+=(2*l+1)
    #
    #
    #
    
    #densfac = (1.0/(scale*scale*scale)) * (0.25/np.pi);
    densfac = 0.25/np.pi
    #potlfac = 1.0/scale;
    den0  *= densfac;
    den1  *= densfac;
    #pot0  *= potlfac;
    #pot1  *= potlfac;
    #potr  *= potlfac/scale;
    #pott  *= potlfac*sinth;
    #potp  *= potlfac;
    
    return den0,den1,pot0,pot1,potr,pott,potp




def force_eval(r, costh, phi, expcoef,\
             xi,p0,d0,cmap,scale,\
             lmax,nmax,\
             evtable,eftable,\
             no_odd=False,verbose=0):
    '''
    force_eval: simple workhorse to evaluate the spherical basis forces

    inputs
    -------
    r is 3-dimensional
    

    outputs
    -------
    potr    :   radial force
    pott    :   theta force
    potp    :   phi force
    pot     :   total potential
    pot0    :   monopole potential

    '''
    # check to see if the lmax, nmax, and evtable, eftable, expcoef agree
    #   this allows for truncation of lmax,nmax in a simple way
    # truncate the table if not doing a full tabulation
    lmax_check = evtable.shape[0] - 1
    nmax_check = evtable.shape[1] - 1

    if lmax < lmax_check:
      if verbose >=4: print('spheresl.force_eval: reducing lmax.')
      evtable = evtable[0:lmax+1,:]
      eftable = eftable[0:lmax+1,:,:]
      expcoef = expcoef[0:(lmax+1)*(lmax+1),:]

    if nmax < nmax_check:
      if verbose >=4: print('spheresl.force_eval: reducing nmax.')
      evtable = evtable[:,0:nmax]
      eftable = eftable[:,0:nmax]
      expcoef = expcoef[:,0:nmax]   



    # compute factorial array
    factorial = factorial_return(lmax)
    fac1 = factorial[0][0];

    # begin function
    #sinth = -np.sqrt(np.abs(1.0 - costh*costh));
    
    #
    # use the basis to get the density,potential,force arrays
    #
    # these three need to be stuffed together into one call to save loops
    dend,dpot,potd = get_halo_dens_pot_force(r, lmax, nmax, evtable, eftable, xi, d0, p0, cmap, scale)
    
    #
    legs,dlegs = dlegendre_R(lmax,costh)
    #
    pot0 = np.sum(fac1 * expcoef[0]*potd[0]);
    potr = np.sum(fac1 * expcoef[0]*dpot[0]);


    # build matrix for phi terms
    morder = np.tile(np.arange(0.,lmax+1,1.),(nmax,1)).T

    try:

        # verify length is that of MMAX
        if len(phi) != lmax+1:
            print('spheresl.force_eval: varying phi detected, with mismatched lengths. breaking...')
            

        else:
            phiarr = np.tile(phi,(nmax,1)).T
            
            cosm = np.cos(phiarr*morder)
            sinm = np.sin(phiarr*morder)
            
    except:
        cosm = np.cos(phi*morder)
        sinm = np.sin(phi*morder)

    
    pot1 = 0.0;
    pott = 0.0;
    potp = 0.0;
    #
    # L loop
    #
    loffset = 1
    for l in range(1,lmax+1):

      # skip odd terms if desired
      if ( (l % 2) != 0) & (no_odd):
                loffset+=(2*l+1)
                continue
        
      # M loop
      moffset = 0
      
      for m in range(0,l+1):
        
        fac1 = factorial[l][m];
        
        if (m==0):
              pot1 += np.sum(fac1* legs[l][m] * (expcoef[loffset+moffset] * potd[l]));
              potr += np.sum(fac1* legs[l][m] * (expcoef[loffset+moffset] * dpot[l]));
              pott += np.sum(fac1*dlegs[l][m] * (expcoef[loffset+moffset] * potd[l]));
              moffset+=1;
              
        else:

              pot1 += np.sum(fac1* legs[l][m] *     ( expcoef[loffset+moffset] * potd[l]*cosm[l] + expcoef[loffset+moffset+1] * potd[l]*sinm[l] ));
              potr += np.sum(fac1* legs[l][m] *     ( expcoef[loffset+moffset] * dpot[l]*cosm[l] + expcoef[loffset+moffset+1] * dpot[l]*sinm[l] ));
              pott += np.sum(fac1*dlegs[l][m] *     ( expcoef[loffset+moffset] * potd[l]*cosm[l] + expcoef[loffset+moffset+1] * potd[l]*sinm[l] ));
              potp += np.sum(fac1* legs[l][m] * m * (-expcoef[loffset+moffset] * potd[l]*sinm[l] + expcoef[loffset+moffset+1] * potd[l]*cosm[l] ));
              moffset +=2;
              
      loffset+=(2*l+1)

    
    #return den0,den1,pot0,pot1,potr,pott,potp


    return potr,pott,potp,pot1,pot0





def all_eval_particles(Particles, expcoef, sph_file, mod_file,verbose,L1=-1000,L2=1000,NO_ODD=False):
  '''
  check against determine_fields_at_point_sph (also _cyl) in SphericalBasis.cc

  '''

  # parse model files
  lmax,nmax,numr,cmap,rmin,rmax,scale,ltable,evtable,eftable = halo_methods.read_cached_table(sph_file)
  xi,rarr,p0,d0 = halo_methods.init_table(mod_file,numr,rmin,rmax,cmap,scale)
  
  # compute factorial array
  factorial = factorial_return(lmax)
  #
  
  # begin function
  norb = len(Particles.xpos)
  r = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos + Particles.zpos*Particles.zpos)**0.5
  rr = (Particles.xpos*Particles.xpos + Particles.ypos*Particles.ypos)**0.5
  phi = np.arctan2(Particles.ypos,Particles.xpos)
  #
  # is this over planar or 3d r??
  costh = Particles.zpos/r
  #
  # allocate arrays
  den0 = np.zeros(norb)
  pot0 = np.zeros(norb)
  potr = np.zeros(norb)
  den1 = np.zeros(norb)
  pot1 = np.zeros(norb)
  pott = np.zeros(norb)
  potp = np.zeros(norb)
  #
  for part in range(0,norb):
      if (verbose > 0) & ( ((float(part)+1.) % 1000. == 0.0) | (part==0)): utils.print_progress(part,norb,'spheresl.all_eval_particles')
      sinth = -np.sqrt(np.abs(1.0 - costh[part]*costh[part]));
      fac1 = factorial[0][0];
      #
      # use the basis to get the density,potential,force arrays
      #
      # these three need to be stuffed together into one call to save loops
      dend,dpot,potd = get_halo_dens_pot_force(r[part], lmax, nmax, evtable, eftable, xi, d0, p0, cmap, scale)
      #
      #
      legs,dlegs = dlegendre_R(lmax,costh[part])
      #
      den0[part] = np.sum(fac1 * expcoef[0]*dend[0]);
      pot0[part] = np.sum(fac1 * expcoef[0]*potd[0]);
      potr[part] = np.sum(fac1 * expcoef[0]*dpot[0]);
      #
      # L loop
      #
      loffset = 1
      for l in range(1,lmax+1):

        #fac1 = (2.0*l+1.0)/(4.0*M_PI);

        # if L truncation is set
        if (l > (L2)) | (l < (L1)):
          
          loffset+=(2*l+1)
          
          continue

        # if NO_ODD
        if (NO_ODD) & (l%2 !=0): 
                
                loffset += (2*l+1)
                
                continue

              
        # M loop
        moffset = 0
        
        for m in range(0,l+1):
          
          fac1 = factorial[l][m];
          
          if (m==0):
                den1[part] += np.sum(fac1* legs[1][m] * expcoef[loffset+moffset] * dend[l]);
                pot1[part] += np.sum(fac1* legs[l][m] * expcoef[loffset+moffset] * potd[l]);
                potr[part] += np.sum(fac1* legs[l][m] * expcoef[loffset+moffset] * dpot[l]);
                pott[part] += np.sum(fac1*dlegs[l][m] * expcoef[loffset+moffset] * potd[l]);
                
                moffset+=1;
                
          else:
                cosm = np.cos(phi[part] * m);
                sinm = np.sin(phi[part] * m);
                den1[part] += np.sum(fac1*  legs[l][m] * dend[l] *     ( expcoef[loffset+moffset] * cosm + expcoef[loffset+moffset+1] * sinm ));
                pot1[part] += np.sum(fac1*  legs[l][m] * potd[l] *     ( expcoef[loffset+moffset] * cosm + expcoef[loffset+moffset+1] * sinm ));
                potr[part] += np.sum(fac1*  legs[l][m] * dpot[l] *     ( expcoef[loffset+moffset] * cosm + expcoef[loffset+moffset+1] * sinm ));
                pott[part] += np.sum(fac1* dlegs[l][m] * potd[l] *     ( expcoef[loffset+moffset] * cosm + expcoef[loffset+moffset+1] * sinm ));
                potp[part] += np.sum(fac1*  legs[l][m] * potd[l] * m * (-expcoef[loffset+moffset] * sinm + expcoef[loffset+moffset+1] * cosm ));
                
                moffset +=2;
                
        loffset+=(2*l+1)
  # END particle loop
  #
  #
  #

  #
  # 02-27-17: why are these here? they are wrong. what is the prescale??
  #
  #densfac = 1.0/(scale*scale*scale) * 0.25/np.pi;
  densfac = 0.25*np.pi
  #potlfac = 1.0/scale;
  
  den0  *= densfac;
  den1  *= densfac;
  #pot0  *= potlfac;
  #pot1  *= potlfac;
  #potr  *= potlfac/scale;
  #pott  *= potlfac*sinth;
  #potp  *= potlfac;
  
  return den0,den1,pot0,pot1,potr,pott,potp,rr








########################################################################################
#
# the tools to save sl coefficient files
#

# make an SL object to carry around interesting bits of data
class SL_Object(object):
    time = None
    dump = None
    comp = None
    nbodies = None
    lmax = None
    nmax = None
    sph_file = None
    model_file = None
    expcoef = None # admittedly this is a confusing structure but probably the best we can do



def sl_coefficients_to_file(f,SL_Object):
    #
    # write an individual dump to file
    #

    np.array([SL_Object.time],dtype='f4').tofile(f)
    np.array([SL_Object.dump],dtype='S100').tofile(f)
    np.array([SL_Object.comp],dtype='S8').tofile(f)
    np.array([SL_Object.nbodies],dtype='i4').tofile(f)
    np.array([SL_Object.sph_file],dtype='S100').tofile(f)
    np.array([SL_Object.model_file],dtype='S100').tofile(f)
    # 4+100+8+4+100+100 = 316 bytes to here
    
    np.array([SL_Object.lmax,SL_Object.nmax],dtype='i4').tofile(f)
    # 4x2 = 8 bytes
    
    np.array(SL_Object.expcoef.reshape(-1,),dtype='f8').tofile(f)
    # 8 bytes x ((lmax)*(lmax+2)+1) x (nmax) bytes to end of array
    


# wrap the coefficients to file
def save_sl_coefficients(outfile,SL_Object,verbose=0):

    # check to see if file exists
    try:
        f = open(outfile,'rb+')
        f.close()
        
    except:
        f = open(outfile,'wb')
        np.array([0],dtype='i4').tofile(f)
        f.close()
        
    
    f = open(outfile,'rb+')

    ndumps = np.memmap(outfile,dtype='i4',shape=(1))
    ndumps += 1
    ndumps.flush() # update the lead value

    if verbose: print('spheresl.save_sl_coefficients: coefficient file currently has {0:d} dumps.'.format(ndumps[0]))

    # seek to the correct position
    # SL_Object must have the same size as previous dumps... not checking if that is true (yet)
    f.seek(4 + (ndumps-1)*(8*((SL_Object.lmax)*(SL_Object.lmax+2)+1)*(SL_Object.nmax)+324) )

    sl_coefficients_to_file(f,SL_Object)

    f.close()


def restore_sl_coefficients(infile):

    try:
        f = open(infile,'rb')
    except:
        print('spheresl.restore_sl_coefficients: no infile of that name exists.')

    f.seek(0)
    [ndumps] = np.fromfile(f,dtype='i4',count=1)
    
    f.seek(4)

    SL_Dict = {}


    for step in range(0,ndumps):
        
        SL_Out = extract_sl_coefficients(f)

        SL_Dict[np.round(SL_Out.time,3)] = SL_Out


    f.close()

    return SL_Out,SL_Dict


    
def extract_sl_coefficients(f):
    # operates on an open file
    SL_Obj = SL_Object()


    [SL_Obj.time] = np.fromfile(f,dtype='f4',count=1)
    [SL_Obj.dump] = np.fromfile(f,dtype='S100',count=1)
    [SL_Obj.comp] = np.fromfile(f,dtype='S8',count=1)
    [SL_Obj.nbodies] = np.fromfile(f,dtype='i4',count=1)
    [SL_Obj.sph_file] = np.fromfile(f,dtype='S100',count=1)
    [SL_Obj.model_file] = np.fromfile(f,dtype='S100',count=1)


    [SL_Obj.lmax,SL_Obj.nmax] = np.fromfile(f,dtype='i4',count=2)
    cosine_flat = np.fromfile(f,dtype='f8',count=((SL_Obj.lmax)*(SL_Obj.lmax+2)+1)*(SL_Obj.nmax))

    SL_Obj.expcoef = cosine_flat.reshape([((SL_Obj.lmax)*(SL_Obj.lmax+2)+1),(SL_Obj.nmax)])
    
    return SL_Obj





##########################################################################################
#
# add ability to parse PSP files for setup in accumulation
#
#


def parse_components(simulation_directory,simulation_name,output_number):

    # set up a dictionary to hold the details
    ComponentDetails = {}

    PSP = psp_io.Input(simulation_directory+'OUT.'+simulation_name+'.%05i' %output_number,validate=True)

    for comp_num in range(0,PSP.ncomp):

        # find components that have sphereSL matches
        if PSP.comp_expansions[comp_num] == 'sphereSL':

            # set up a dictionary based on the component name
            ComponentDetails[PSP.comp_titles[comp_num]] = {}

            # set up expansion flag
            ComponentDetails[PSP.comp_titles[comp_num]]['expansion'] = 'sphereSL'

            # population dictionary with desirables
            ComponentDetails[PSP.comp_titles[comp_num]]['nbodies'] = PSP.comp_nbodies[comp_num]

            # break basis string for eof_file
            broken_basis = PSP.comp_basis[comp_num].split(',')
            broken_basis = [v.strip() for v in broken_basis] # rip out spaces
            basis_dict = {}
            for value in broken_basis:  basis_dict[value.split('=')[0]] = value.split('=')[1]

            # ^^^
            # ideally this will be populated with defaults as well so that all values used are known

            try:
                ComponentDetails[PSP.comp_titles[comp_num]]['modelname'] = simulation_directory+basis_dict['modelname']
                
            except:
                print('spheresl.parse_components: Component %s has no Spherical Model file specified (setting None).' %PSP.comp_titles[comp_num])
                ComponentDetails[PSP.comp_titles[comp_num]]['modelname'] = None

            # guess at the cache name from standard Sphere nomenclature. should this be flagged somehow?
            ComponentDetails[PSP.comp_titles[comp_num]]['cachename'] = simulation_directory+'SLGridSph.cache.'+simulation_name

    return ComponentDetails





#
# visualizing routines
#



def make_sl_wake(SLObj,halofac=1.,exclude=False,orders=None,l1=0,l2=1000,xline = np.linspace(-0.03,0.03,75),zaspect=1.,zoffset=0.,coord='Y',axis=False,cyl=False):
    '''
    make_sl_wake: evaluate a simple grid of points along an axis

    inputs
    ---------
    SLObj: 





    '''
    #     now a simple grid
    #
    # this will always be square in resolution--could think how to change this?
    zline = xline*zaspect
    xgrid,ygrid = np.meshgrid(xline,zline)

    if axis:
        zline = np.array([0.])
        xgrid = xline[np.where(xline>=0.)[0]]
        xline = xgrid
        ygrid = np.array([0.])
    
    #
    P = psp_io.particle_holder()
    P.xpos = xgrid.reshape(-1,)

    # set the secondary coordinate
    if coord=='Y':
        P.ypos = ygrid.reshape(-1,)
        P.zpos = np.zeros(xline.shape[0]*zline.shape[0]) + zoffset

    if coord=='Z':
        P.ypos = np.zeros(xline.shape[0]*zline.shape[0]) + zoffset
        P.zpos = ygrid.reshape(-1,)
    
    P.mass = np.zeros(xline.shape[0]*zline.shape[0]) # mass doesn't matter for evaluations, just get field values
    #
    #
    coefs_in = np.copy(SLObj.expcoef)
    #
    if exclude:
        #for i in [1,2,3,9,10,11,12,13,14,15]:
        for i in orders:
            coefs_in[i] = np.zeros(SLObj.nmax)
    #
    den0,den1,pot0,pot1,potr,pott,potp,rr = eval_particles(P,coefs_in*halofac,SLObj.sph_file,SLObj.model_file,l1=l1,l2=l2)
    #
    #

    # do a conversion to cylindrical?
    halo_rforce = ( rr*potr + P.zpos*pott )/( rr*2. + P.zpos**2.)**0.5
    halo_zforce = ( P.zpos*potr - rr*pott )/( rr*2. + P.zpos**2.)**0.5

    
    wake = {}
    wake['X'] = xgrid
    wake['Y'] = ygrid

    if zline.shape[0] > 1:
        wake['P'] = (pot0+pot1).reshape([xline.shape[0],zline.shape[0]])
        wake['D'] = (den0+den1).reshape([xline.shape[0],zline.shape[0]])
        wake['fR'] = halo_rforce.reshape([xline.shape[0],zline.shape[0]])
        wake['R'] = rr.reshape([xline.shape[0],zline.shape[0]])
        wake['fP'] = potp.reshape([xline.shape[0],zline.shape[0]])
        wake['fZ'] = halo_zforce.reshape([xline.shape[0],zline.shape[0]])

    else:
        wake['P'] = pot0+pot1
        wake['D'] = den0+den1
        wake['fR'] = halo_rforce
        wake['R'] = rr
        wake['fP'] = potp
        wake['fZ'] = halo_zforce

        
    return wake




def read_binary_sl_coefficients(coeffile):
    '''
    read_binary_sl_coefficients
        definitions to read EXP-generated binary coefficient files (generated by SphericalBasis.cc dump_coefs)
        the file is self-describing, so no other items need to be supplied.

    inputs
    ----------------------
    coeffile   : input coefficient file to be parsed

    returns
    ----------------------
    times      : vector, time values for which coefficients are sampled
    coef_array : (rank 3 matrix)
                 0: times
                 2: azimuthal (L) order
                 3: radial order

    '''


    f = open(coeffile)

    # get the length of the file
    f.seek(0, os.SEEK_END)
    filesize = f.tell()

    # return to beginning
    f.seek(0)

    
    [string1] = np.fromfile(f, dtype='a64',count=1)
    [time0,scale] = np.fromfile(f, dtype=np.float,count=2)
    [nmax,lmax] = np.fromfile(f, dtype=np.uint32,count=2)

    # hard-coded to match specifications.
    n_outputs = int(filesize/(8*(lmax*(lmax+2)+1)*nmax + 4*2 + 8*2 + 64))


    # set up arrays given derived quantities
    times = np.zeros(n_outputs)
    coef_array = np.zeros([n_outputs,lmax*(lmax+2)+1,nmax])


    # return to beginning
    f.seek(0)


    for tt in range(0,n_outputs):
        
        [string1] = np.fromfile(f, dtype='a64',count=1)
        [time0,scale] = np.fromfile(f, dtype=np.float,count=2)
        [nmax,lmax] = np.fromfile(f, dtype=np.uint32,count=2)
        
        times[tt] = time0

        for nn in range(0,nmax):

            coef_array[tt,:,nn] = np.fromfile(f, dtype=np.float,count=lmax*(lmax+2)+1)

            
        #for ll in range(0,lmax*(lmax+2)+1):
            
            #coef_array[tt,ll,:] = np.fromfile(f, dtype=np.float,count=nmax)



    return times,coef_array


