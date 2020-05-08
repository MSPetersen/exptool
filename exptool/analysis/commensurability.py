"""

commensurability.py: part of exptool
      tools to handle various commensurability finding items


Aval = calculate_area(Orbit,ratio=5.)
print(Aval)



"""
from __future__ import absolute_import, division, print_function, unicode_literals

# standard imports
import numpy as np

#
# not relevant here, but removed in matplotlib 2.2, use skimage.measure.find_contour instead
#
#from matplotlib import _cntr as cntr



import matplotlib.pyplot as plt
import matplotlib.cm as cm

# exptool imports
from exptool.utils import utils


# also check the scipy
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import filters
import scipy.ndimage.filters

from scipy.spatial import Delaunay
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.Delaunay.html


# need a check here to see if this will actually import and a clause if not

try:
    from skimage.morphology import skeletonize
    able_to_skel = True
except:
    able_to_skel = False






def calculate_area(Orbit,ratio=10.,usex='TX',usey='TY'):
    """calculate the area of an orbit using volume tesselation

    inputs
    --------------------
    Orbit : dictionary of orbit quantities
    ratio : (float, default=10.) ratio of minimum side length to maximum to not count toward the area total
    usex  : (string, default='TX') key in dictionary for x coordinate
    usey  : (string, default='TY') key in dictionary for y coordinate

    returns
    -------------------
    the area value, normalized by the maximum circle area



    todo
    ------------------
    1. turn into 3d
    
    """
    points = np.array([[Orbit[usex][x],Orbit[usey][x]] for  x in range(0,len(Orbit[usex]))])

    # do the triangulation
    tri = Delaunay(points)


    A = np.zeros(tri.simplices.shape[0])
    legs = np.zeros([tri.simplices.shape[0],3])



    for t in range(0,tri.simplices.shape[0]):
        
        xvals = tri.simplices[t]
        x1 = Orbit[usex][xvals[0]]
        x2 = Orbit[usex][xvals[1]]
        x3 = Orbit[usex][xvals[2]]
        y1 = Orbit[usey][xvals[0]]
        y2 = Orbit[usey][xvals[1]]
        y3 = Orbit[usey][xvals[2]]

        A[t] = 0.5*((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    
        tmplegs = np.array([np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)),\
                       np.sqrt((x1-x3)*(x1-x3) + (y1-y3)*(y1-y3)),\
                       np.sqrt((x2-x3)*(x2-x3) + (y2-y3)*(y2-y3))])
    
        legs[t] = tmplegs[tmplegs.argsort()]

    med = np.median(legs[:,0])
    include = legs[:,1]/med
    
    # this has the normalization in it now
    return np.sum(A[include < ratio])/(np.pi*np.nanmax(Orbit['X'])*np.nanmax(Orbit['X']))
    
    










# read in orbit integrations

# follows file format of integrator output
def read_integrations(infile):
    f = open(infile,'r')
    D = {}
    D['R'] = {}
    D['V'] = {}
    D['dT'] = {}
    D['TX'] = {}
    D['TY'] = {}
    #
    linenum = 0
    for line in f:
        d = [float(q) for q in line.split()]
        npoints = int(d[0])
        D['R'][linenum] = d[1]
        D['V'][linenum] = d[2]
        D['dT'][linenum] = d[3]
        D['TX'][linenum] = d[4:npoints+4]
        D['TY'][linenum] = d[npoints+4:(2*npoints)+4]
        linenum += 1
    #
    f.close()
    return D





def make_orbit_plot(D,num,outdir='',T=''):
    plt.clf()
    fig = plt.gcf()
    ax = fig.add_axes([0.25,0.25,0.6,0.6])
    ax.plot(D['TX'][num],D['TY'][num])
    ax.set_title('T'+T+', R'+str(np.round(D['R'][num],3))+', V'+str(np.round(D['V'][num],3)),size=18)
    ax.set_ylabel('Ybar',size=18)
    ax.set_xlabel('Xbar',size=18)
    ax.axis([-1.1*D['TX'][num][0],1.1*D['TX'][num][0],-1.1*D['TX'][num][0],1.1*D['TX'][num][0]])
    for label in ax.get_xticklabels(): label.set_rotation(30); label.set_horizontalalignment("right")
    plt.savefig(outdir+'T'+T+'R'+str(np.round(D['R'][num],3))+'V'+str(np.round(D['V'][num],3))+'.png')




'''
    
D = np.genfromtxt('/scratch/mpetersen/processed.txt')

Rarr = D[:,0].reshape(rads.size,vels.size)
Varr = D[:,1].reshape(rads.size,vels.size)
Aarr = D[:,2].reshape(rads.size,vels.size)





# pull a contour from the area array
c = cntr.Cntr(Rarr.T,Varr.T,Aarr.T/(np.pi*Rarr.T*Rarr.T))

res = c.trace(0.03,nchunk=2)

for j in range(0,len(res)):
    xcon = []
    ycon = []
    try:
        if len(res[j])>10:
            for i in range(0,len(res[j])):
    	        xcon.append(res[j][i][0])
    	        ycon.append(res[j][i][1])
            #print xcon,ycon
            XCON = np.array(xcon)
            YCON = np.array(ycon)
            ax2.plot(XCON,YCON,color='red')
        else:
	    pass
    except:
	pass


'''



def map_skeleton(Rarr,Varr,Aarr,\
                 sigma=(3.,3.),ridge_cutoff=0.2,\
                 scaling = 256.,scalefac = 6.,skelcut=0.002):
    '''
    map_skeleton:
         take a grid of integrated orbits




    '''
    
    area_map = scaling*Aarr
    
    
    
    # add a pre-process
    area_map[area_map > scaling/scalefac] = scaling/scalefac
    
    area_map = np.log10(area_map)

    #
    # only do the needed derivatives
    #
    #rp_00 = filters.gaussian_filter(area_map,sigma,order=0)      # r
    #rp_01 = filters.gaussian_filter(area_map,sigma,order=(0,1))  # ry
    #rp_10 = filters.gaussian_filter(area_map,sigma,order=(1,0))  # rx
    rp_11 = filters.gaussian_filter(area_map,sigma,order=(1,1))  # rxy
    rp_20 = filters.gaussian_filter(area_map,sigma,order=(2,0))  # rxx
    rp_02 = filters.gaussian_filter(area_map,sigma,order=(0,2))  # ryy
    
    #
    # set up the hessian
    #
    hess = np.array([[rp_20.T,rp_11.T],[rp_11.T,rp_02.T]]).T

    #
    # solve the eigenvalue problem
    #
    w, v = np.linalg.eig(hess)
    
    #
    # find maximum eigenvalue
    # 
    ridge_map = np.max(w,axis=2)
    
    #
    # now threshold the map, which includes an adaptive cutoff
    #
    threshold_map = np.zeros_like(ridge_map)
    good_values = np.where(ridge_map > ridge_cutoff)
    threshold_map[good_values] = 1.

    if able_to_skel:
        skeleton = skeletonize(threshold_map)

        # alternate methodology to explore
        #skeleton, distance = medial_axis(threshold_map, return_distance=True)


        #
        # nan the irrelevant values
        #
        ridge_trace = np.zeros_like(ridge_map)
        good_values = np.where(skeleton < skelcut)
        ridge_trace[good_values] = np.nan
        
    else:
        print('exptool.commensurability: skimage is not available. falling back to non-pruned area plots.')

        ridge_trace = threshold_map
        
    
    return ridge_trace


    
def print_skeleton(infile,pskel,Rarr,Varr):

    f = open(infile,'w')

    pp = np.flipud(pskel).reshape(-1,)
    ccc = (Rarr.T).reshape(-1,)
    ddd = (Varr.T).reshape(-1,)

    for indx in range(0,len(pp)):
        print >>f,ccc[indx],ddd[indx],pp[indx]
    
    f.close()




def make_fishbone(infile):
    E = np.genfromtxt(infile) 
    #
    rads = np.unique(E[:,0])
    vels = np.unique(E[:,1])
    #
    Rarr,Varr = np.meshgrid(rads,vels)  
    Rarr = Rarr.T; Varr = Varr.T
    Aarr = E[:,2].reshape(rads.size,vels.size)
    #
    dd = Aarr.T/(np.pi*Rarr.T*Rarr.T)
    #
    return Rarr.T,Varr.T,dd






def construct_live_fishbones(infile,percentile=25.,particle_limit=0,rads = np.linspace(0.,0.05,80),vels = np.linspace(-0.2,1.6,80)):
    '''
    construct_live_fishbones



    inputs
    ----------------------
    infile         :              input file with calculated orbit areas
    percentile     : (default=25)
    particle_limit : (default=0)  minimum number of particles per bin to consider
    rads           : (default)
    vels           : (default)



    returns
    ----------------------
    LF             :              dictionary with keys listed below
       rr
       vv
       min
       mean
       med
       std
       perc
       


    '''
    E = np.genfromtxt(infile)
    
    # transform to relative area
    AA = E[:,2]/(np.pi*E[:,0]*E[:,0])

    #
    # brute force the binning
    #  
    dr = rads[1]-rads[0]
    dv = vels[1]-vels[0]
    rr,vv = np.meshgrid(rads,vels)
    
    LF = {}
    LF['rr'] = rr
    LF['vv'] = vv
    
    LF['min']  = np.ones([rads.size,vels.size])
    LF['mean'] = np.ones([rads.size,vels.size])
    LF['med']  = np.ones([rads.size,vels.size])
    LF['std']  = np.ones([rads.size,vels.size])
    LF['perc'] = np.ones([rads.size,vels.size])


    for rindx,rad in enumerate(rads):
        for vindx,vel in enumerate(vels):

            # calculate particles in a given bin
            w = np.where( (np.abs(E[:,0] - rad) < dr) & (np.abs(E[:,1] - vel) < dv))

            if len(w[0]) > particle_limit:
                LF['min'][rindx,vindx]  = np.min(AA[w])
                LF['mean'][rindx,vindx] = np.mean(AA[w])
                LF['med'][rindx,vindx]  = np.median(AA[w])
                LF['std'][rindx,vindx] = np.std(AA[w])
                
                arr_aa = AA[w][AA[w].argsort()]
                quartile = int(np.floor(len(w[0])*float(percentile/100.)))
                LF['perc'][rindx,vindx] = arr_aa[quartile]

    return LF




