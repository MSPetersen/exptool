############
#
# CODE FOR SHADOW BAR PLOTS
#        -HALO DENSITY OVER TIME, HALO DENSITY COMPARED TO NORMAL
#        -HALO VELOCITY COMPARED TO NORMAL, ON/OFF BAR
#
#
'''
Fourier methods for images

'''



def excise_center(xpos,ypos,zpos,xvel,yvel,zvel,pote,mass,cutoff):
    r = (xpos*xpos+ypos*ypos+zpos*zpos)**0.5
    lowr = np.where(r < cutoff)[0]
    return xpos[lowr],ypos[lowr],zpos[lowr],xvel[lowr],yvel[lowr],zvel[lowr],pote[lowr],mass[lowr]



def quik_transform(xpos,ypos,time,barfile):
    tst = SimReadThree()
    tst.X = np.zeros([1,len(xpos)])
    tst.X[0,:] = xpos
    tst.Y = np.zeros([1,len(ypos)])
    tst.Y[0,:] = ypos
    tst.T = time
    tst.bartransform(barfile)
    return tst.TX[0,:],tst.TY[0,:]


def add_file(infile,runtag,samplei,rcut):
    filein = infile+'.'+runtag+'.%05.i' %samplei
    times,massi,xposi,yposi,zposi,xveli,yveli,zveli,potei,nbodies,comptitle = psp_full_read(filein,'dark',2,10000000)
    xpos,ypos,zpos,xvel,yvel,zvel,pote,mass = excise_center(xposi,yposi,zposi,xveli,yveli,zveli,potei,massi,rcut)
    return xpos,ypos,zpos,xvel,yvel,zvel,pote,mass,times


def within_annulus(xpos,ypos,zpos,rcentral,rwidth,zheight):
    # calculate whether a particle is within a given annulus (and some given height cut)
    #
    rval = (xpos**2.+ypos**2.)**0.5
    if ( (rval > rcentral-rwidth) &  (rval < rcentral+rwidth) & (abs(zpos) < zheight) ): return 1
    else: return 0




# make the 3d density

binz = 50
tdbins = np.linspace(-0.07,0.07,binz)
dtd = tdbins[1]-tdbins[0]
tddist = np.zeros([binz,binz,binz])

for i in range(0,len(XPOS)):
    indx = int(np.floor( (XPOS[i] - tdbins[0])/dtd))
    indy = int(np.floor( (YPOS[i] - tdbins[0])/dtd))
    indz = int(np.floor( (ZPOS[i] - tdbins[0])/dtd))
    if (indx>=0) and (indx<len(tdbins)) and  (indy>=0) and (indy<len(tdbins)) and  (indz>=0) and (indz<len(tdbins)):
        tddist[indx,indy,indz] += MASS[i]



# make a 2d density

binz = 50
tdbins = np.linspace(-0.04,0.04,binz)
dtd = tdbins[1]-tdbins[0]
tddist = np.zeros([binz,binz])

for i in range(0,len(XPOS)):
    indx = int(np.floor( (XPOS[i] - tdbins[0])/dtd))
    indy = int(np.floor( (YPOS[i] - tdbins[0])/dtd))
    indz = int(np.floor( (ZPOS[i] - tdbins[0])/dtd))
    if (indx>=0) and (indx<len(tdbins)) and  (indy>=0) and (indy<len(tdbins)) and  (indz>=0) and (indz<len(tdbins)):
        tddist[indx,indy] += MASS[i]




        
# coordinate colors by percentile?

vals=  tddist.reshape(-1,)
percs = np.linspace(10,100,12)
cpl = np.percentile(np.log10(tddist),percs)



revmin = np.percentile(np.log10(tddist),5)

tddist[np.where(np.log10(tddist) < revmin)[0]] = 10.**revmin

cpl = np.linspace(revmin,np.max(np.log10(vals)),48)



plt.plot(np.log10(vals[vals.argsort()]),color='black')
for i in cpl: plt.plot([0,2500],[i,i],'--',color='black',lw=1.)


cpl = np.linspace(np.min(np.log10(vals)),np.max(np.log10(vals)),24)

xx,yy = np.meshgrid(tdbins,tdbins)
cbar = plt.contourf(xx+dtd,yy+dtd,np.log10(tddist).T,48)

goodc = np.linspace(0,2,48)
relcu = 100.-10**goodc
colv = relcu[relcu.argsort()]
cpl = np.percentile(tddist,colv)

#
# make the particle arrays by stacking timesteps
#
t1 = time.time()
startimg = 0
numsteps = 1
zcut = 0.001

xpos = [ [] for i in range(0,numsteps)]
ypos = [ [] for i in range(0,numsteps)]
zpos = [ [] for i in range(0,numsteps)]
xvel = [ [] for i in range(0,numsteps)]
yvel = [ [] for i in range(0,numsteps)]
zvel = [ [] for i in range(0,numsteps)]
pote = [ [] for i in range(0,numsteps)]
mass = [ [] for i in range(0,numsteps)]
times = [ [] for i in range(0,numsteps)]
for i in range(0,numsteps):
    xpos_tmp,ypos_tmp,zpos_tmp,xvel_tmp,yvel_tmp,zvel_tmp,pote_tmp,mass_tmp,times_tmp = add_file(infile,runtag,startimg+i,0.06,comp)
    xbar_tmp,ybar_tmp = quik_transform(xpos_tmp,ypos_tmp,times_tmp,barfile)
    xpos[i] = xbar_tmp
    ypos[i] = ybar_tmp
    zpos[i] = zpos_tmp
    xvel[i] = xvel_tmp
    yvel[i] = yvel_tmp
    zvel[i] = zvel_tmp
    mass[i] = mass_tmp


#
# calculate the velocity of the bar as a function of radius?? For the ploting if nothing else
#

    
# flatten these out
XPOS = np.concatenate(xpos)
YPOS = np.concatenate(ypos)
ZPOS = np.concatenate(zpos)
XVEL = np.concatenate(xvel)
YVEL = np.concatenate(yvel)
ZVEL = np.concatenate(zvel)
#POTE = np.concatenate(pote)
MASS = np.concatenate(mass) 
#TIMES = np.concatenate(times)

#print 'It took %3.2f seconds to build the particle list.' % (time.time()-t1)




xpos_tmp,ypos_tmp,zpos_tmp,xvel_tmp,yvel_tmp,zvel_tmp,pote_tmp,mass_tmp,times_tmp = add_file(infile,runtag,startimg+i,0.06,comp)

times,massi,xposi,yposi,zposi,xveli,yveli,zveli,potei,nbodies,comptitle



TI,MASS,XPOS,YPOS,ZPOS,XVEL,YVEL,ZVEL,POTE,NP,CT = psp_full_read('/Users/mpetersen/Research/NBody/Disk064a/OUT.run064a.01000','star',2,10000000)



rbins = np.linspace(0.0,0.1,20)

RVAL = (YPOS**2.+XPOS**2.)**0.5

lowx = np.where(RVAL < 0.1)[0]
xpos = XPOS[lowx]
ypos = YPOS[lowx]
mass = MASS[lowx]


def compute_fourier(XPOS,YPOS,MASS,RBINS,mmax):
    r_dig = np.digitize( (YPOS**2.+XPOS**2.)**0.5,rbins,right=True)
    aval = np.zeros([mmax,len(rbins)])
    bval = np.zeros([mmax,len(rbins)])
    for indx,r in enumerate(rbins):
        yes = np.where( r_dig == indx)[0]
        #print len(yes)
        aval[0,indx] = np.sum(MASS[yes])
        for m in range(1,mmax):
            aval[m,indx] = np.sum(MASS[yes] * np.cos(float(m)*np.arctan2(YPOS[yes],XPOS[yes])))
            bval[m,indx] = np.sum(MASS[yes] * np.sin(float(m)*np.arctan2(YPOS[yes],XPOS[yes])))
    return aval,bval



aval,bval = compute_fourier(xpos,ypos,mass,rbins,20)
mvals = np.linspace(0,20,1)
imval = (aval**2.+bval**2.)**0.5

