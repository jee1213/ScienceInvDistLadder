# This code creates a grid of velocity dispersion anisotropy parameters (for Osipkov-Merritt, hereafter OM,
# and its two-parameter extension model, hereatfer TPE model).
# The code is parallelized in OpenMP (shared memory). 

from scipy import interpolate
import numpy
import sys, distances, math, os
from math import log10,pi
import cPickle

# to get the cosmology samples:
# =============================

# redshift of the lens (zl) ad the source (zs)
zl = 0.295 
zs = 0.654 

nsamp = 100 #todo: update
DdsDsmax = 0.8
DdsDsmin = 0.2
DdsDs_samp = numpy.random.random(nsamp)*(DdsDsmax-DdsDsmin) + DdsDsmin
#DdsDs_samp = numpy.ones(nsamp)*DdsDsmin

Ddmax = 1500
Ddmin = 100
Dd_samp = numpy.random.random(nsamp)*(Ddmax-Ddmin) + Ddmin

Dt_samp = (1+zl)*Dd_samp/DdsDs_samp

wht_samp_prior = numpy.ones(nsamp)


# Importance sample the Power-Law mass model:
# ===========================================
# ===========================================

# Posterior probability distribution of the key lensing parameters for the power-law model can be read from the chain "RXJ1131_PLMod_GaussChain.dat"
# The columns are: weight, gam, Dtmod, reP, qP, gext
# where weight is the weight of the sample, gam is the density profile index of the power-law model (kappa(r) propto r^(-2*gam)), Dtmod is the modeled time-delay distance, reP is the normalisation parameter (close to the spherical Einstein radius, see below) of the primary lens galaxy (in arcsec), qP is the axis ratio of the primary lens galaxy, gext is the external shear strength

# Given the samples of the Dt, kappa_ext and gam from priors, can importance sample with the lensing likelihood P(Dtmod,gam), to obtain the lensing weights.


# ====================================================


# the spherical equivalent einstein radius from lensing chain is
LT_thE = pow((2./(1.+LT_qP)),(1./(2.*LT_gam))) * numpy.sqrt(LT_qP) * LT_reP

nLTsamp = LT_reP.size

nwrap = nsamp/nLTsamp + 1
thE_samp = numpy.tile(LT_thE, nwrap)[0:nsamp]


# compute the mass within thE given the cosmo samples (for Dds, Ds, Dd from Sigma_crit)
# recall in solar masses:
#    m_physical = Dd^2 * pi * thE^2 * Sigma_crit
#               = c^2/(4*G) * Dd*Ds / Dds * thetaE^2 
# where thetaE is in radians

# factor to convert thE in arcsec and Dist in Mpc into m in M_solar
mfact = pow(2.9979e8,2)*0.25/6.6738e-11 * 1.e6 * 3.08568e16 * pow((pi/(180.*3600.)),2) / 1.98892e30
mE_samp = mfact * Dd_samp / DdsDs_samp * thE_samp * thE_samp


# get the likelihood of the dynamics data by comparing the estimated velocity dispersion given
# anisotropy parameters (to be marginalized) to the observed luminosity-weighted projected veloity dispersion
# ==========================================
from numpy.random import random as R, randn as RN
from vdmodel import sigma_model as sm
# convert slope from lensing model to gamma' (gamma' = 2*gam + 1)
#gammaP_samp = gam_samp * 2. + 1

# inputs needed to dynamics code:
reff = 1.85 
seeing = 0.7  
ap = [-0.405,0.405,-0.35,0.35]
vd = 323. 
vderr = 20.

# have the following for dynamics:
# cosmo samps: Dd_samp, DdsDs_samp
# mE_samp
# thE_samp
# kext_samp
# gammaP_samp

# have anisotropy parameters

mass_samp = (1-kext_samp)*mE_samp # mass of galaxy

# if mass is negative, set to zero
for i in range(0,nsamp):
    if (mass_samp[i] < 0.):
        print "mass < 0 at i=",i,"  Will set to zero so dyn code doesn't break"
        mass_samp[i] = 0.


# For OM profile, create a grid (with e.g. gamma'= [1,3] and rani = [0.5*reff,5*reff])
gamma = #numpy.linspace(1.,3.,121)
riso = #numpy.logspace(log10(0.5*reff),log10(5*reff),121)

import time
t = time.time()
# the module modelGrids estimate the velocity disperion at each points on the 2D grid (gamma',rani)
model = sm.modelGrids(gamma,riso,ap,reff,seeing)
print "Time to create grid: ",time.time()-t
filesigv = #name of the output velocity dispersion grid file
f = open(filesigv,'wb')
cPickle.dump(model,f,2)
f.close()


# Compute model sigma and logp
# getSigmaFromGrid2 interpolates the grid at given anisotropy parameters (gammaP_samp, riso_samp)
# and normalize using (mass_samp, thE_samp, Dd_samp) to estimate the velocity dispersion at the sample point 
sigma = sm.getSigmaFromGrid2(mass_samp, thE_samp, gammaP_samp, riso_samp, model,Dd_samp)
# logp returns the likelihood of the velocity dispersion
logp = -0.5*(vd-sigma)**2/vderr**2 - 0.5*numpy.log(2*pi*vderr**2)

wht_samp_D = numpy.exp(logp)
vd_PL_samp = sigma


# Importance sample the 2-component mass model:
# ===========================================
# ===========================================

# to get kext samples 
# ====================
# kext text file with 2 columns to describe the cumulative distribution function (cdf) for kext
filename = 'RXJ1131_kext_CompMod.txt'

kext_cdf_table = numpy.loadtxt(filename)

# extract only the entries with unique cdf value (for interpolation with cdf as x coordinate)
kext_val_uniq = numpy.empty(0)
kext_cdf_uniq = numpy.empty(0)

icount = 0
for i in range (0,len(kext_cdf_table)):
    if (kext_cdf_table[icount,1] != kext_cdf_table[i,1]):
        # next entry is uniq, so add to list
        kext_val_uniq=numpy.append(kext_val_uniq, kext_cdf_table[i,0])

        kext_cdf_uniq=numpy.append(kext_cdf_uniq, kext_cdf_table[i,1])
        icount = i

kext_cdf_model = interpolate.splrep(kext_cdf_uniq, kext_val_uniq, s=0, k=1)

# get kext samples (array of size nsamp)
kext_2comp_samp = interpolate.splev(numpy.random.random(nsamp),kext_cdf_model)

# get uniform samples of the other lens parameters
nfw_rE_min = 0.05
nfw_rE_max = 0.38
nfw_rE_2comp_samp = numpy.random.random(nsamp)*(nfw_rE_max - nfw_rE_min) + nfw_rE_min


nfw_rs_min = 12.
nfw_rs_max = 30.
nfw_rs_2comp_samp = numpy.random.random(nsamp)*(nfw_rs_max - nfw_rs_min) + nfw_rs_min

bary_MtoL_min = 1.
bary_MtoL_max = 3.
bary_MtoL_2comp_samp = numpy.random.random(nsamp)*(bary_MtoL_max - bary_MtoL_min) + bary_MtoL_min

Dt_2comp_samp = (1.-kext_2comp_samp)*Dt_samp

# same prior as the PL model above: this time we have TPE, so beta_in and beta_out are also sampled
rani_min = ranilo*reff
rani_max = ranihi*reff
rani_samp = numpy.random.random(nsamp)*(rani_max - rani_min) + rani_min

bin_min = -0.6
bin_max = 0.6
bin_samp = numpy.random.random(nsamp)*(bin_max - bin_min) + bin_min

bout_min = -0.6
bout_max = 0.6
bout_samp = numpy.random.random(nsamp)*(bout_max - bout_min) + bout_min

# get the lensing+tdelay likelihood weights based on Gaussian approximation
# ==========================================

meanGfilename = 'RXJ1131_CompMod_GaussMean_4par.dat'
meanGauss2comp = numpy.loadtxt(meanGfilename)

covGfilename = 'RXJ1131_CompMod_GaussCov_4par.dat'
covGauss2comp = numpy.loadtxt(covGfilename)

covInvGauss2comp = numpy.linalg.inv(covGauss2comp)

wht_samp_LT_2comp = numpy.zeros(nsamp)

for i in range (0,nsamp):
    dx = numpy.array([nfw_rE_2comp_samp[i], nfw_rs_2comp_samp[i], bary_MtoL_2comp_samp[i], Dt_2comp_samp[i]])
    dx = dx - meanGauss2comp
    chi2 = numpy.dot(dx, numpy.dot(covInvGauss2comp, dx))
    wht_samp_LT_2comp[i] = numpy.exp(-0.5*chi2)



# get the dynamics likelihood weights (see Ale's bulget+halo_test.py)
# ===================================
import datetime
currenttime = datetime.datetime.now().time()
print 'current time before Ale dyn like comp ', currenttime

from vdmodel_2013 import sigma_model,profiles
from numpy import *
from cgsconstants import *
import cosmology
import NFW,tPIEMD

vd_2comp_samp = numpy.zeros(nsamp)

MpcToCm = 3.08567758e24
Dd_samp = MpcToCm * Dd_samp

S_cr_samp = c**2/(4*pi*G)/Dd_samp/DdsDs_samp/M_Sun*(Dd_samp*arcsec2rad)**2 

menc_NFW_samp = S_cr_samp*pi* nfw_rE_2comp_samp**2
norm_samp = numpy.ones(nsamp)
for i in range(0,nsamp):
    norm_samp[i] = menc_NFW_samp[i]/NFW.M2d(nfw_rE_2comp_samp[i],nfw_rs_2comp_samp[i])

# the two piemds describing the baryons
wc1 = 2.031239
wt1 = 2.472729
rein1_samp = bary_MtoL_2comp_samp * 5.409 
q1 = 0.882587
rc1 = wc1*2*q1**0.5/(1.+q1)
rt1 = wt1*2*q1**0.5/(1.+q1)

sigma01_samp = S_cr_samp*rein1_samp/2.*(1./wc1 - 1./wt1)
Mstar1_samp = sigma01_samp/tPIEMD.Sigma(0.,[rc1,rt1])


wc2 = 0.063157
wt2 = 0.667333
rein2_samp = bary_MtoL_2comp_samp * 1.26192 
q2 = 0.847040
rc2 = wc2*2*q2**0.5/(1.+q2)
rt2 = wt2*2*q2**0.5/(1.+q2)

sigma02_samp = S_cr_samp*rein2_samp/2.*(1./wc2 - 1./wt2)
Mstar2_samp = sigma02_samp/tPIEMD.Sigma(0.,[rc2,rt2])

Mstar_samp = Mstar1_samp + Mstar2_samp

a1_samp = Mstar1_samp/Mstar_samp
a2_samp = Mstar2_samp/Mstar_samp

# Note: a1 and a2 are the same for all samples, since the ratio of the two light profiles does not change.  Set it to the value a1 and a2.
a1 = a1_samp[0]
a2 = a2_samp[0]

# for removing kext contribution
rein = 1.63 
m2d_tpiemd1 = tPIEMD.M2d(rein,[rc1,rt1])
m2d_tpiemd2 = tPIEMD.M2d(rein,[rc2,rt2])

#defines the vertices of the rectangular aperture to calculate vdisp in.
aperture = [0.,0.35,0.,0.405]
seeing = 0.7


# stars: 
# =====
currenttime = datetime.datetime.now().time()
print 'current time before gridding and interpolating stars s2', currenttime

starsfile = "TPE_stargrid"
try:
	print starsfile
	if starsfile ==0:
		print "cannot find the file"
	else:
		print "star file found"
	f = open(starsfile,'rb')
	stars_s2_interp_model = cPickle.load(f)
	f.close()
	print "star grid read"
#    axis_rani = gridmodel[:,0]
#    b_in = gridmodel[:,1]
#    b_out = gridmodel[:,2]
#    stars_s2_grid = gridmodel[:,3]
except:
    # evaluate onto grid for interpolation on rani
    print "star grid does not exist"
    ngrid = 121
    drani = (rani_max - rani_min)/(float(ngrid-1))
    axis_rani = (numpy.arange(ngrid))*drani + rani_min 

    stars_s2_grid = numpy.zeros(ngrid)
    lp_pars = [a1,rc1,rt1,a2,rc2,rt2]

    for i in range(ngrid):
        stars_s2_grid[i] = sigma_model.sigma2general(lambda r: a1*tPIEMD.M3d(r,[rc1,rt1]) + a2*tPIEMD.M3d(r,[rc2,rt2]),aperture,lp_pars,seeing=seeing,light_profile=profiles.twotPIEMD,reval0=0.5*(rt1 + rt2), anisotropy='TPE', anis_par=axis_rani[i])

    # write to output file
    fout = open(starsfile, 'w')
    fout.write("# r_ani (arcsec), stars_s2 (unnormalized) \n")

    for i in range(ngrid):
        fout.write("%le\t%le\n"%(axis_rani[i], stars_s2_grid[i]))

    fout.close()


# halo:
# =====
currenttime = datetime.datetime.now().time()
print 'current time before gridding and interpolating halo s2', currenttime

halofile = 'TPE_halogrid'

ngrid = 121

try:
    f = open(halofile,'rb')
    halo_s2_interp_model = cPickle.load(f)
    f.close()
    print "halo grid read"
        

except:
    # evaluate onto grid for interpolation
    print "halo grid does not exist"
    drs = (nfw_rs_max - nfw_rs_min)/(float(ngrid-1))
    axis_rs = (numpy.arange(ngrid))*drs + nfw_rs_min

    drani = (rani_max - rani_min)/(float(ngrid-1))
    axis_rani = (numpy.arange(ngrid))*drani + rani_min

    halo_s2_grid = numpy.zeros((ngrid,ngrid))
    lp_pars = [a1, rc1,rt1, a2, rc2,rt2]

    for i in range(ngrid):
        for j in range(ngrid):
            halo_s2_grid[i,j] = sigma_model.sigma2general(lambda r: NFW.M3d(r,axis_rs[i]),aperture,lp_pars,seeing=seeing,light_profile=profiles.twotPIEMD,reval0=0.5*(rt1 + rt2), anisotropy='TPE', anis_par=axis_rani[j])

    # write to output file
    fout = open(halofile, 'w')
    fout.write("# nfw_rs, halo_s2 (unnormalized) \n")

    for i in range(ngrid):
        for j in range(ngrid):
            fout.write("%le\t%le\t%le\n"%(axis_rs[i], axis_rani[j], halo_s2_grid[i,j]))

    fout.close()


## evaluation of predicted velocity dispersion of samples
# ------------------------------------------------------
currenttime = datetime.datetime.now().time()
print 'current time before computing s2 of samples', currenttime

# get the mass normalizations of the stars and halo distributions
for i in range(0,nsamp):
    #now lets remove the external convergence: that's 0.07 within the measured Einstein radius of 1.63".
    Menc_stars = Mstar1_samp[i]*m2d_tpiemd1 + Mstar2_samp[i]*m2d_tpiemd2
    Menc_halo = norm_samp[i] *NFW.M2d(rein,nfw_rs_2comp_samp[i])
    fs = Menc_stars/(Menc_stars + Menc_halo)
    fh = Menc_halo/(Menc_stars + Menc_halo)

    Mstar_samp[i] *= 1 - kext_2comp_samp[i]
    norm_samp[i] *= 1 - kext_2comp_samp[i]


stars_evalpoints = numpy.empty((nsamp,3))
stars_evalpoints[:,0] = rani_samp
stars_evalpoints[:,1] = bin_samp
stars_evalpoints[:,2] = bout_samp
print stars_evalpoints
s2_stars_samp = stars_s2_interp_model(stars_evalpoints)
## get into km/s units
s2_stars_samp *= G*Mstar_samp*M_Sun/10.**10/(Dd_samp*arcsec2rad)


halo_evalpoints = numpy.empty((nsamp,4))
halo_evalpoints[:,0] = nfw_rs_2comp_samp
halo_evalpoints[:,1] = rani_samp
halo_evalpoints[:,2] = bin_samp
halo_evalpoints[:,3] = bout_samp
s2_halo_samp = halo_s2_interp_model(halo_evalpoints)
# get into km/s units
s2_halo_samp *= G*norm_samp*M_Sun/10.**10/(Dd_samp*arcsec2rad)
vd_2comp_samp = (s2_stars_samp + s2_halo_samp)**0.5
wht_samp_D_2comp = numpy.exp(-0.5*(vd - vd_2comp_samp)**2/vderr**2)

# scale back Dist to Mpc 
Dd_samp = Dd_samp / MpcToCm

currenttime = datetime.datetime.now().time()
print 'current time after Ale dyn like comp ', currenttime




