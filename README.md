# Introduction
This set of codes and data products is published for the readers of the Science article 
''Measurement of the Hubble constant from angular diameter distances to two gravitational lenses'' (Jee, Suyu, Komatsu et al. 2019).
Following the script, the readers will be able to reproduce the results presented in the paper.

# Code Summary
We provide data summarizing the lensing and cosmological analysis, and example scripts as follows.
## Lensing analysis
- the lensing parameters from the lensing analysis for RXJ1131-1231, provided either through a MCMC chain (file "RXJ1131_PLMod_GaussChain.dat" for the power-law mass model) or Gaussian covariance matrix (file "RXJ1131_CompMod_GaussMean_4par.dat" for the composite model).  For B1608+656, the information is provided in Suyu et al. (2010). <br>
- the external convergence distributions of B1608+656 and RXJ1131-1231 (Hilbert et al. 2009), as files with "kext" in them.<br>
- kinematics code implementing Osipkov-Merrit (OM) and its two-parameter extension (TPE) anisotropic velocity dispersion models.<br> 
The original code is developed by Matthew Auger, and a kernel method (Mamon & Lokas, 2004) has been implemented by Alessandro Sonnenfeld, which is available at http://github.com/astrosonnen/spherical_jeans).<br> 
We added TPE anisotropy calculation using the kernel method as introduced in Section S1.2. 
This modification of code is included in sigma_model_TPE.py and should replace the original file sigma_model.py.<br>
- an example script to call kinematics function is given <br>
- For additional discussions presented in supplementary section S1.2 regarding the spherical Jeans modeling assumption, 
Akin Yildirim performed analysis using the Jeans Anisotropic Model (JAM) developed by Michele Cappellari (Cappellari,2002). 
The code is available at https://www-astro.physics.ox.ac.uk/~mxc/software/#jam <br>
For more details on the axisymmetric power-law mass model and the axis ratio values tested, see Yildirim et al. 2019. 

## Cosmological analysis
- log-normal likelihood of the lensing angular diameter distances for MontePython, a Monte-Carlo code for cosmological parameter extraction,
(http://baudren.github.io/montepython.html) developed by Audren et al. 2013 and Brinkmann & Lesgourgues 2018, and associated with the CLASS code (Lesgourgues 2011).<br>
The folder lensing_da_lognormal can be placed in the likelihood directory of MontePython and can be called
using the name 'lensing_da_lognormal'.<br>
- the outputs of our MontePython runs are summarized in the files in the directory MontePython_outputs
