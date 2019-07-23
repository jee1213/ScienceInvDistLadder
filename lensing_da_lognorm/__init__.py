import os
import numpy as np
from montepython.likelihood_class import Likelihood


class lensing_da_lognorm(Likelihood):

    # The initialization routine is no different from the parent class one.
    def __init__(self,path,data,command_line):
	Likelihood.__init__(self,path,data,command_line)
	self.zl = np.array([],'float64')
	self.da = np.array([],'float64')
        # read redshifts and data points
        for line in open(os.path.join(
                self.data_directory, self.data), 'r'):
            if (line.find('#') == -1):
                self.zl = np.append(self.zl, float(line.split()[0]))
                self.da = np.append(self.da, float(line.split()[1]))
#                self.da_error = np.append(self.da_error, float(line.split()[3]))

        # number of data points
        self.num_points = np.shape(self.zl)[0]
	print self.num_points
        # define correlation m,atrix
        lognorm_params = np.zeros((self.num_points, 3), 'float64')

        # file containing correlation matrix
        if self.has_syscovmat:
            param_filename = self.covmat_sys
        else:
            param_filename = self.covmat_nosys

        # read correlation matrix
        i = 0
        for line in open(os.path.join(
                self.data_directory, param_filename), 'r'):
            if (line.find('#') == -1):
                lognorm_params[i] = line.split()
                i += 1
	print lognorm_params
	self.shape = np.zeros((self.num_points),'float64')
	self.loc = np.zeros((self.num_points),'float64')
	self.scale = np.zeros((self.num_points),'float64')
	for m in range(self.num_points):
		self.shape[m] = lognorm_params[m][0]
		self.loc[m] = lognorm_params[m][1]
		self.scale[m] = lognorm_params[m][2]

    # compute likelihood
    def loglkl(self, cosmo, data):
	lkl = 0
	lensing_da_lognorm = np.zeros((self.num_points),'float64')
        if cosmo.Omega_fld() and -1.e-3 < cosmo.Omega_lambda() <1.e-3:
                params = [cosmo.Omega_m(),cosmo.Omega_fld(),cosmo.h(),cosmo.w0(),cosmo.wa()]
        else:
                params = [cosmo.Omega_m(),cosmo.Omega_lambda(),cosmo.h(),-1.,0.]
                #params = [self.cosmo_arguments['Omega_m'],self.cosmo_arguments['Omega_fld'],self.cosmo_arguments['h'],-1.,0.]
        import distances
        D = distances.Distance(params)

	for i in range(self.num_points):
	        lensing_da_lognorm[i] = D.angular_diameter_distance(self.zl[i])
		chi2 = np.log(lensing_da_lognorm[i]-self.loc[i])+((np.log(lensing_da_lognorm[i]-self.loc[i])-np.log(self.scale[i]))/self.shape[i])**2./2.
        # return ln(L)
  	        lkl -= chi2

        return lkl
