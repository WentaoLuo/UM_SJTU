#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import gglens
import camb
from camb import model, initialpower
import emcee 
import mcfit
from scipy import integrate
from optparse import OptionParser
import warnings
warnings.filterwarnings('ignore')
#This code is to modelling GGlensing signal
# using NFW and two point correlation functions.

# Basic Parameters---------------------------
h       = 1.0
w       = -1.0
omega_m = 0.315
omega_l = 0.685
omega_k = 1.0-omega_m-omega_l
rho_crt0= 2.78e11                # M_sun Mpc^-3 *h*h 
rho_bar0= rho_crt0*omega_m       # M_sun Mpc^-3 *h*h
pi      = np.pi
nps     = 0.965
alphas  = -0.04
sigma8  = 1.048

#---PART I: generate gglensing for two halo term----
def simR(r,rx,corr,Rp):
  return np.interp(r,rx,corr)*r/np.sqrt(r*r-Rp*Rp)
def simLR(y,rx,corr,Rp):
  Rp   = y
  rSta = y+0.001
  rEnd = 90.0
  res  = y*integrate.quad(simR,rSta,rEnd,args=(rx,corr,Rp),epsabs=1e-2)[0]
  return res
def matterps2corr2esd(redshift,cosmology):
  z    = redshift
  H_null,Ombh2,Omch2=cosmology
  pars = camb.CAMBparams()
  pars.set_cosmology(H0=H_null,ombh2=Ombh2,omch2=Omch2)
  pars.set_dark_energy()
  pars.InitPower.set_params(ns=nps)
  pars.set_matter_power(redshifts=[0.0,z],kmax=2.0)

  #Linear spectra--------
  pars.NonLinear=model.NonLinear_none
  results       =camb.get_results(pars)
  kh,znon,pk    =results.get_matter_power_spectrum(minkh=1e-4,maxkh=1000.0,npoints=1024)
  xi            =mcfit.P2xi(kh,l=0)
  #Calculate corr from PS------
  nxx   = 50
  Rmin  = -2.0
  Rmax  = 2.0
  rr    = np.logspace(Rmin,Rmax,nxx) 
  rx,corrfunc   =xi(pk,extrap=True)
  corr  = np.interp(rr,rx,corrfunc[0,:])
  #Calculate ESD from corr------
  nxx   = 10
  Rp    = np.linspace(0.03,20.,nxx)   
  ESD   = np.zeros(nxx) 

  for i in range(nxx):
    tmp1     = integrate.quad(simR,Rp[i]+0.001,90.0,args=(rr,corr,Rp[i]),epsabs=1e-4)
    tmp2     = integrate.quad(simLR,0.001,Rp[i],args=(rr,corr,Rp[i]),epsabs=1e-4)
    ESD[i]   = (4.0*tmp2[0]/Rp[i]/Rp[i]-2.0*tmp1[0])*omega_m*h  #Luo et al 2017ApJ 836 38L EQ. 39-40

  return {'Rp':Rp,'ESD':ESD}

#---PART II: generate gglensing for one halo term----
#using NFW profile---- 
def funcs(Rp,rs):
  x   = Rp/rs
  x1  = x*x-1.0
  x2  = 1.0/np.sqrt(np.abs(1.0-x*x))
  x3  = np.sqrt(np.abs(1.0-x*x))
  x4  = np.log((1.0+x3)/(x))
  s1  = Rp*0.0
  s2  = Rp*0.0
  
  ixa = x>0.
  ixb = x<1.0
  ix1 = ixa&ixb
  s1[ix1] = 1.0/x1[ix1]*(1.0-x2[ix1]*x4[ix1])
  s2[ix1] = 2.0/(x1[ix1]+1.0)*(np.log(0.5*x[ix1])\
            +x2[ix1]*x4[ix1])

  ix2 = x==1.0
  s1[ix2] = 1.0/3.0
  s2[ix2] = 2.0+2.0*np.log(0.5)

  ix3 = x>1.0

  s1[ix3] = 1.0/x1[ix3]*(1.0-x2[ix3]*np.arctan(x3[ix3]))
  s2[ix3] = 2.0/(x1[ix3]+1.0)*(np.log(0.5*x[ix3])+\
            x2[ix3]*np.arctan(x3[ix3]))

  res = {'funcf':s1,'funcg':s2}
  return res
# one halo term-------------------------------------
def nfwesd(theta,z,Rp):
  Mh,c      = theta
  efunc     = 1.0/np.sqrt(omega_m*(1.0+z)**3+\
              omega_l*(1.0+z)**(3*(1.0+w))+\
              omega_k*(1.0+z)**2)
  rhoc      = rho_crt0/efunc/efunc
  omegmz    = omega_m*(1.0+z)**3*efunc**2
  ov        = 1.0/omegmz-1.0
  dv        = 18.8*pi*pi*(1.0+0.4093*ov**0.9052)
  rhom      = rhoc*omegmz

  r200 = (10.0**Mh*3.0/200./rhom/pi)**(1./3.)
  rs   = r200/c
  delta= (200./3.0)*(c**3)\
              /(np.log(1.0+c)-c/(1.0+c))
  amp  = 2.0*rs*delta*rhoc*10e-14

  functions = funcs(Rp,rs)
  funcf     = functions['funcf']
  funcg     = functions['funcg']
  esd       = amp*(funcg-funcf)

  return esd

# two halo term----------------------------------------
def galaxybias(logM):
  Mnl = 8.73*10e+12
  Mh  = 10.0**logM
  xx  = Mh/Mnl
  b0  = 0.53+0.39*xx**0.45+(0.13/(40.0*xx+1.0))\
        +5.0*0.0004*xx**1.5
  bias= b0+\
        np.log10(xx)*(0.4*(omega_m-0.3+nps-1)+\
        0.3*(sigma8-0.9+h-0.7)+0.8*alphas)
  return bias

#---PART III: likelihood for mcmc----
def lnprior(theta):
  logM,con= theta
  if 11.0<logM<20.0 and 4.0<con<4.5:
      return 0.0
  return -np.inf
#---------------------------------------------
def lnlike(theta,Rp,esd,esdtwo,covar,z):
  logM,con = theta
  #stellar  = M0/2.0/pi/r/r
  nrr      = len(Rp)
  #stars    = stellar[0:nrr]
  bias     = galaxybias(logM)
  model= nfwesd(theta,z,Rp)+bias*esdtwo
  cov  = np.dot(np.linalg.inv(covar),(model-esd))
  chi2 = np.dot((model-esd).T,cov)
  #invers= 1.0/err/err
  diff  = -0.5*(chi2)
  return diff.sum()
#---------------------------------------------------
def lnprob(theta,Rp,esd,esdtwo,err,z):
  lp = lnprior(theta)
  if not np.isfinite(lp):
        return -np.inf
  return lp+lnlike(theta,Rp,esd,esdtwo,err,z)
#---END of functions----------------
#---PART IV: MAIN function---------
def main():
   parser     = OptionParser()
   parser.add_option("--MODE",dest="MODE",default="EASY",
          help="EASY or HARD",metavar="value",
          type="string")
   (o,args)   = parser.parse_args()
   cosmology  = [100,0.022,0.122]

   redshift = 0.1
   data     = np.loadtxt('shear_c4_richd_mean',unpack=True) 
   covar    = np.loadtxt('covar_c4_richd',unpack=True) 
   std      = np.loadtxt('std_c4_richd',unpack=True) 
   covar[0,0]= covar[0,0]+std[0]*std[0]
   covar[1,1]= covar[1,1]+std[1]*std[1]
   covar[2,2]= covar[2,2]+std[2]*std[2]
   covar[3,3]= covar[3,3]+std[3]*std[3]
   Rp       = data[0,:]
   esd      = data[1,:]
   err      = data[2,:]

   if o.MODE=='HARD':
      results = matterps2corr2esd(redshift,cosmology)
      RR      = results['Rp']
      ESDR    = results['ESD']
      esdtwo  = np.interp(Rp,RR,ESDR)
   if o.MODE=='EASY':
      data= np.loadtxt('corrs',unpack=True,skiprows=2)
      RR   = data[0,:]
      ESDR = data[1,:]
      esdtwo= np.interp(Rp,RR,ESDR)

   logM = 14.0
   con  = 4.0
   zl   = 0.1562
   pars = np.array([logM,con])

   ndim,nwalkers = 2,200
   pos = [pars+1e-4*np.random.randn(ndim) for i in range(nwalkers)]
   sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(Rp,esd,esdtwo,covar,zl))
   sampler.run_mcmc(pos,2000)

   burnin = 200
   samples=sampler.chain[:,burnin:,:].reshape((-1,ndim))
   Mh,cn= map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples,[16,50,84],axis=0)))
   print 'Halo Mass:    ',Mh
   print 'concentration:',cn
   print galaxybias(Mh[0])

if __name__=='__main__':
   main()

#---END of every thing--------------
