#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import gglens
import camb
from camb import model, initialpower
import emcee as mcham
from scipy.interpolate import interp1d
from scipy import integrate
from optparse import OptionParser

#This code is to modelling GGlensing signal
# using NFW and two point correlation functions.

# Basic Parameters---------------------------
h       = 1.0
w       = -1.0
omega_m = 0.28
omega_l = 0.72
omega_k = 1.0-omega_m-omega_l
rho_crt0= 2.78e11                # M_sun Mpc^-3 *h*h 
rho_bar0= rho_crt0*omega_m       # M_sun Mpc^-3 *h*h
pi      = np.pi
nps     = 0.965
alphas  = -0.04
sigma8  = 1.048

#---PART I: generate gglensing for two halo term----

def matterpower(redshift,cosmology):
  z    = redshift
  H_null,Ombh2,Omch2=cosmology
  pars = camb.CAMBparams()
  pars.set_cosmology(H0=H_null,ombh2=Ombh2,omch2=Omch2)
  pars.set_dark_energy()
  pars.InitPower.set_params(ns=nps)
  pars.set_matter_power(redshifts=[0.,z],kmax=1000.0)
#Linear spectra
  pars.NonLinear = model.NonLinear_none
  results = camb.get_results(pars)
  kh,zl,pk= results.get_matter_power_spectrum(minkh=1e-4,maxkh=1000,npoints=10000)
  s8      = np.array(results.get_sigma8())
#Non-Linear spectra(Halofit)
  pars.NonLinear = model.NonLinear_both
  results.calc_power_spectra(pars)
  kh_nonlin,z_nonlin,pk_nonlin=results.get_matter_power_spectrum(minkh=1e-4,maxkh=1000,npoints=10000)

  return {'kh':kh,'linpw':pk,'nonpw':pk_nonlin}

def ps2xi(kh,power,rr): 
  xi  = np.zeros(len(rr)) 
  step= (np.max(kh)-np.min(kh))/len(kh) 
  for i in range(len(rr)):
      xi[i] = (step*kh*kh*power*np.sin(kh*rr[i])/(kh*rr[i])).sum()/2.0/np.pi
  return xi 

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

def twohaloesd(Mh,rr,xi,Rp):
  bias = galaxybias(Mh)
  corr = bias*xi
  eta = (1.0+1.17*xi)**1.49/(1.0+0.69*xi)**2.09
  finterp1 = interp1d(rr,xi)
  finterp2 = interp1d(rr,eta)

  nr       = len(Rp)
  SigR     = np.zeros(nr)
  SigRR    = np.zeros(nr)
  warnings.simplefilter("ignore")
  esd  = np.zeros(len(nr))
  for i in range(nr):
      funcR    = lambda x:finterp1(x)*finterp2(x)*x/np.sqrt(x*x-Rp[i]*Rp[i])
      SigR[i],err= integrate.quad(funcR,Rp[i],100.0)
      funcRR   = lambda x,y:y*finterp1(x)*finterp2(x)*x/np.sqrt(x*x-y*y)
      glim     = lambda z:z
      hlim     = lambda z:100.0
      SigRR[i],err = integrate.dblquad(funcRR,np.min(rr),Rp[i],glim,hlim)
      esd[i]   = (4.0*SigRR[i]/Rp[i]/Rp[i]-SigR[i])*omega_m*h
  return esd
#---PART III: likelihood for mcmc----
def lnprior(theta):
  logM,con= theta
  if 12.0<logM<20.0 and 1.0<con<16.0:
      return 0.0
  return -np.inf
#---------------------------------------------
def lnlike(theta,z,Rp,esd,err):
  logM,con = theta
  #stellar  = M0/2.0/pi/r/r
  nrr      = len(Rp)
  #stars    = stellar[0:nrr]

  struct= nfwesd(theta,z,Rp)
  #model = stars+struct["res"]
  model = struct["res"]
  invers= 1.0/err/err
  diff  = -0.5*((esd-model)**2*invers-np.log(invers))
  return diff.sum()
#---------------------------------------------------
def lnprob(theta,Rp,esd,err,z):
  lp = lnprior(theta)
  if not np.isfinite(lp):
        return -np.inf
  return lp+lnlike(theta,Rp,esd,err,z)

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
   data     = np.loadtxt('shear_umsjtu_richd_3',unpack=True) 
   
   Rp       = data[0,:]
   esd      = data[1,:]
   err      = data[2,:]

   if o.MODE=='HARD':
      results  = matterpower(redshift,cosmology)
      kh       = results["kh"]
      lpower   = results["nonpw"]
      rr       = np.linspace(0.001,100,500)
      xi       = ps2xi(kh,lpower[1,:],rr)
      #onehesd  = nfwesd(14.0,4,0.1,Rp)
      for i in range(len(rr)):
	      print rr[i],xi[i]
      #twohesd  = twohaloesd(Mh,rr,xi,Rp)

   if o.MODE=='EASY':
      data= np.loadtxt('corr_linear',unpack=True,skiprows=2)
      RR   = data[0,:]
      ESDR = data[1,:]
      ESDRR= data[2,:]
      logM = 15.0
      con  = 4.0
      zl   = 0.07
      pars = np.array([logM,con])

      ndim,nwalkers = 2,200
      pos = [pars+1e-4*np.random.randn(ndim) for i in range(nwalkers)]
      sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(Rp,esd,err,zl))
	        sampler.run_mcmc(pos,3000)

      burnin = 100
      samples=sampler.chain[:,burnin:,:].reshape((-1,ndim))
      Mh,cn= map(lambda v: (v[1],v[2]-v[1],v[1]-v[0]),zip(*np.percentile(samples,[16,50,84],axis=0)))
      fig = corner.corner(samples,labels=["logM","c"],\
          truths=[Mh[0],cn[0]])
      plt.show()
      plt.savefig('mcmc_8.eps')

      #finterp3 = interp1d(RR,ESDR)
      #finterp4 = interp1d(RR,ESDRR)
      #esdfunc3 = np.array(finterp3(Rp[3:5]))
      #esdfunc4 = np.array(finterp4(Rp[3:5]))
      #onehesd  = nfwesd(15.5,5,0.1,Rp)
      #bias     = galaxybias(15.5)
      #twohesd  = np.zeros(len(esd))
      #twohesd[3:5] = omega_m*h*(esdfunc4-esdfunc3)
#   print bias           
#   plt.errorbar(Rp,esd,yerr=err,fmt='k.',ms=10,elinewidth=3)
#   plt.plot(Rp,onehesd,'k-.',linewidth=3)
#   plt.plot(Rp,bias*twohesd,'k--',linewidth=3)
#   plt.plot(Rp,onehesd+bias*twohesd,'k-',linewidth=3)
#   plt.xlabel('Rp')
#   plt.ylabel('esd')
#   plt.ylim(5.,1000.)
#   plt.xlim(0.05,4.)
#   plt.yscale('log')
#   plt.xscale('log')
#   plt.show()

if __name__=='__main__':
   main()

#---END of every thing--------------
