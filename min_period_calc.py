from __future__ import print_function,division
import numpy as np
import sys
import matplotlib.pyplot as plt
from os.path import expanduser
import pandas as pd

from astropy import constants as acon

Rsol=acon.R_sun.value
Msol=acon.M_sun.value


def numax(logg,teff):
	g=(10**logg)/100.
	g=g/274
	nu=3150*g*(teff/5777.)**-0.5
	return nu


def minP(limMass=1,R=1):
	print(limMass)
	limlogg=np.arange(0.5,4.5,0.1)
	grav=(10**limlogg)/100
	limR=R*np.sqrt(limMass*2e30*6.67e-11/grav)
	limP2=4*np.pi**2 * limR**3 /(limMass*2e30*6.67e-11)
	limP=np.sqrt(limP2)
	limP=limP/(86400)
	return limlogg,limP

def minPeriod(limMass=1,R=1):
	mass=limMass*Msol
	radius=R*Rsol
	limP2=4*np.pi**2 * radius**3 /(mass*6.67e-11)
	limP=np.sqrt(limP2)
	limP=limP/(86400)
	return limP


home=expanduser('~')

#data=pd.read_csv(home+'/Dropbox/PhD/Python_Codes/conf_planets_all_rows_all_col_Oct16.csv',comment='#')
#data=pd.read_csv(home+'/Dropbox/PhD/Year_4/SONG_all_together_now/Any_others/conf_planets_aug_17.csv',comment='#') 
data=pd.read_csv('../../../all_planets_nov17.csv',comment='#')
data['st_numax']=numax(data['st_logg'],data['st_teff'])
data=data[data['st_logg']<3.6]
print(max(data['st_numax']))

period,logg,method=data['pl_orbper'],data['st_logg'],data['pl_discmethod']
mass=data['st_mass']
host=data['pl_hostname']
#(0.99Ms, 10.64Rs,and logg = 2.42)  30.350days
print(minPeriod(0.99,10.64),'days')

lg,lp=minP(limMass=1.0)
lg2,lp2=minP(limMass=1.0,R=2)
lg3,lp3=minP(limMass=1.0,R=3)

plt.figure()
plt.plot(lp,lg,'k-',label=r'$a=R_{\star}$, $1M_{\odot}$')
plt.plot(lp2,lg2,'b--',label=r'$a=2R_{\star}$, $1M_{\odot}$')
plt.plot(lp3,lg3,'r:',label=r'$a=3R_{\star}$, $1M_{\odot}$')


plt.plot(period[method=='Radial Velocity'],logg[method=='Radial Velocity'],'kx',label='Radial Velocity',markersize=8)
plt.plot(period[(method=='Transit')],logg[(method=='Transit')],'r+',label='Transit',markersize=8)
plt.plot( 30.350,2.42,'b*',label='24 Boo',ms=14)
plt.xlabel('Orbital Period (days)',fontsize=16)
plt.ylabel(r'Stellar $\log_{10}{g}$',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.ylim(3.6,1.0)
plt.xlim(0.3,1e4)
plt.xscale('log')
plt.legend(loc='best',framealpha=1.0,numpoints=1)
plt.tight_layout()
plt.show()




