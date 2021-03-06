#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function,division
import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner
from os.path import expanduser
import sys
from scipy import stats

import pandas as pd
#import gatspy.periodic as gp
from scipy.optimize import fsolve
from scipy import optimize
from tqdm import tqdm
import os
from scipy import constants
import celerite
from celerite import terms
from celerite.modeling import Model, ConstantModel
import gatspy.periodic as gats


G=constants.G
home=expanduser('~')
def b_est(numax,teff=None):
	if teff==None:
		teff=4800#guestimate

	scale=0.13*np.sqrt(teff/5777)*(1/40)#convert to rv (gran)
	srv=(1/400)*(teff/5777)

	a1=scale*3382*numax**-0.609
	a2=scale*3382*numax**-0.609
	b1=0.317*(numax**0.97)
	b2=0.948*(numax**0.992)
	height=srv*2.03e7*(numax**-2.38)
	return a1,a2,b1,b2,height

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def fold(time, period, origo=0.0, shift=0.0):
	return ((time - origo)/period + shift) % 1.

def rebin(x, r):
    m = len(x) // r
    return x[:m*r].reshape((m,r)).mean(axis=1),r


def lomb(time,flux,Nf=15):
	if type(time) is np.ndarray:
		pass
	else:
		time=time.values
		flux=flux.values
	time=time-time[0]
	time=time*86400#time in seconds

	c=[]
	for i in range(len(time)-1):
		c.append(time[i+1]-time[i])
	c=np.median(c)
	nyq=1/(2*(time[1]-time[0]))
	nyq=1/(2*c)
	df=1/time[-1]

	f,p=gats.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=0,df=df,Nf=Nf*(nyq/df))
	t=1/f#in sec
	return t/86400,p#in days

def ps(time,flux,Nf=1):
	time=time-time[0]
	#if time[1]<1:
	time=time*86400

	c=[]
	for i in range(len(time)-1):
		c.append(time[i+1]-time[i])
	c=np.median(c)
	#print(c)
	nyq=1/(2*(time[1]-time[0]))
	nyq=1/(2*c)
	#print(nyq*1e6)
	df=1/time[-1]

	f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=0,df=df,Nf=Nf*(nyq/df))
	lhs=(1/len(time))*np.sum(flux**2)
	rhs= np.sum(p)
	ratio=lhs/rhs
	p*=ratio/(df*1e6)#ppm^2/uHz
	f*=1e6
	return f,p
###################################
###################################
###################################
def tru_anom(params,time):
	rvsys, K, w, ecc, Tr, P = params
	w=np.radians(w)
	n=(2*np.pi)/P
	M=n*(time-Tr)#if e==0 then E==M
	E=np.zeros(len(M))

	if len(time)<150:
		E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230
	else:
		for ii,element in enumerate(M): # compute eccentric anomaly
			E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)

	#f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))
	f=2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(0.5*E))

	return f

def rv_pl(time,params):
	rvsys, K, w, ecc, Tr, P=params
	w=np.radians(w)
	if w<0:
		w=w+(2*np.pi)
	if ecc<0:
		ecc=np.abs(ecc)

	if ecc==0:
		w=np.radians(w)
		n=(2*np.pi)/P
		M=n*(time-Tr)
		E=np.zeros(len(M))
		V=rvsys+K*(np.cos(w+M))
	else:
		f=tru_anom(params,time)
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V

class RVModel(Model):
	parameter_names = ('vfiber', 'vslit', 'K', 'w', 'e', 'Tr', 'period')

	def get_value(self, t):
		params=[self.vfiber,self.K,self.w,self.e,self.Tr,self.period]
		model1=rv_pl(RV['time'][RV['how']=='fiber'],params)

		params=[self.vslit,self.K,self.w,self.e,self.Tr,self.period]
		model2=rv_pl(RV['time'][RV['how']=='slit'],params)


		RV['model'][RV['how']=='fiber']=model1
		RV['model'][RV['how']=='slit']=model2

		return RV['model']



muhzconv = 1e6 / (3600*24)
def muhz2idays(muhz):
    return muhz / muhzconv
def muhz2omega(muhz):
    return muhz2idays(muhz) * 2.0 * np.pi
def idays2muhz(idays):
    return idays * muhzconv
def omega2muhz(omega):
    return idays2muhz(omega / (2.0 * np.pi))

#print(np.log(muhz2omega(3)), np.log(muhz2omega(50)))

###########################################################
loc='./CELERITE_RV_two_dataset'

if not os.path.isdir(loc):
	os.makedirs(loc)

#READ IN AND LOMB##
i='./Data/24Boo_GammLibra/RV_24boo.dat'
RV=pd.read_csv(i,delim_whitespace=True,header=None,names=['time','rv','erv','how'])#time=JD-2450000
RV['model']=0 #will need later

slit=RV[RV['how']=='slit']
fiber=RV[RV['how']=='fiber']


periods,power=lomb(RV['time'],RV['rv'],Nf=10)
periods2,power2=lomb(fiber['time'],fiber['rv'],Nf=10)
# plot the results
plt.plot(periods, power)
plt.plot(periods2, power2)
plt.show()
print('guess p: ', periods[power==max(power)])
print('guess v: ',np.median(slit['rv']))
print('guess K: ',np.std(slit['rv']-np.median(slit['rv'])))

print('guess v: ',np.median(fiber['rv']))
print('guess K: ',np.std(fiber['rv']-np.median(fiber['rv'])))
#print(len(slit))67
#print(len(fiber))82


#set the GP parameters-FROM SAM RAW
#First granulation
Q = 1.0 / np.sqrt(2.0)
w0 = muhz2omega(20)
S0 = np.var(fiber['rv']) / (w0*Q)

kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                       bounds=[(-20, 20), (-15, 15), (np.log(muhz2omega(0.1)), np.log(muhz2omega(100)))]) #omega upper bound: 100 muhz
kernel.freeze_parameter("log_Q") #to make it a Harvey model


#numax
Q = np.exp(3.0)
w0 = muhz2omega(25) #peak of oscillations
S0 = np.var(fiber['rv']) / (w0*Q)

kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                      bounds=[(-20, 20), (0.1, 5), (np.log(muhz2omega(5)), np.log(muhz2omega(50)))])

kernel += terms.JitterTerm(log_sigma=1, bounds=[(-20,40)])


#initial guess of RV model
initial = RVModel(vfiber=0, vslit=0,K=50,w=90,e=0.01,Tr=5613.2744,period=30.37, bounds=[(-20,20), (-20,20), (0,100), (0,360), (0,0.99), (5600,5700), (28,32)])

#	parameter_names = ("vfiber, vslit", "K", "w", "e", "Tr","P")
#time,rv,erv=fiber.as_matrix(['time','rv','erv']).T.astype(float)

gp = celerite.GP(kernel, mean=initial, fit_mean=True)
gp.compute(RV['time'], RV['erv'])


#labels=gp.get_parameter_names()
labels=['logS01', 'logomega01', 'logS0osc', 'logQosc', 'logomega0osc', 'logsigma', "vfiber", "vslit", "K", "w", "e", "Tr","P"]

samples=np.load(loc+'/samples.npy')#Save out samples for other code

#fig = corner.corner(samples, labels=labels)
#fig.savefig(loc+"/corner.png")#still in log space parameters
#plt.show()

quantiles = np.percentile(samples,[16,50,84],axis=0).T
logmedians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-

#vfiber, vslit, K, w, ecc, T, period=logmedians[-7:]
#labels=['logS01', 'logomega01', 'logS0osc', 'logQosc', 'logomega0osc', 'logsigma', "vfiber", "vslit", "K", "w", "e", "Tr","P"]
for i in range(0,len(labels)):
	print(labels[i],logmedians[i],'+',uerr[i],'-',lerr[i])

samples[:,:-8]=np.exp(samples[:,:-8])
samples[:,1]=omega2muhz(samples[:,1])#convert omega1 to uHz
samples[:,4]=omega2muhz(samples[:,4])#convert omega osc to uHz

labels2=labels
labels2[:5]=map(lambda x: x[3:],labels[:5])

labels2[1]='f1'
labels2[4]='fosc'
labels2[5]=r'ln$\sigma$'


fig = corner.corner(samples, labels=labels2)
fig.savefig(loc+"/no_log_tran_corner.png")
plt.show()

quantiles = np.percentile(samples,[16,50,84],axis=0).T

medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-
for i in range(0,len(labels2)):
	print(labels2[i],medians[i],'+',uerr[i], '-', lerr[i])


###############################################################
print('estimate from scaling relations')
l=['a1','a2','b1','b2','H']
print('numax:',medians[4])
sc_r=b_est(medians[4],teff=4893)
for i in range(0,len(l)):
	print(l[i],sc_r[i])



###############################################################
###############################################################
###############################################################
gp.set_parameter_vector(logmedians)
gp.compute(RV['time'], RV['erv'])

vfiber, vslit, K, w, ecc, T, period=logmedians[-7:]
RV['final'] = RVModel(vfiber, vslit, K, w, ecc, T, period).get_value(RV)#final rv model
RV['gp']=gp.predict(RV['erv'],RV['time'],return_cov=False)-RV['final']#final gp+rv - rv

plt.subplot(211)
plt.errorbar(RV['time'],RV['rv'],RV['erv'],fmt='.')
plt.plot(RV['time'],RV['final'])
plt.plot(RV['time'],RV['gp'])

###############################################################
###############################################################
RV['rv'][RV['how']=='slit']=RV['rv'][RV['how']=='slit']-vslit
RV['rv'][RV['how']=='fiber']=RV['rv'][RV['how']=='fiber']-vfiber
finalparams=[0,K, w, ecc, T, period]
mod_time=np.arange(np.min(RV['time']),np.max(RV['time']),1)
finalmod = rv_pl(mod_time,finalparams)

phaserv=fold(RV['time'],period,T)
phase=fold(mod_time,period,T)
idx=np.argsort(phase)

plt.subplot(212)
plt.errorbar(phaserv,RV['rv'],RV['erv'],fmt='.')
plt.plot(phase[idx],finalmod[idx],'r--')
plt.show()
