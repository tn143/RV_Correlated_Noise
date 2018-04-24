#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import emcee
import corner
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import corner
from os.path import expanduser
import sys
from scipy.optimize import fsolve
import pandas as pd
import gatspy.periodic as gp
from tqdm import tqdm
from time import sleep
from scipy import stats
import os

home=expanduser('~')

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

	f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=0,df=df,Nf=Nf*(nyq/df))
	t=1/f#in sec
	return t/86400,p#in days


def fold(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def rv_pl(t,params):
	rvsys, K, w, ecc, T, period=params
	w=np.radians(w)
	if w<0:
		w=w+(2*np.pi)
	if ecc<0:
		ecc=np.abs(ecc)

	n=(2*np.pi)/period
	M=n*(t-T)
	E=np.zeros(len(M))
	for ii,element in enumerate(M): # compute eccentric anomaly
		E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)
	f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))

	V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V

# Define the probability function as likelihood * prior.
def lnprior(theta):
	vslit,vfiber, K, w, ecc, T0, period,sigslit,sigfiber = theta
	sqrte=np.sqrt(ecc)
	logp=np.log10(period)
	#logr1=np.log10(vslit)
	#logr2=np.log10(vfiber)

	logk=np.log10(K)

	if -100 < vslit < 100 and -100<vfiber<100 and -2 < logk < 3 and -1<sqrte*np.cos(np.deg2rad(w))<1 \
	and -1<sqrte*np.sin(np.deg2rad(w))<1 and ecc<1 and w<360 and 5600<T0<5700 and 29<period<31 and 0<sigslit<50 and 0<sigfiber<50:
		return 0.0
	else:
		return -np.inf

def lnlike(theta, timefiber, rvfiber, ervfiber,time,rv,erv):
    vslit,vfiber, K, w, ecc, T0, period,sigslit,sigfiber = theta
    #SLIT
    param=[vslit,K, w, ecc, T0, period]
    model =rv_pl(time,param)
    inv_sigma2 = 1.0/(sigslit**2+erv**2)
    loglike1=-0.5*(np.sum((rv-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    #####FIBER
    param=[vfiber,K, w, ecc, T0, period]
    model =rv_pl(timefiber,param)
    inv_sigma2 = 1.0/(sigfiber**2+ervfiber**2)
    loglike2=-0.5*(np.sum((rvfiber-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    return loglike1+loglike2

def lnprob(theta, timefiber, rvfiber, ervfiber,time,rv,erv):
    lp = lnprior(theta)
    if not np.isfinite(lp):
		return -np.inf
    return lp + lnlike(theta, timefiber, rvfiber, ervfiber,time,rv,erv)

loc='./24Boo_slit_fiber'
if not os.path.exists(loc):
    os.makedirs(loc)

#READ IN AND LOMB##
i='./Data/24Boo_GammLibra/RV_24boo.dat'
RV=pd.read_csv(i,delim_whitespace=True,header=None,names=['time','rv','erv','how'])#time=JD-2450000

slit=RV[RV['how']=='slit']
fiber=RV[RV['how']=='fiber']

time,rv,erv=slit.as_matrix(['time','rv','erv']).T.astype(float)
timefiber,rvfiber,ervfiber=fiber.as_matrix(['time','rv','erv']).T.astype(float)

periods,power=lomb(time,rv,Nf=4)
periods2,power2=lomb(timefiber,rvfiber,Nf=4)
# plot the results
plt.plot(periods, power)
plt.plot(periods2, power2)
plt.show()
print('guess p: ', periods[power==max(power)])
print('guess v: ',np.mean(rv))
print('guess K: ',np.std(rv-np.mean(rv)))

print('guess v: ',np.mean(rvfiber))
print('guess K: ',np.std(rvfiber-np.mean(rvfiber)))

###################################################
#vslit,vfiber, K, w, ecc, T0, period
labels=['vslit','vfiber', 'K', 'w', 'ecc', 'T0', 'period','sigslit','sigfiber']
initial=[0,-5.25,50,90,0.01,5613.2744,30.37,5,5]

vslit,vfiber, K, w, ecc, T0, period,sigslit,sigfiber=initial
#slit time,rv,erv-vslit
#fiber timefiber,rvfiber,ervfiber-vfiber
#rv=rv-vslit
#rvfiber=rvfiber-vfiber

mod_time=np.arange(np.min([np.min(time), np.min(timefiber)]), np.max([np.max(time), np.max(timefiber)]),1)
params=np.append(0,initial[2:-2])
final=rv_pl(mod_time,params)

plt.subplot(211)
plt.errorbar(time,rv,np.sqrt(erv**2+sigslit**2),fmt='k.')
plt.errorbar(timefiber,rvfiber,np.sqrt(ervfiber**2+sigfiber**2),fmt='r.')
plt.plot(mod_time,final,'b--')

phaserv=fold(time,period,T0)
phaservfiber=fold(timefiber,period,T0)

phase=fold(mod_time,period,T0)
idx=np.argsort(phase)

plt.subplot(212)
plt.errorbar(phaserv,rv,np.sqrt(erv**2+sigslit**2),fmt='.')
plt.errorbar(phaservfiber,rvfiber,np.sqrt(ervfiber**2+sigfiber**2),fmt='.')

plt.plot(phase[idx],final[idx],'r--')
plt.savefig(loc+'/final.png')
plt.show()






# Set up the sampler.
nwalkers, niter, ndim = 100, 1000, len(labels)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(timefiber, rvfiber, ervfiber,time,rv,erv))

p0 = np.zeros([nwalkers, ndim])

for j in range(nwalkers):
	p0[j,:] = initial + 1e-2*np.random.randn(ndim)

print('... burning in ...')
#for p, lnprob, state in sampler.sample(p0, iterations=niter):
for p, lnprob, state in tqdm(sampler.sample(p0, iterations=int(niter/2)),total=int(niter/2)):
	sleep(0.001)

#Find best walker and resamples around it
p = p[np.argmax(lnprob)]
print(p)
p0 = np.zeros([nwalkers, ndim])

for j in range(nwalkers):
	p0[j,:] = p + 1e-3*np.random.randn(ndim)

# Clear and run the production chain.
#sampler.reset()
print('... running sampler ...')
#for p, lnprob, state in sampler.sample(p, lnprob0=lnprob,iterations=niter):
for p, lnprob, state in tqdm(sampler.sample(p0, lnprob0=lnprob,iterations=niter),total=niter):
	sleep(0.001)

fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8, 9))
burnin=int(niter/2)

for i in range(0,len(initial)):
	axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
fig.savefig(loc+'/chain.png')
plt.show()
#print('n temps:', ntemps, "log evidence: ", sampler.thermodynamic_integration_log_evidence())

# Make the corner plot.
burnin = int(niter/2)
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
np.save(loc+'/24Boo_pdfs.npy',samples)#Save out samples for other code

fig = corner.corner(samples, labels=labels)
fig.savefig(loc+"/corner.png")
plt.close()

quantiles = np.percentile(samples,[16,50,84],axis=0).T 
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-

#np.savetxt(loc+'/RV_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')

for i in range(0,len(labels)):
	print(labels[i],medians[i],'+',uerr[i], '-', lerr[i])


vslit,vfiber, K, w, ecc, T0, period,sigslit,sigfiber=medians
#slit time,rv,erv-vslit
#fiber timefiber,rvfiber,ervfiber-vfiber
rv=rv-vslit
rvfiber=rvfiber-vfiber

mod_time=np.arange(np.min([np.min(time), np.min(timefiber)]), np.max([np.max(time), np.max(timefiber)]),1)
params=np.append(0,medians[2:-2])
final=rv_pl(mod_time,params)

plt.subplot(211)
plt.errorbar(time,rv,np.sqrt(erv**2+sigslit**2),fmt='k.')
plt.errorbar(timefiber,rvfiber,np.sqrt(ervfiber**2+sigfiber**2),fmt='r.')
plt.plot(mod_time,final,'b--')

phaserv=fold(time,period,T0)
phaservfiber=fold(timefiber,period,T0)

phase=fold(mod_time,period,T0)
idx=np.argsort(phase)

plt.subplot(212)
plt.errorbar(phaserv,rv,np.sqrt(erv**2+sigslit**2),fmt='.')
plt.errorbar(phaservfiber,rvfiber,np.sqrt(ervfiber**2+sigfiber**2),fmt='.')

plt.plot(phase[idx],final[idx],'r--')
plt.savefig(loc+'/final.png')
plt.show()













