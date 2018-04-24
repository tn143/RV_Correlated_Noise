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

home=expanduser('~')

def phase(time, period, origo=0.0, shift=0.5):
    return (((time - origo)/period + shift) % 1.)-0.5

#need period, ecc, time, w, rvsys, t0
#tan(f/2)=np.sqrt((1+e)/(1-e))*tan(E/2)
#n=(2*np.pi)/period
#E-(e*np.sin(E))=n(time-T)
#v=rvsys+K*(np.cos(f+w)+e*np.cos(w))

def rv_pl(time,params):
	rvsys, K, w, ecc, T, period=params
	w=np.radians(w)
	n=(2*np.pi)/period
	M=n*(time-T)
	E=np.zeros(len(M))
	if ecc==0:
		#w=0#undefined for circular orbit
		V=rvsys+K*(np.cos(w+M))
	else:
		if len(time)<150:
			E= fsolve(lambda x: x-ecc*np.sin(x) - M,M)#slower for N >~230
		else:
			for ii,element in enumerate(M): # compute eccentric anomaly
				E[ii] = fsolve(lambda x: element- x+ecc*np.sin(x) ,element)

		f=2*np.arctan2(np.sqrt((1+ecc))*np.sin(0.5*E),np.sqrt(1-ecc)*np.cos(0.5*E))
		V=rvsys+K*(np.cos(w+f)+(ecc*np.cos(w)))
	return V

# Define the probability function as likelihood * prior.
def lnprior(theta):
	rvsys, K, w, ecc, Tr, period,sig2 = theta
	logsig2=np.log10(sig2)
	logp=np.log10(period)
	logr=np.log10(rvsys)
	logk=np.log10(K)
	sqrte=np.sqrt(ecc)
	#if -100<rvsys<100 and -2 < logk < 3 and 0.0<w<360 and 30<period<50 and 0<Tr<100:
	#	ln_e=stats.beta(0.867,3.03)
	#	lnp_per=-0.5*(period-42.795)**2/(0.01**2)
	#	return ln_e.logpdf(ecc)+lnp_per
	if -100<rvsys<100 and -2 < logk < 3 and -1<sqrte*np.cos(np.deg2rad(w))<1 and -1<sqrte*np.sin(np.deg2rad(w))<1 and ecc<1\
	and 0<Tr<45 and  6.2<period<6.3 and 0<w<360 and -10<logsig2<10:
		lnp_per=-0.5*(period-6.2467)**2/(0.01**2)
		return lnp_per
	else:
		return -np.inf

def lnlike(theta, time, rv, erv):
    rvsys, K, w, ecc, T0, period,sig2 = theta
    model =rv_pl(time,theta[:-1])

    inv_sigma2 = 1.0/(sig2**2+erv**2)# + model**2)
    return -0.5*(np.sum((rv-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprob(theta, time, rv, erv):
    lp = lnprior(theta)
    if not np.isfinite(lp):
		return -np.inf
    return lp + lnlike(theta, time, rv, erv)

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

#np.random.seed(14392)#meaning of life you know?!
#Synthetic
#Nobs=9
#time=np.random.uniform(0,150,Nobs)
#time=np.sort(time)
#rvsys, K, w, ecc, T, period
#initial=[5,45,100.8956,0.15,13.3345,42.795446754]
#rvsys, K, w, ecc, T, period=initial
#rv=rv_pl(time,initial)#generate rv curve
#erv=2+0.2*np.random.rand(len(time))#random errors ~10ms 
#rv+=erv*np.random.randn(len(time))
#rv+=np.random.normal(0,5,len(rv))#and random scatter of same level jitter term
#plt.plot(mod_time,rv_pl(mod_time,initial))


i='./Data/24Boo_GammLibra/RV_24boo.dat'
#time,rv,erv=#np.loadtxt(i,delimiter='\t',unpack=True,usecols=(0,1,2))

RV=pd.read_csv(i,delim_whitespace=True,header=None,names=['time','rv','erv','type'])
slit=RV[RV['how']=='slit']
fiber=RV[RV['how']=='fiber']


time,rv,erv,_=slit.as_matrix().T
time2,rv2,erv2,_=fiber.as_matrix().T
#time=JD-2450000
##############################
plt.errorbar(time,rv,yerr=erv,fmt='.')
plt.show()

periods,power=lomb(time,rv,Nf=1)

# plot the results
plt.plot(periods, power)
plt.show()

print('guess p: ', periods[power==max(power)])
print('guess sys v: ',np.mean(rv))
print('guess K: ',np.std(rv-np.mean(rv)))

sys.exit()

#rvsys, K, w, ecc, T0, period
labels=['rvsys', 'K', 'w', 'ecc', 'T0', 'period','sig2']
initial=[0,40,90,0.01,15,6.25,20]

#VALUES
print(np.c_[labels,initial])


plt.errorbar(time,rv,erv,fmt='.')
plt.plot(mod_time,rv_pl(mod_time,initial[:-1]))
plt.show()

# Set up the sampler.
ntemps, nwalkers, niter, ndim = 2, 500, 2500, len(labels)
sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlike, lnprior, loglargs=(time, rv, erv))
p0 = np.zeros([ntemps, nwalkers, ndim])
for i in range(ntemps):
    for j in range(nwalkers):
	p0[i,j,:] = initial + 1e-2*np.random.randn(ndim)

print('... burning in ...')
for p, lnprob, lnlike in tqdm(sampler.sample(p0, iterations=niter),total=niter):
	sleep(0.001)

#print(np.shape(p0))
#print(np.shape(p))
# Clear and run the production chain.
sampler.reset()

print('... running sampler ...')
for p, lnprob, lnlike in tqdm(sampler.sample(p, lnprob0=lnprob,lnlike0=lnlike,iterations=niter),total=niter):
	sleep(0.001)

fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(8, 9))

for i in range(0,len(initial)):
	axes[i].plot(sampler.chain[0, :, :, i].T, color="k", alpha=0.4)
	axes[i].yaxis.set_major_locator(MaxNLocator(5))
	axes[i].set_ylabel(labels[i])

fig.tight_layout(h_pad=0.0)
plt.show()
#print('n temps:', ntemps, "log evidence: ", sampler.thermodynamic_integration_log_evidence())

# Make the corner plot.
burnin = (niter/2)
samples = sampler.chain[0, :, burnin:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=labels)
plt.show()

quantiles = np.percentile(samples,[16,50,84],axis=0).T 
medians,uerr,lerr=quantiles[:,1],quantiles[:,2]-quantiles[:,1],quantiles[:,1]-quantiles[:,0]#median,+/-

np.savetxt('RV_results.txt',np.c_[np.array(labels),medians,uerr,lerr],fmt='%s',header='Param,median,uerr,lerr')

for i in range(0,len(labels)):
	print(labels[i],medians[i],'+/-',np.mean((uerr[i],lerr[i])))

plt.errorbar(time,rv,erv,fmt='.')
plt.plot(mod_time,rv_pl(mod_time,medians[:-1]))
plt.show()


