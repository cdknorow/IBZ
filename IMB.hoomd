#! /usr/bin/env hoomd
from hoomd_script import *
import os
import numpy as np
import math
import scipy.interpolate as interpolate
directory = ""

#Simulation parameters
N=1000
run_time = 1e5
T=1.0
n=11.5
rho = 1e4/(19.740231**3)
print rho
L=(N/rho)**(1./3.)
print L
L = [L,L,L]

##
#Define the potential
def U_power(r, rmin, rmax, n,  epsilon, sigma):
  V = epsilon * ( (sigma / r)**n);
  F = epsilon / r * ( n * (sigma / r)**n);
  return (V, F)

V = []
gr_target = []
x_target =[]
x_edges = []
fid = open('hist_target.inp','r')
for line in fid.readlines():
    s = line.split()
    x_target.append(float(s[0]))
    gr_target.append(float(s[1]))
    V.append(U_power(x_target[-1],1,1,n,1,1)[0])
fid.close()
fid = open('hist_edges.inp','r')
for line in fid.readlines():
    s = line.split()
    x_edges.append(float(s[0]))
fid.close()
#fid = open('pmf.inp','r')
#fid.readline()
#for line in fid.readlines():
#    s = line.split()
#    V.append(float(s[1]))
#fid.close()
delta = 12
iterate_number= 50
rmin = x_edges[0]
rmax = x_edges[-1]
L_start = 15
alpha = np.linspace(.1,0,len(x_target)) 


system = init.create_random(N=N, box=data.boxdim(L=L_start), min_dist = .91)
table = pair.table(width=len(x_target))
table.pair_coeff.set('A', 'A', func=U_power, rmin=rmin, rmax=rmax, coeff=dict(n=n, epsilon=1.0, sigma=1.0))
#table.set_from_file('A','A',filename='pmf.inp')

fire=integrate.mode_minimize_fire( group=group.all(), dt=0.005, ftol=1e-2, Etol=1e-7)
while not(fire.has_converged()):
	run(100)
del fire


integrate.mode_standard(dt=0.005)
bd = integrate.nvt(group=group.all(), T=1.0,tau=0.65)

box_resize = update.box_resize(L = variant.linear_interp([(0,L_start),(4e3,L[0])]),period=100) 
run(5e3)
box_resize.disable()

# dump a .mol2 file for the structure information
mol2 = dump.mol2()
mol2.write(filename=directory+'dna.mol2')
zeroer= update.zero_momentum(period=run_time/2)
logger = analyze.log(filename=directory+'mylog.log', period=2000, quantities=['temperature','potential_energy','kinetic_energy','volume','pair_table_energy','pressure'])
pressure = analyze.log(filename=directory+'pressure.log', period=2000, quantities=['pressure',
			'pressure_xx','pressure_yy','pressure_zz','pressure_xy',
			'pressure_yz','pressure_xz'])					


# dump a .dcd file for the trajectory
dcd = dump.dcd(filename=directory+'dna.dcd', period=(run_time))
#xml = dump.xml(filename=directory+"atoms.dump", period=run_time)
#xml.set_params(all=True)

sys.path.append('/home/cdknorow/Dropbox/Software/')
import MD
from MD.analysis.particle_distance import particle_distance

def F_fit(gr_target, gr_new):
    num = 0
    den = 0
    for i in range(len(gr_new)):
        num += abs(gr_target[i] - gr_new[i])
        den += abs(gr_target[i] + gr_new[i])
    fid = open('f_fit.dat','a')
    fid.write('%f\n'%(1-num/den))
    fid.close()

def Smooth(V):
	V_new = [V[0]]
	for i in range(1,len(V)-1):
		V_new.append((V[i-1]+V[i]+V[i+1])/3.) 
	V_new.append(V[-1])
	return V_new

#Find the radial distribution Function
def find_gr(M,L,x_target,x_edges,delta=5):
	#finds end to end distances and plots a histogram with that data
    	print "#############    FINDING GR ##############"
	distance=particle_distance(M,M,L,rmax=L[0]/2)
	bin_set = np.linspace(.8,2.2,75)
	hist_s, xs = np.histogram(distance,bins=bin_set)
	r_new = np.zeros(xs.shape[0]-1)
	for i in range(r_new.shape[0]):
		r_new[i] = (xs[i]+xs[i+1])/2
	hist = np.zeros(hist_s.shape)
	delta_r = xs[1] - xs[0]
	#normalize the function with respect to an ideal gas
	for i in range(hist_s.shape[0]):
		r = r_new[i]
		hist[i] = hist_s[i] / ((delta*M.shape[1])*(4*math.pi*delta_r*r**2*(M.shape[1]/L[0]**3)))
	#interpolate g(r) for very smooth plot
	fhist = interpolate.interp1d(r_new, Smooth(hist))
	shist = []
	for i in x_target:
		try:
			shist.append(fhist(i))
		except:
			try:
				shist.append(fhist(i+.0001))
			except:
				shist.append(fhist(i-.0001))
	return x_target,shist

##############################
## ITERATIVE BOLTZMAN 
##############################
Trescale = update.rescale_temp(period=5000,T=1.0)
zeroer= update.zero_momentum(period=5000)
for i in range(iterate_number):
	#Equilibrate System
	integrate.mode_standard(dt=0.0005)
	Trescale.enable()
	zeroer.enable()
	run(1e5)
	zeroer.disable()
	Trescale.disable()
	integrate.mode_standard(dt=0.005)
	run(1e5)
	#iterate over the system to get the g(r) 
	M = np.zeros((delta,N,3)) 
	for k in range(delta):
		run(run_time)
		for j in range(N):
			M[k][j] = system.particles[j].position
	#find the new g(r)
	x_new, gr_new = find_gr(M,L,x_target,x_edges,delta=delta)
	fid = open('histogram/hist%i.dat'%i,'w')
	for j in range(len(x_target)):
		fid.write('%f %f\n'%(x_target[j],gr_new[j]))
	fid.close()
	
	#test the fit
	F_fit(gr_target,gr_new)
	#add a boltzman to the potential to get the next potential
	V_new = []
	for j in range(len(x_target)):
		try:
			boltz = .2*T*math.log(gr_new[j]/gr_target[j])
			if math.isnan(boltz):
				V_new.append(V[j])
			elif math.isinf(boltz):
				V_new.append(V[j])
			else:
				V_new.append(V[j] + alpha[j]*T*math.log(gr_new[j]/gr_target[j]))
		except:
			print 'error',gr_new[j],gr_target[j]
			V_new.append(V[j])
	#Smooth Out V
	V = Smooth(V_new)
	F_new = -np.gradient(V,x_target[1]-x_target[0])
	#write new potential table
	pot = open('potential/potential%i.dat'%i,'w')
	pot.write('#r V F\n')
	for j in range(len(x_target)):
		pot.write('%f %f %f\n'%(x_target[j],V[j],F_new[j]))
	pot.close()
	table.set_from_file('A','A',filename='potential/potential%i.dat'%i)

