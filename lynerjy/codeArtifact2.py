# NK model generation adapted from Maciej Workiewicz (2014) https://github.com/Mac13kW/NK_model

import numpy as np
import itertools
import random
import copy
from time import time
from apgl.graph import *
from apgl.generator.SmallWorldGenerator import SmallWorldGenerator

# Network parameters
people = 100
k = 2 # starting number of people you're connected to
p = 0.1 # rewiring probability

# NK space parameters. N=15 and K=5 takes about 16 seconds to make.
N = 12
K = 3
landscapes = 20  # number of landscapes to produce (use 100 or less for testing)

# Simulation parameters
simtime = 40 # number of timesteps to run sim for
wt = 3 # how many time steps a node will wait before jumping to a higher value
v = 0.02 # how much value is added per person on the same string
pcap = 3 # number of people who can be on an idea before the value maxes out


# interaction matrix (epistasis)
def matrix_rand(N, K):
    Int_matrix_rand = np.zeros((N, N))
    for aa1 in np.arange(N):
        Indexes_1 = range(N)
        Indexes_1.remove(aa1)
        np.random.shuffle(Indexes_1)
        Indexes_1.append(aa1)
        Chosen_ones = Indexes_1[-(K+1):]  # this takes the last K+1 indexes
        for aa2 in Chosen_ones:
            Int_matrix_rand[aa1, aa2] = 1
    return(Int_matrix_rand)

def powerkey(N):
    Power_key = np.power(2, np.arange(N - 1, -1, -1))
    return(Power_key)

def nkland(N):
    NK_land = np.random.rand(2**N, N)
    return(NK_land)

def calc_fit(N, NK_land, inter_m, Current_position, Power_key):
    Fit_vector = np.zeros(N)
    for ad1 in np.arange(N):
        Fit_vector[ad1] = NK_land[np.sum(Current_position * inter_m[ad1]
                                         * Power_key), ad1]
    return(Fit_vector)

def comb_and_values(N, NK_land, Power_key, inter_m):
    """
    Calculates values for all combinations on the landscape.
    - the first N columns are for the combinations of N decision variables DV
    - the second N columns are for the contribution values of each DV
    - the next valuer is for the total fit (avg of N contributions)
    - the last one is to find out whether it is the local peak (0 or 1)
    """
    Comb_and_value = np.zeros((2**N, N*2+2))
    c1 = 0  # starting counter for location
    for c2 in itertools.product(range(2), repeat=N):
        '''
        this takes time so carefull
        '''
        Combination1 = np.array(c2)  # taking each combination
        fit_1 = calc_fit(N, NK_land, inter_m, Combination1, Power_key)
        Comb_and_value[c1, :N] = Combination1  # combination and values
        Comb_and_value[c1, N:2*N] = fit_1
        Comb_and_value[c1, 2*N] = np.mean(fit_1)
        c1 = c1 + 1
    for c3 in np.arange(2**N):  # now let's see if that is a local peak
        loc_p = 1  # assume it is
        for c4 in np.arange(N):  # check for the neighbourhood
            new_comb = Comb_and_value[c3, :N].copy()
            new_comb[c4] = abs(new_comb[c4] - 1)
            if ((Comb_and_value[c3, 2*N] <
                 Comb_and_value[np.sum(new_comb*Power_key), 2*N])):
                loc_p = 0  # if smaller than the neighbour then not peak
        Comb_and_value[c3, 2*N+1] = loc_p
    return(Comb_and_value)



# Generate smallworld graph
graph = SparseGraph(VertexList(people, 1))
generator = SmallWorldGenerator(p, k)
graph = generator.generate(graph)
am = graph.adjacencyMatrix()
print(graph.degreeSequence())
# print(am)


# Generate NK Space
Power_key = powerkey(N)
NK = np.zeros((landscapes, 2**N, N*2+2))
start = time()
for i_1 in np.arange(landscapes):
    NK_land = nkland(N)
    Int_matrix = matrix_rand(N, K)
    NK[i_1] = comb_and_values(N, NK_land, Power_key, Int_matrix)
NKend = time()
NKb = copy.copy(NK) #copy to preserve the original values
print(' time to make NK: ' + str("%.2f" % (NKend-start)) + ' sec')


# Initialize eventual value arrays
avevals = np.zeros((simtime,landscapes))
avefundvals = np.zeros((simtime,landscapes))


# Start simulation
for land in np.arange(landscapes):

    # Initialize arrays
    belief = np.zeros((people,N),dtype=np.int)
    updated = np.zeros((people,N),dtype=np.int)
    vals = np.zeros((simtime,people))
    fundvals = np.zeros((simtime,people)) # fundamental potential, indpt of exploitation
    timespent = np.zeros((people,2))
    timespent[:,0] = np.arange(people) # matrix to store timespent with the same belief
    timespent[:,1] = wt # so that everyone will start moving at first time step

    # Assign random starting beliefs
    for i in np.arange(people):
        belief[i] = np.random.randint(2, size=N)
    
    # Update beliefs at each timestep
    for t in np.arange(simtime):
        allstrings = []
        # 1. one round first just to collate the number of people with common strings
        for i in np.arange(people):
            # find own value from NK map
            ownstring = map(str,belief[i])
            allstrings.append(ownstring)
        # 2. another round to update the true values of NK, depending on the number of people on the same strings
        for i in np.arange(people):
            ownstring = map(str,belief[i])
            ownind = int((''.join(ownstring)),2)
            baseval = copy.copy(NKb[land][ownind][2*N])
            numfriends = allstrings.count(ownstring)-1
            if numfriends>pcap:
                numfriends = copy.copy(pcap)
            ownval = baseval + v*numfriends
            NK[land][ownind][2*N] = ownval
        # 3. next round to influenc eone another
        for i in np.arange(people):
            # find own value from NK map
            ownstring = map(str,belief[i])
            ownind = int((''.join(ownstring)),2)
            refind = copy.copy(ownind)
            # Look around only if you've waited long enough
            if timespent[i,1]>=wt:
                # See what neighbours have
                neighbours = graph.neighbours(i)
                refval = copy.copy(ownval)
                for n in neighbours:
                    nstring = map(str,belief[n])
                    nind = int((''.join(nstring)),2)
                    nval = copy.copy(NK[land][nind][2*N])
                    # if there is a higher neighbour, take its value
                    if nval>refval:
                        refval = copy.copy(nval)
                        # print('refval:'+str(refval))
                        refind = copy.copy(nind)
                        nn = copy.copy(n)
                        timespent[i,1]= -1 # restart waiting time clock
            # Update values
            updated[i] = copy.copy(NK[land][refind][0:N])
            NK[land][refind][2*N] = refval
            vals[t,i] = copy.copy(refval)
            fundvals[t,i] = copy.copy(NKb[land][refind][2*N])
            timespent[i,1]= timespent[i,1]+1
        # find average total value of strings at that timepoint
        avevals[t,land]=np.mean(vals[t,:])
        # find average fundamental poitential of strings at that timepoint
        avefundvals[t,land]=np.mean(fundvals[t,:])
        belief = copy.copy(updated)
        print(str(avevals[t,land]),str(avefundvals[t,land]))

# save file
np.savetxt('lingersimresults/avevals-p_'+str(p)+'-v_'+str(v)+'-wt_'+str(wt)+'.csv', avevals, delimiter=",")
np.savetxt('lingersimresults/avefundvals-p_'+str(p)+'-v_'+str(v)+'-wt_'+str(wt)+'.csv', avefundvals, delimiter=",")

        




