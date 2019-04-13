from __future__ import division
from gurobipy import *
import sys, os
import numpy, math
import itertools
import copy
numpy.random.seed(1)
import subprocess, multiprocessing, time, gc, json
from scipy import stats

class Node():
    def __init__(self, id=None, t=None, P=None, M=None, V=None, X=[]):
        '''
        P and B uniquely define a state
        '''
        self.id = id
        self.t = t
        self.P = P
        self.M = M
        self.V = V
        self.X = X
        self.prob_out = {}
        self.prob_in = {}
        self.covered = False
        self.mpp = {} # matches per period that lead to that node

def ComputeDistDeathMatches(m,T,prob):
    prob_pmd = {0:{0:1}}
    for nm in range(1,m*T+1):
        prob_pmd[nm] = {}
        dbinom = stats.binom(nm,prob)
        for i in range(nm+1):
            prob_pmd[nm][i] = dbinom.pmf(i)
    return prob_pmd

def ComputeProbabilities(qualities,assortments,t,theta,num_matches,potentials,CV,distr_death_matches,df):
    '''
    Computes probability that k profiles are liked if assortment S is offered with M current matches
    '''
    P, U, EV = {},{},{}
    for s in range(len(assortments)):
        S = list(assortments[s])
        U[s] = {}
        P[s] = {}
        # compute utilities for each number of previous matches
        for j in range(len(S)):
            U[s][j] = qualities[S[j]] + theta[0]*sum([qualities[S[j]]-qualities[S[k]] for k in range(len(S))]) + theta[1]*(theta[2]-num_matches)
        # compute prob of k matches given utilities
        P[s][0] = 1-distr.cdf(U[s][0])
        P[s][len(S)] = distr.cdf(U[s][len(S)-1])
        for k in range(len(S)-1): # sum(S) = m
            P[s][k+1] =  distr.cdf(U[s][k])-distr.cdf(U[s][k+1])

        # compute expected value considering new matches and death matches
        P_next = tuple(sorted(set(potentials).difference(set(S))))
        EV[s] = 0
        for nm in range(len(S)+1): # new matches
            for dm in distr_death_matches[num_matches]:
                EV[s] += (nm + df*CV[P_next,num_matches+nm-dm]) * P[s][nm] * distr_death_matches[num_matches][dm]

    return U,P,EV

def SolveBackwards(n,m,T,theta,dmd,df=1):
    P = range(n)
    V,X = {t+1:{} for t in range(T+1)},{t+1:{} for t in range(T+1)}
    nodes = {}
    nid = 0
    for t in reversed(range(1,T+2)):
        print "Solving period ", t
        nodes[t] = {}
        if t > T:
            Ps = list(itertools.combinations(P, n-m*(t-1)))
            for sP in Ps:
                for nm in range(m*(t-1)+1):
                    V[t][sP,nm] = 0
                    nodes[t][sP,nm] = Node(nid,t,sP,nm,0)
                    nid+=1
        else:
            Ps = list(itertools.combinations(P, n-m*(t-1)))
            for sP in Ps:
                for nm in range(m*(t-1)+1):
                    # for sP, nm, we choose optimal assortment
                    As = list(itertools.combinations(sP, m))
                    Us, Pr, EV = ComputeProbabilities(Qs,As,t,theta,nm,sP,V[t+1],dmd,df)
                    V[t][sP,nm], key = BuildModel(EV)
                    X[t][sP,nm] = As[int(key)]
                    nodes[t][sP,nm] = Node(nid,t,sP,nm,V[t][sP,nm],X[t][sP,nm])
                    nid+=1
                    sPn = tuple(sorted(set(sP).difference(set(X[t][sP,nm]))))
                    prob_out = {}
                    for k in range(len(X[t][sP,nm])+1):
                        prob_out[sPn,nm+k] = Pr[int(key)][k]
                        nodes[t+1][sPn,nm+k].prob_in[sP,nm] = Pr[int(key)][k]
                    nodes[t][sP,nm].prob_out = prob_out

    return V,X,nodes

def BuildModel(EV):
    # init model
    nS = len(EV)
    model = Model('mip1')
    model.setParam('OutputFlag', 0)
    x = {}
    for s in range(nS):
        x[s] = model.addVar(vtype=GRB.CONTINUOUS, name='x'+str(s))

    # add constraints
    model.addConstr(sum([x[s] for s in range(nS)]) == 1, "OAPP")

    # Set objective
    model.setObjective(sum([x[s]*EV[s] for s in range(nS)]) \
                   , GRB.MAXIMIZE)
    model.optimize()
    obj = model.objVal
    key = -1
    for v in model.getVars():
        if v.x == 1:
            key = v.varName[1:]
            break
    return obj, key

def WriteSolution(V,X,nodes,outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir,'all_nodes.csv'), 'w') as f:
        f.write("Period,Potentials,Matches,Assortment,Value\n")
        for t in sorted(X):
            for sP, nm in X[t]:
                f.write(str(t) + ',' + ';'.join([str(sP[i]) for i in range(len(sP))]) \
                        + ',' + str(nm) \
                        + ',' + ';'.join([str(X[t][sP,nm][i]) for i in range(len(X[t][sP,nm]))])\
                        + ',' + str(V[t][sP,nm]) + '\n')
    with open(os.path.join(outdir,'nodes_in_opt_path.csv'), 'w') as f:
        f.write("Period,Potentials,Matches,Assortment,Value\n")
        aux = [nodes[1][key] for key in nodes[1]]
        while len(aux) > 0:
            nd = aux.pop()
            nd.covered = True
            f.write(str(nd.t) + ',' + ';'.join([str(nd.P[i]) for i in range(len(nd.P))]) \
                    + ',' + str(nd.M) \
                    + ',' + ';'.join([str(nd.X[i]) for i in range(len(nd.X))])\
                    + ',' + str(nd.V) + '\n')
            aux.extend([nodes[nd.t+1][sP,nm] for (sP,nm) in nd.prob_out if nodes[nd.t+1][sP,nm].covered == False])


if __name__ == '__main__':
    # Parameters
    n,m,T,df,pmd = 8,3,2,1,0 # last is probability match dies
    theta = [0.1, 0.1, 2]
    distr = stats.norm()
    dmd = ComputeDistDeathMatches(m,T,pmd) # distribution number of matches dead
    # qualities
    Qs = sorted(list(distr.rvs(size=n)), reverse=True)

    # solve
    V,X,nodes = SolveBackwards(n,m,T,theta,dmd,df)
    WriteSolution(V,X,nodes,'outputs')
    sys.exit(1)























#
