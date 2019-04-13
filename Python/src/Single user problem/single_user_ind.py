from __future__ import division
from gurobipy import *
import sys, os
import numpy, math
import itertools
import copy
numpy.random.seed(1)
import subprocess, multiprocessing, time, gc, json
from scipy import stats

def ComputeProbabilities(qualities,assortments,t,theta,num_matches,potentials,CV):
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
        # try:
        P_next = tuple(sorted(set(potentials).difference(set(S))))
        EV[s] = P[s][0]*CV[P_next,num_matches] + P[s][len(S)]*CV[P_next,num_matches+len(S)]
        # except:
        #     print potentials, S
        #     print sorted(set(potentials).difference(set(S)))
        #     sys.exit(1)
        for k in range(len(S)-1): # sum(S) = m
            P[s][k+1] =  distr.cdf(U[s][k])-distr.cdf(U[s][k+1])
            EV[s]+=P[s][k+1]*(k+1 + CV[P_next,num_matches+k+1])
    return U,P,EV

def SolveBackwards_old(n,m,P,T,theta,df=1):
    V,X = {t+1:{} for t in range(T+1)},{t+1:{} for t in range(T+1)}
    for t in reversed(range(1,T+2)):
        print "Solvinf period ", t
        if t > T:
            Ps = list(itertools.combinations(P, n-m*(t-1)))
            for sP in Ps:
                for nm in range(m*(t-1)):
                    V[t][sP,nm] = 0
        else:
            Ps = list(itertools.combinations(P, n-m*(t-1)))
            for sP in Ps:
                for nm in range(m*(t-1)):
                    # for sP, nm, we choose optimal assortment
                    mx_val, mx_S = -1e10, -1
                    As = list(itertools.combinations(sP, m))
                    Us, Pr = ComputeProbabilities(Qs,As,t,theta,nm)
                    for s in range(len(As)):
                        S = As[s]
                        P_next = tuple(set(sP).difference(set(S)))
                        value = sum([(k + df*CV[P_next,nm+k])*Pr[s][k]  for k in range(len(S)+1)])
                        if value > mx_val:
                            mx_val = value
                            mx_S = S
                    V[t][sP, nm], X[t][sP, nm] = mx_val, S
    return V,X

def SolveBackwards(n,m,T,theta,df=1):
    P = range(n)
    V,X = {t+1:{} for t in range(T+1)},{t+1:{} for t in range(T+1)}
    for t in reversed(range(1,T+2)):
        print "Solving period ", t
        if t > T:
            Ps = list(itertools.combinations(P, n-m*(t-1)))
            for sP in Ps:
                for nm in range(m*(t-1)+1):
                    V[t][sP,nm] = 0
        else:
            Ps = list(itertools.combinations(P, n-m*(t-1)))
            for sP in Ps:
                for nm in range(m*(t-1)+1):
                    # for sP, nm, we choose optimal assortment
                    As = list(itertools.combinations(sP, m))
                    Us, Pr, EV = ComputeProbabilities(Qs,As,t,theta,nm,sP,V[t+1])
                    V[t][sP,nm], key = BuildModel(EV)
                    X[t][sP,nm] = As[int(key)]
    return V,X

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

def WriteSolution(V,X,outfile):
    with open(os.path.join(outfile), 'w') as f:
        f.write("Period,Potentials,Matches,Assortment,Value,Path\n")
        for t in sorted(X):
            for sP, nm in X[t]:
                f.write(str(t) + ',' + ';'.join([str(sP[i]) for i in range(len(sP))]) \
                        + ',' + str(nm) \
                        + ',' + ';'.join([str(X[t][sP,nm][i]) for i in range(len(X[t][sP,nm]))])\
                        + ',' + str(V[t][sP,nm]) + '\n')


if __name__ == '__main__':
    # Parameters
    n,m,T,df = 8,3,2,1
    theta = [0.1, 0.1, 2]
    distr = stats.norm()

    # qualities
    Qs = sorted(list(distr.rvs(size=n)), reverse=True)

    # solve
    V,X = SolveBackwards(n,m,T,theta,df=1)
    WriteSolution(V,X,'solution.csv')
    sys.exit(1)

    a = (1,2)
    '-'.join([str(a[i]) for i in range(len(a))])

    #
    # # start from last period
    # V,X = {t+1:{} for t in range(T+1)},{t+1:{} for t in range(T+1)}
    # t = T+1
    # Ps = list(itertools.combinations(P, n-m*(t-1)))
    # for sP in Ps:
    #     for nm in range(m*(t-1)):
    #         V[t][sP,nm] = 0
    #
    # # previous period
    # t = T
    # Ps = list(itertools.combinations(P, n-m*(t-1)))
    # print len(Ps)
    # print len(range(m*(t-1)))
    # for sP in Ps:
    #     print sP
    #     for nm in range(m*(t-1)):
    #         # for sP, nm, we choose optimal assortment
    #         As = list(itertools.combinations(sP, m))
    #         Us, Pr, EV = ComputeProbabilities(Qs,As,t,theta,nm,sP,V[t+1])
    #         V[t][sP,nm], key = BuildModel(EV)
    #         X[t][sP,nm] = As[int(key)]
    #
    #
    # Ps = list(itertools.combinations(P, n))
    # print Ps
    #
    # sys.exit(1)
    #
    #
    # t = 1
    # As = list(itertools.combinations(P, m))
    # Us, Pr = ComputeProbabilities(Qs,As,t,theta)
    # for s in range(len(As)):
    #     S = As[s]
    #     P_next = set(P).difference(set(S))
    #     print tuple(P_next)
    #     for k in range(len(S)+1): # loop over number of matches
    #
    #
    #
    #
    #
    # # compute probabilities
    #
    #
    #
    # n,m,df = 3,5,0.9
    # Q = [0.45, 0.4, 0.3]
    # # distr = stats.norm()
    #
    # # Generate initial inventory of types in backlog
    # T = 5
    # inventory = [m,m*T,m*T]
    # # theta = [0.1,0.1,2] # effect of assortment, effect of history, and target numebr of matches
    # theta = [0.1,0.1,2] # effect of assortment, effect of history, and target numebr of matches
    #
    #
    # Ss = GenerateAssortments(n, m, inventory)
    # print Ss
    #
    # U,P = ComputeProbabilities(Ss, m, 1, theta)
    #
    # V,X = SolveBackwards(n,m,T,theta,inventory, df)
    # print X[1]
    #
    #
    # eis = {0:inventory} # ending inventories
    # for t in range(1, T+1):
    #     eis[t] = {}
    #     if t == 1:
    #         print '------------------------'
    #         print 'Period, Inventory, Num Matches, Decision, Value '
    #         for k in X[t]:
    #             inv, nm = list(k[:3]), k[3]
    #             print t, inv, nm, X[t][k], V[t][k]
    #             eis[t][k] = [inventory[i] - X[t][k][i] for i in range(n)]
    #         print '------------------------'
    #     else:
    #         print '------------------------'
    #         print 'Period, Inventory, Num Matches, Decision, Value '
    #         for k in eis[t-1]:
    #             prev_m = k[3]
    #             for nm in range(m+1): # for number of matches in previous period
    #                 nm_t = nm + prev_m
    #                 idx = copy.copy(eis[t-1][k])
    #                 idx.append(nm_t)
    #                 print t, eis[t-1][k], nm_t, X[t][tuple(idx)], V[t][tuple(idx)]
    #                 eis[t][tuple(idx)] = [eis[t-1][k][i] - X[t][tuple(idx)][i] for i in range(n)]
    #             print '------------------------'
    #
    #
    #
    #
    # # Simulations
    # for sim in range(10):
    #     Q = sorted(list(numpy.random.uniform(0,1,n)), reverse=True)
    #     V,X = SolveBackwards(n,m,T,theta,inventory)
    #
    #     scn_data = {'X':X[1][X[1].keys()[0]], 'V':V[1][X[1].keys()[0]], 'Q':Q}
    #     with open(os.path.join('outputs', 'scenario_'+str(sim)+'.txt'), 'w') as outfile:
    #         json.dump(scn_data, outfile)
    #
    #
    #
    # eis = {0:inventory} # ending inventories
    # for t in range(1, T+1):
    #     eis[t] = {}
    #     if t == 1:
    #         print X[t], V[t]
    #         for k in X[t]:
    #             inv, nm = list(k[:3]), k[3]
    #             eis[t][k] = [inventory[i] - X[t][k][i] for i in range(n)]
    #     else:
    #         for k in eis[t-1]:
    #             prev_m = k[3]
    #             for nm in range(m+1): # for number of matches in previous period
    #                 nm_t = nm + prev_m
    #                 idx = copy.copy(eis[t-1][k])
    #                 idx.append(nm_t)
    #                 print nm_t, X[t][tuple(idx)], V[t][tuple(idx)]
    #                 eis[t][tuple(idx)] = [eis[t-1][k][i] - X[t][tuple(idx)][i] for i in range(n)]
    #
    # norm = stats.norm()
    # q1 = -2.49
    # q2 = -2.5
    # delta = 0.01
    # print 2*norm.cdf(q1)
    # print norm.cdf(q1+delta*(q1-q2)) + norm.cdf(q1-(1+delta)*(q1-q2))
