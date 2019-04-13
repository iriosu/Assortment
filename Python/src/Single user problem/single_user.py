from gurobipy import *
import sys, os
import numpy, math
import itertools
import copy
numpy.random.seed(1)
import subprocess, multiprocessing, time, gc, json
from scipy import stats

def GenerateAssortments(n,m,inv):
    def Generator(n,m,ls,out,inv):
        #print ls, n, m
        if len(ls) == n and sum(ls) == m:
            out.append(copy.copy(ls))
        elif len(ls) < n and sum(ls) <= m:
            #print len(ls), min(m,inv[len(ls)])
            for j in range(min(m,inv[len(ls)])+1):
                ls.append(j)
                Generator(n,m,ls,out,inv)
                ls.pop()
        else:
            pass
    out, ls = [], []
    for i in range(min(m,inv[len(ls)])+1):
        ls.append(i)
        Generator(n,m,ls,out,inv)
        ls.pop()
    return out

def GenerateEndingBacklogs(n,m,inv,t):
    def Generator(n,m,ls,out,inv):
        '''
        n = number of types
        m = length of assortment
        '''
        if len(ls) == n and sum(ls) == sum(inv)-m*(t-1):
            out.append(copy.copy(ls))
        elif len(ls) < n and sum(ls) <= sum(inv)-m*(t-1):
            for j in range(inv[len(ls)]+1):
                ls.append(j)
                Generator(n,m,ls,out,inv)
                ls.pop()
        else:
            pass
    out, ls = [], []
    for i in range(inv[len(ls)]+1):
        ls.append(i)
        Generator(n,m,ls,out,inv)
        ls.pop()
    return out

def ComputeProbabilities(Ss,m,t,theta):
    '''
    Computes probability that k profiles are liked if assortment S is offered with M current matches
    '''
    P, U = {},{}
    for s in range(len(Ss)):
        S = Ss[s]
        QS = []
        for i in range(len(S)):
            QS.extend([Q[i] for j in range(S[i])])
        QS = sorted(QS, reverse=True)
        U[s] = {}
        P[s] = {}
        # compute utilities for each number of previous matches
        for nm in range(m*(T-1)+1):
            U[s][nm], P[s][nm] = {}, {}
            for q in QS:
                if q in U[s][nm]:
                    continue
                else:
                    U[s][nm][q] = q + theta[0]*sum([q-qp for qp in QS]) + theta[1]*(theta[2]-nm)
            # compute prob of k matches given utilities
            P[s][nm][0] = 1-distr.cdf(U[s][nm][QS[0]])
            P[s][nm][len(QS)] = distr.cdf(U[s][nm][QS[-1]])
            for k in range(len(QS)-1): # sum(S) = m
                P[s][nm][k+1] =  distr.cdf(U[s][nm][QS[k]])-distr.cdf(U[s][nm][QS[k+1]])
    return U,P

def SolveBackwards(n,m,T,theta,inventory,df=1):
    V = {t+1:{} for t in range(0,T+1)}
    X = {t+1:{} for t in range(0,T+1)}
    EI = {t+1:GenerateEndingBacklogs(n,m,inventory,t+1) for t in range(0,T+1)}
    for t in reversed(range(1,T+2)):
        print "Solvinf period ", t
        if t > T:
            for inv in EI[t]:
                for nm in range(m*(t-1)+1):
                    idx = copy.copy(inv)
                    idx.append(nm)
                    V[t][tuple(idx)] = 0
        else:
            for inv in EI[t]:
                for nm in range(m*(t-1)+1):
                    # solve problem for each possible state (inv, M)
                    Ss = GenerateAssortments(n, m, inv)
                    U,P = ComputeProbabilities(Ss, m, t, theta)
                    nS = len(Ss)
                    mx, idx_mx = -1e10, -1
                    for s in range(nS): # for each assortment (indexed by s) there is a value function
                        aux, S = 0, Ss[s]
                        inv_next = [inv[i]-S[i] for i in range(len(S))] # next inventory is deterministic function of current and asortment
                        for k in range(m+1):
                            idx_next = copy.copy(inv_next)
                            idx_next.append(nm+k)
                            aux+= P[s][nm][k]*(k+df*V[t+1][tuple(idx_next)])
                        if aux > mx:
                            mx, idx_mx = aux, s
                    idx_x = copy.copy(inv)
                    idx_x.append(nm)
                    X[t][tuple(idx_x)] = Ss[idx_mx]
                    V[t][tuple(idx_x)] = mx
    return V,X


if __name__ == '__main__':
    # Generate initial types
    n,m,df = 3,5,0.9
    Q = [0.45, 0.4, 0.3]
    # distr = stats.norm()
    distr = stats.uniform()
    # Generate initial inventory of types in backlog
    T = 5
    inventory = [m,m*T,m*T]
    # theta = [0.1,0.1,2] # effect of assortment, effect of history, and target numebr of matches
    theta = [0.1,0.1,2] # effect of assortment, effect of history, and target numebr of matches


    Ss = GenerateAssortments(n, m, inventory)
    print Ss

    U,P = ComputeProbabilities(Ss, m, 1, theta)

    V,X = SolveBackwards(n,m,T,theta,inventory, df)
    print X[1]


    eis = {0:inventory} # ending inventories
    for t in range(1, T+1):
        eis[t] = {}
        if t == 1:
            print '------------------------'
            print 'Period, Inventory, Num Matches, Decision, Value '
            for k in X[t]:
                inv, nm = list(k[:3]), k[3]
                print t, inv, nm, X[t][k], V[t][k]
                eis[t][k] = [inventory[i] - X[t][k][i] for i in range(n)]
            print '------------------------'
        else:
            print '------------------------'
            print 'Period, Inventory, Num Matches, Decision, Value '
            for k in eis[t-1]:
                prev_m = k[3]
                for nm in range(m+1): # for number of matches in previous period
                    nm_t = nm + prev_m
                    idx = copy.copy(eis[t-1][k])
                    idx.append(nm_t)
                    print t, eis[t-1][k], nm_t, X[t][tuple(idx)], V[t][tuple(idx)]
                    eis[t][tuple(idx)] = [eis[t-1][k][i] - X[t][tuple(idx)][i] for i in range(n)]
                print '------------------------'




    # Simulations
    for sim in range(10):
        Q = sorted(list(numpy.random.uniform(0,1,n)), reverse=True)
        V,X = SolveBackwards(n,m,T,theta,inventory)

        scn_data = {'X':X[1][X[1].keys()[0]], 'V':V[1][X[1].keys()[0]], 'Q':Q}
        with open(os.path.join('outputs', 'scenario_'+str(sim)+'.txt'), 'w') as outfile:
            json.dump(scn_data, outfile)



    eis = {0:inventory} # ending inventories
    for t in range(1, T+1):
        eis[t] = {}
        if t == 1:
            print X[t], V[t]
            for k in X[t]:
                inv, nm = list(k[:3]), k[3]
                eis[t][k] = [inventory[i] - X[t][k][i] for i in range(n)]
        else:
            for k in eis[t-1]:
                prev_m = k[3]
                for nm in range(m+1): # for number of matches in previous period
                    nm_t = nm + prev_m
                    idx = copy.copy(eis[t-1][k])
                    idx.append(nm_t)
                    print nm_t, X[t][tuple(idx)], V[t][tuple(idx)]
                    eis[t][tuple(idx)] = [eis[t-1][k][i] - X[t][tuple(idx)][i] for i in range(n)]

    norm = stats.norm()
    q1 = -2.49
    q2 = -2.5
    delta = 0.01
    print 2*norm.cdf(q1)
    print norm.cdf(q1+delta*(q1-q2)) + norm.cdf(q1-(1+delta)*(q1-q2))
