from gurobipy import *
import sys, os
import numpy, math
import itertools
import copy
numpy.random.seed(1)
import subprocess, multiprocessing, time, gc, json
import matplotlib.pyplot as plt


class Node():
    def __init__(self, B=None, P=None, id=None, prob=None):
        '''
        P and B uniquely define a state
        '''
        self.B = B
        self.P = P
        self.id = id
        self.prob = prob
        self.scn = None
        self.value = None
        self.own_value = None
        self.cont_value = None
        self.adj_out = []
        self.adj_in = []
        self.prob_out = []
        self.prob_in = []
        self.xI = None
        self.xJ = None


def SimulatedBacklogs(n,xI,xJ,B_prev):
    '''
    Returns a dictionary where keys are users, and for each user
    we have a list of lists containing all possible backlogs for that user.
    '''
    BI = {i:[] for i in range(n)}
    for i in range(n):
        idxs = [j for j in range(n) if xJ[j][i] == 1 and xI[i][j] == 0 and B_prev[1][j][i] == 0]
        # we need to update these indexes in BI, i.e. make them 0 or 1 with all combinations
        combs = [p for p in itertools.product([0,1], repeat=len(idxs))]
        if len(idxs) == 0:
            BI[i].append(B_prev[0][i])
        else:
            for comb in combs:
                aux = copy.copy(B_prev[0][i])
                for k in range(len(idxs)):
                    aux[idxs[k]] = comb[k]
                BI[i].append(aux)
    BJ = {j:[] for j in range(n)}
    for j in range(n):
        idxs = [i for i in range(n) if xI[i][j] == 1 and xJ[j][i] == 0 and B_prev[0][i][j] == 0]
        # we need to update these indexes in BI, i.e. make them 0 or 1 with all combinations
        combs = [p for p in itertools.product([0,1], repeat=len(idxs))]
        if len(idxs) == 0:
            BJ[j].append(B_prev[1][j])
        else:
            for comb in combs:
                aux = copy.copy(B_prev[1][j])
                for k in range(len(idxs)):
                    aux[idxs[k]] = comb[k]
                BJ[j].append(aux)
    print '    BI = ', BI
    print '    BJ = ', BJ
    return BI, BJ

def CombineBacklogs(Bs):
    def Merge(n,ls,out,Bs):
        if len(ls) == n:
            out.append(copy.copy(ls))
        else:
            for l1 in Bs[len(ls)]:
                ls.append(l1)
                Merge(n,ls,out,Bs)
                ls.pop()

    n = len(Bs)
    jaux = []
    lst = []
    for l in Bs[len(lst)]:
        lst.append(l)
        Merge(n,lst,jaux,Bs)
        lst.pop()
    return jaux

def CreatePairsMapping(n):
    all_pairings = []
    for i in range(n):
        for j in range(n):
            all_pairings.append([i,j])
    mp = {}
    for i in range(len(all_pairings)):
        mp[all_pairings[i][0],all_pairings[i][1]] = i
    DI = []
    DJ = []
    for i in range(n):
        DI.append([int(pair[0] == i) for pair in all_pairings])
        DJ.append([int(pair[1] == i) for pair in all_pairings])
    DI = numpy.array(DI)
    DJ = numpy.array(DJ)
    return DI, DJ, mp, all_pairings

def GenerateRandomUtilities(n):
    U = sorted(list(numpy.random.uniform(0.001,0.999,n)), reverse=True)
    U = [round(u,3) for u in U]
    V = sorted(list(numpy.random.uniform(0.001,0.999,n)), reverse=True)
    V = [round(v,3) for v in V]
    return U,V

def CreateInputs(U,V,m,delta,boo_m=True):
    def phi(i, A, values, beta=1):
        # Assume uniform distribution for outside option
        if i not in A:
            return 0
        else:
            return max(min(values[i] + beta*sum([values[i]-values[j] for j in A]),1),0)
    pairs = []
    for i in range(n):
        for j in range(n):
            pairs.append([i,j])

    S = []
    if boo_m:
        for comb in itertools.combinations(range(n), m):
            S.append(comb)
    else:
        for mm in range(1,m+1):
            for comb in itertools.combinations(range(n), mm):
                S.append(comb)

    S_inv = {S[i]:i for i in range(len(S))}

    d, pI, pJ = [], [], []
    for i in range(n):
        d.append([int(i in comb) for comb in S]) # mapping of users to assortments
        pI.append([phi(i, comb, V, delta) for comb in S]) # probabilities that users on side J are liked
        pJ.append([phi(i, comb, U, delta) for comb in S]) # probabilities that users on side I are liked

    return pairs, S, S_inv, d, pI, pJ

def ComputeTransitionProb(xIS,xJS,pI,pJ,B_start,B_end,n,nS,d):
    def TransitionBySide(X,Y,Bst,Bend,pr):
        prob_out=1
        for i in range(n):
            for s in range(nS):
                if X[i][s] == 0:
                    continue
                for j in range(n):
                    if Bst[j][i] == 0:
                        if (X[i][s]*d[j][s] == 0 or sum([d[i][r]*Y[j][r] for r in range(nS)]) == 1) and Bend[j][i] == 0:
                            prob_out*=1
                            # print("    ", Bst[j][i], Bend[j][i], X[i][s]*d[j][s])
                        elif (X[i][s]*d[j][s] == 0 or sum([d[i][r]*Y[j][r] for r in range(nS)]) == 1) and Bend[j][i] == 1:
                            return 0 # cannot be added to backlog if i does not see j or if simultaneous show
                        else:
                            prob_out*=math.pow(pr[j][s], Bend[j][i]) * math.pow(1-pr[j][s], 1-Bend[j][i])
                    elif Bst[j][i] == 1:
                        if (X[i][s]*d[j][s] == 0 and Bend[j][i] == 0) or (X[i][s]*d[j][s] == 1 and Bend[j][i] == 1):
                            return 0
                        elif (X[i][s]*d[j][s] == 0 and Bend[j][i] == 1) or (X[i][s]*d[j][s] == 1 and Bend[j][i] == 0):
                            prob_out*=1
                        else:
                            print("Error: unkonwn case.")
                            print(Bst[j][i], Bend[j][i])
                            sys.exit(1)
                    else:
                        print("Error: unkonwn case.")
                        print(Bst[j][i], Bend[j][i])
                        sys.exit(1)
        return prob_out
    probI = TransitionBySide(xIS, xJS, B_start[1], B_end[1], pI)
    probJ = TransitionBySide(xJS, xIS, B_start[0], B_end[0], pJ)
    prob = probI*probJ
    return prob

def FindNode(B_in, P_in, nodes):
    idx = -1
    for id in nodes:
        if nodes[id].B ==  B_in and nodes[id].P == P_in:
            idx = id
            break
    return idx

# =======================================
# POLICIES
# =======================================
def RandomPolicy(n,m,nS,P_prev,B_prev,S_inv):
    '''
    In every period select randomly from potentials and backlog
    '''
    xI, xIS =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for i in range(n)]
    for i in range(n):
        aux = [j for j in range(n) if P_prev[i][j] == 1 or B_prev[0][i][j] == 1]
        if len(aux) > 0:
            ast = tuple(sorted(numpy.random.choice(aux, min(m,len(aux)), False)))
            xIS[i][S_inv[ast]]=1
            for j in range(len(ast)):
                xI[i][ast[j]] = 1

    xJ, xJS =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for j in range(n)]
    for j in range(n):
        aux = [i for i in range(n) if P_prev[i][j] == 1 or B_prev[1][j][i] == 1]
        if len(aux) > 0:
            ast = tuple(sorted(numpy.random.choice(aux, min(m,len(aux)), False)))
            xJS[j][S_inv[ast]]=1
            for i in range(len(ast)):
                xJ[j][ast[i]] = 1
    return xI, xJ, xIS, xJS

def RandomGreedyPolicy(n,m,nS,P_prev,B_prev,S_inv):
    '''
    When backlog is empty, selects randomly. When backlog is not empty,
    selects as many profiles from backlog as possible
    '''
    xI, xIS =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for i in range(n)]
    for i in range(n):
        aux_P = [j for j in range(n) if P_prev[i][j] == 1]
        aux_B = [j for j in range(n) if B_prev[0][i][j] == 1]
        boo = True
        if len(aux_B) > 0 and len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            aux2 = numpy.random.choice(aux_P, m-len(aux1), False)
            ast = tuple(sorted(numpy.append(aux1, aux2)))
        elif len(aux_B) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            ast = tuple(sorted(aux1))
        elif len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_P, min(m,len(aux_P)), False)
            ast = tuple(sorted(aux1))
        else:
            boo = False
        if boo:
            xIS[i][S_inv[ast]]=1
            for j in range(len(ast)):
                xI[i][ast[j]] = 1

    xJ, xJS =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for j in range(n)]
    for j in range(n):
        aux_P = [i for i in range(n) if P_prev[i][j] == 1]
        aux_B = [i for i in range(n) if B_prev[1][j][i] == 1]
        boo = True
        if len(aux_B) > 0 and len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            aux2 = numpy.random.choice(aux_P, m-len(aux1), False)
            ast = tuple(sorted(numpy.append(aux1, aux2)))
        elif len(aux_B) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            ast = tuple(sorted(aux1))
        elif len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_P, min(m,len(aux_P)), False)
            ast = tuple(sorted(aux1))
        else:
            boo = False
        if boo:
            xJS[j][S_inv[ast]]=1
            for i in range(len(ast)):
                xJ[j][ast[i]] = 1
    return xI, xJ, xIS, xJS

def SymmetricGreedyPolicy(n,m,nS,P_prev,B_prev,S_inv):
    '''
    When backlog is empty, selects randomly but symmetric. When backlog is not empty,
    selects as many profiles from backlog as possible
    '''
    xI, xIS =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for i in range(n)]
    for i in range(n):
        aux_P = [j for j in range(n) if P_prev[i][j] == 1]
        aux_B = [j for j in range(n) if B_prev[0][i][j] == 1]
        boo = True
        if len(aux_B) > 0 and len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            aux2 = numpy.random.choice(aux_P, m-len(aux1), False)
            ast = tuple(sorted(numpy.append(aux1, aux2)))
        elif len(aux_B) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            ast = tuple(sorted(aux1))
        elif len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_P, min(m,len(aux_P)), False)
            ast = tuple(sorted(aux1))
        else:
            boo = False
        if boo:
            xIS[i][S_inv[ast]]=1
            for j in range(len(ast)):
                xI[i][ast[j]] = 1

    xJ, xJS = xI, xIS
    for j in range(n):
        aux_P = [i for i in range(n) if P_prev[i][j] == 1]
        aux_B = [i for i in range(n) if B_prev[1][j][i] == 1]
        boo = True
        if len(aux_B) > 0 and len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            aux2 = numpy.random.choice(aux_P, m-len(aux1), False)
            ast = tuple(sorted(numpy.append(aux1, aux2)))
        elif len(aux_B) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            ast = tuple(sorted(aux1))
        else:
            boo = False
        if boo:
            xJS[j][S_inv[ast]]=1
            for i in range(len(ast)):
                xJ[j][ast[i]] = 1
    return xI, xJ, xIS, xJS

def PrioritizeBacklog(n,m,nS,P_prev,B_prev,S_inv):
    '''
    When backlog is empty, selects randomly. When backlog is not empty,
    selects pairs that maximizes epxected number of matches
    '''
    xI, xIS =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for i in range(n)]
    for i in range(n):
        aux_P = [j for j in range(n) if P_prev[i][j] == 1]
        aux_B = [j for j in range(n) if B_prev[0][i][j] == 1]
        boo = True
        if len(aux_B) > 0 and len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            aux2 = numpy.random.choice(aux_P, m-len(aux1), False)
            ast = tuple(sorted(numpy.append(aux1, aux2)))
        elif len(aux_B) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            ast = tuple(sorted(aux1))
        elif len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_P, min(m,len(aux_P)), False)
            ast = tuple(sorted(aux1))
        else:
            boo = False
        if boo:
            xIS[i][S_inv[ast]]=1
            for j in range(len(ast)):
                xI[i][ast[j]] = 1

    xJ, xJS =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for j in range(n)]
    for j in range(n):
        aux_P = [i for i in range(n) if P_prev[i][j] == 1]
        aux_B = [i for i in range(n) if B_prev[1][j][i] == 1]
        boo = True
        if len(aux_B) > 0 and len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            aux2 = numpy.random.choice(aux_P, m-len(aux1), False)
            ast = tuple(sorted(numpy.append(aux1, aux2)))
        elif len(aux_B) > 0:
            aux1 = numpy.random.choice(aux_B, min(m,len(aux_B)), False)
            ast = tuple(sorted(aux1))
        elif len(aux_P) > 0:
            aux1 = numpy.random.choice(aux_P, min(m,len(aux_P)), False)
            ast = tuple(sorted(aux1))
        else:
            boo = False
        if boo:
            xJS[j][S_inv[ast]]=1
            for i in range(len(ast)):
                xJ[j][ast[i]] = 1
    return xI, xJ, xIS, xJS

def GreedyOpt(n,m,nS,P_prev,B_prev,d,prI,prJ):
    def BuildSolveModel(data):
        # init model
        n, m, Bs, Ps, nS, d, prI, prJ = data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7]

        model = Model('mip1')
        xI,xJ,Z = {},{},{}
        for i in range(n):
            for s in range(nS):
                # xI[i,s] = model.addVar(vtype=GRB.CONTINUOUS, name='xI'+str(i)+'_'+str(s))
                xI[i,s] = model.addVar(vtype=GRB.BINARY, name='xI'+str(i)+'_'+str(s))
        for j in range(n):
            for s in range(nS):
                #xJ[j,s] = model.addVar(vtype=GRB.CONTINUOUS, name='xJ'+str(j)+'_'+str(s))
                xJ[j,s] = model.addVar(vtype=GRB.BINARY, name='xJ'+str(j)+'_'+str(s))
        for i in range(n):
            for j in range(n):
                for r in range(nS):
                    for s in range(nS):
                        Z[i,j,r,s] = model.addVar(vtype=GRB.CONTINUOUS, name='Z'+str(i)+'_'+str(j)+'_'+str(r)+'_'+str(s))

        # =====================
        # add constraints
        # =====================
        #if users are not in potentials nor in backlog they cannot be shown
        for i in range(n):
            for j in range(n):
                model.addConstr(sum([xI[i,r] * d[j][r] for r in range(nS)]) <= Bs[0][i][j] + Ps[i][j], "PI"+str(i)+'_'+str(j))
        for j in range(n):
            for i in range(n):
                model.addConstr(sum([xJ[j,s] * d[i][s] for s in range(nS)]) <= Bs[1][j][i] + Ps[i][j], "PJ"+str(i)+'_'+str(j))

        # at most one assortment per period
        for i in range(n):
            model.addConstr(sum([xI[i,r] for r in range(nS)]) <= 1, "OAI"+str(i))
        for j in range(n):
            model.addConstr(sum([xJ[j,s] for s in range(nS)]) <= 1, "OAJ"+str(j))

        # show m profiles in each period
        for i in range(n):
            model.addConstr(sum([xI[i,r]*d[j][r] for j in range(n) for r in range(nS)]) == min(m,sum([Bs[0][i][j] + Ps[i][j] for j in range(n)])), "SMI"+str(i))
        for j in range(n):
            model.addConstr(sum([xJ[j,s]*d[i][s] for i in range(n) for s in range(nS)]) == min(m,sum([Bs[0][i][j] + Ps[i][j] for j in range(n)])), "SMJ"+str(j))

        # simultaneous shows
        for i in range(n):
            for j in range(n):
                for r in range(nS):
                    for s in range(nS):
                        model.addConstr(Z[i,j,r,s] <= xI[i,r], "MSI"+str(i)+'_'+str(j)+'_'+str(r)+'_'+str(s))
                        model.addConstr(Z[i,j,r,s] <= xJ[j,s], "MSJ"+str(i)+'_'+str(j)+'_'+str(r)+'_'+str(s))

        # Set objective
        model.setObjective(sum([xI[i,r] * d[j][r] * Bs[0][i][j] * prI[j][r] for i in range(n) for j in range(n) for r in range(nS)]) \
                       + sum([xJ[j,s] * d[i][s] * Bs[1][j][i] * prJ[i][s] for i in range(n) for j in range(n) for s in range(nS)]) \
                       + sum([Z[i,j,r,s]* d[j][r] * d[i][s] * prI[j][r] * prJ[i][s] for i in range(n) for j in range(n) for r in range(nS) for s in range(nS)]) \
                       , GRB.MAXIMIZE)
        model.write('greedy_opt_fs.lp')
        model.optimize()
        obj = model.objVal

        xI_opt, xIS_opt =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for i in range(n)]
        xJ_opt, xJS_opt =  [[0 for i in range(n)] for j in range(n)], [[0 for k in range(nS)] for i in range(n)]
        for v in model.getVars():
            if v.varName[0] == 'x':
                key = v.varName[2:]
                if 'xI' in v.varName:
                    pieces = key.split('_')
                    i = int(pieces[0])
                    r = int(pieces[1])
                    xIS_opt[i][r] = v.x
                else:
                    pieces = key.split('_')
                    j = int(pieces[0])
                    s = int(pieces[1])
                    xJS_opt[j][s] = v.x
            else:
                pass
        for i in range(n):
            for j in range(n):
                xI_opt[i][j] = sum([xIS_opt[i][r]*d[j][r] for r in range(nS)])
                xJ_opt[j][i] = sum([xJS_opt[j][s]*d[i][s] for s in range(nS)])

        return xI_opt, xJ_opt, xIS_opt, xJS_opt

    # Execution
    indata = (n,m, B_prev, P_prev, nS, d, prI, prJ)
    xI, xJ, xIS, xJS = BuildSolveModel(indata)
    return xI, xJ, xIS, xJS

# =======================================
# EXECUTION METODS
# =======================================
def SimulatePolicy(n,m,T,U,V,delta,policy):
    pairs, S, S_inv, d, pI, pJ = CreateInputs(U,V,m,delta,False)
    B1 = [[[0 for i in range(n)] for j in range(n)] for k in range(2)]
    P1 = [[1 for i in range(n)] for j in range(n)]

    nS = len(S)
    all_nodes = {}
    all_nodes[0] = Node(B1,P1,0,1)
    nodes_in_period = {t:[] for t in range(T)}
    nodes_in_period[0] = [0] # nodes by id
    nct = 0
    for t in range(T):
        # 1. Define what assortments to show for a given policy for a given scenario
        for nid in nodes_in_period[t]:
            print("Processing node "+str(nid) + " in period "+str(t))
            current_node = all_nodes[nid]
            B_current, P_current = current_node.B, current_node.P

            #TODO: Change policy here
            if policy == 'random':
                xI, xJ, xIS, xJS = RandomPolicy(n,m,nS,P_current,B_current,S_inv)
            elif policy == 'greedy':
                xI, xJ, xIS, xJS = RandomGreedyPolicy(n,m,nS,P_current,B_current,S_inv)
            elif policy == 'symmetric':
                xI, xJ, xIS, xJS = SymmetricGreedyPolicy(n,m,nS,P_current,B_current,S_inv)
            elif policy == 'greedy optimal':
                xI, xJ, xIS, xJS = GreedyOpt(n,m,nS,P_current,B_current,d,pI,pJ)
            else:
                print("ERROR: Unknown policy.")
                sys.exit(1)
            current_node.xI = xI
            current_node.xJ = xJ
            # 2. Compute expected number of matches in current situation
            # (i) from simultaneous shows
            exp_val = sum([xIS[i][r] * xJS[j][s] * d[j][r] * d[i][s] * pI[j][r] * pJ[i][s]\
                           for i in range(n) for j in range(n) \
                           for r in range(nS) for s in range(nS)])
            # (ii) from backlog
            exp_val += sum([xIS[i][r] * d[j][r] * B_current[0][i][j] * pI[j][r] for i in range(n) for j in range(n) for r in range(nS)]) \
                    + sum([xJS[j][s] * d[i][s] * B_current[1][j][i] * pJ[i][s] for i in range(n) for j in range(n) for s in range(nS)])

            current_node.own_value = exp_val
            print('======================')
            print("Value: ", str(exp_val))
            print("Value: ", str(current_node.prob))
            print("xI: ", xI)
            print("xJ: ", xJ)
            print('======================')
            #### FROM HERE ON, NO CHANGE RELATIVE TO THE POLICY
            if t < T-1:
                # 3. Update state variables and derive next period scenarios; only needed if not in last period
                P_next = [[P_current[i][j] - max(xI[i][j], xJ[j][i]) for j in range(n)] for i in range(n)]
                BI, BJ = SimulatedBacklogs(n,xI,xJ,B_current)
                BIc = CombineBacklogs(BI)
                BJc = CombineBacklogs(BJ)
                print '    BIc = ', BIc
                print '    BJc = ', BJc
                cum_prob = 0
                for bic in BIc:
                    for bjc in BJc:
                        print '    bic = ', bic
                        print '    bjc = ', bjc
                        if sum([bic[i][j] + bjc[j][i] >= 2 for i in range(n) for j in range(n)]) > 0:
                            continue # cannot have simultaneously to users in eachothers backlog
                        B_next = [bic, bjc]
                        nct+=1
                        tp = ComputeTransitionProb(xIS,xJS,pI,pJ,B_current,B_next,n,nS,d)
                        if tp == 0:
                            continue
                        print '    BI = ', bic
                        print '    BJ = ', bjc
                        print '    prob = ', tp
                        all_nodes[nct] = Node(B_next, P_next, nct, tp)
                        nodes_in_period[t+1].append(nct)
                        all_nodes[nct].adj_in.append(current_node)
                        all_nodes[nct].prob_in.append(tp)
                        current_node.adj_out.append(all_nodes[nct])
                        current_node.prob_in.append(tp)
                        cum_prob+=tp
                if cum_prob < 1-1e-4:
                    print("ERROR: transition probabilities don't add up to 1.")
                    print("Current node id: " + str(nid))
                    print("Current period: " + str(t))
                    print("Cumulative probability: " +str(cum_prob))
                    sys.exit(1)
            else:
                current_node.value = current_node.own_value # no continuation


    # Compute expected value based on all nodes
    for t in sorted(nodes_in_period, reverse=True):
        if t == T-1:
            continue
        for nid in nodes_in_period[t]:
            exp_val = 0
            current_node = all_nodes[nid]
            for next_node in current_node.adj_out:
                exp_val+=next_node.value * next_node.prob
            current_node.value = exp_val + current_node.own_value

    return all_nodes[0].value, all_nodes[0].xI, all_nodes[0].xJ



if __name__ == '__main__':
    n,m,T = 5,2,2
    delta = 0.1
    outdir = 'outputs'
    debug = True

    if debug:
        U,V = GenerateRandomUtilities(n)
        print U,V
        val_gopt, xI_gopt, xJ_gopt = SimulatePolicy(n,m,T,U,V,delta,'greedy optimal')
        print val_gopt, xI_gopt, xJ_gopt
    else:
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        nsim = 200
        # implementing random policy (not necesarily simultaneous)
        values = {}
        for sim in range(nsim):
            stim = str(sim)
            U,V = GenerateRandomUtilities(n)
            val_rand, xI_rand, xJ_rand = SimulatePolicy(n,m,T,U,V,delta,'random')
            val_greed, xI_greed, xJ_greed = SimulatePolicy(n,m,T,U,V,delta,'greedy')
            val_symmetric, xI_symmetric, xJ_symmetric = SimulatePolicy(n,m,T,U,V,delta,'symmetric')
            values[stim]={'random': {'value':val_rand, 'xI': xI_rand, 'xJ':xJ_rand},\
                            'greedy':{'value':val_greed, 'xI': xI_greed, 'xJ':xJ_greed}, \
                            'symmetric':{'value':val_symmetric, 'xI': xI_symmetric, 'xJ':xJ_symmetric}}

        with open(os.path.join(outdir, 'simulated_policies.txt'), 'w') as outfile:
            json.dump(values, outfile)
    sys.exit(1)


    policy = 'greedy' #'random'
    pairs, S, S_inv, d, pI, pJ = CreateInputs(U,V,m,delta,False)
    B1 = [[[0 for i in range(n)] for j in range(n)] for k in range(2)]
    P1 = [[1 for i in range(n)] for j in range(n)]
    nS = len(S)
    all_nodes = {}
    all_nodes[0] = Node(B1,P1,0,1)
    nodes_in_period = {t:[] for t in range(T)}
    nodes_in_period[0] = [0] # nodes by id
    nct = 0
    for t in range(T):
        # 1. Define what assortments to show for a given policy for a given scenario
        for nid in nodes_in_period[t]:
            print("Processing node "+str(nid) + " in period "+str(t))
            current_node = all_nodes[nid]
            B_current, P_current = current_node.B, current_node.P

            #TODO: Change policy here
            if policy == 'random':
                xI, xJ, xIS, xJS = RandomPolicy(n,m,nS,P_current,B_current)
            elif policy == 'greedy':
                xI, xJ, xIS, xJS = RandomGreedyPolicy(n,m,nS,P_current,B_current)
            else:
                print("ERROR: Unknown policy.")
                sys.exit(1)

            current_node.xI = xI
            current_node.xJ = xJ
            # 2. Compute expected number of matches in current situation
            # (i) from simultaneous shows
            exp_val = sum([xIS[i][r] * xJS[j][s] * d[j][r] * d[i][s] * pI[j][r] * pJ[i][s]\
                           for i in range(n) for j in range(n) \
                           for r in range(nS) for s in range(nS)])
            # (ii) from backlog
            exp_val += sum([xIS[i][r] * d[j][r] * B_current[0][i][j] * pI[j][r] for i in range(n) for j in range(n) for r in range(nS)]) \
                    + sum([xJS[j][s] * d[i][s] * B_current[1][j][i] * pJ[i][s] for i in range(n) for j in range(n) for s in range(nS)])

            current_node.own_value = exp_val
            print('======================')
            print("Value: ", str(exp_val))
            print("Value: ", str(current_node.prob))
            print("xI: ", xI)
            print("xJ: ", xJ)
            print('======================')
            #### FROM HERE ON, NO CHANGE RELATIVE TO THE POLICY
            if t < T-1:
                # 3. Update state variables and derive next period scenarios; only needed if not in last period
                P_next = [[P_current[i][j] - max(xI[i][j], xJ[j][i]) for j in range(n)] for i in range(n)]
                BI, BJ = SimulatedBacklogs(n,xI,xJ,B_current)
                BIc = CombineBacklogs(BI)
                BJc = CombineBacklogs(BJ)
                cum_prob = 0
                for bic in BIc:
                    for bjc in BJc:
                        if sum([bic[i][j] + bjc[j][i] >= 2 for i in range(n) for j in range(n)]) > 0:
                            continue # cannot have simultaneously to users in eachothers backlog
                        B_next = [bic, bjc]
                        nct+=1
                        tp = ComputeTransitionProb(xIS,xJS,pI,pJ,B_current,B_next,n,nS,d)
                        if tp == 0:
                            continue
                        all_nodes[nct] = Node(B_next, P_next, nct, tp)
                        nodes_in_period[t+1].append(nct)
                        all_nodes[nct].adj_in.append(current_node)
                        all_nodes[nct].prob_in.append(tp)
                        current_node.adj_out.append(all_nodes[nct])
                        current_node.prob_in.append(tp)
                        cum_prob+=tp
                if cum_prob < 1-1e-4:
                    print("ERROR: transition probabilities don't add up to 1.")
                    print("Current node id: " + str(nid))
                    print("Current period: " + str(t))
                    print("Cumulative probability: " +str(cum_prob))
                    sys.exit(1)
            else:
                current_node.value = current_node.own_value # no continuation


    # Compute expected value based on all nodes
    for t in sorted(nodes_in_period, reverse=True):
        if t == T-1:
            continue
        for nid in nodes_in_period[t]:
            exp_val = 0
            current_node = all_nodes[nid]
            for next_node in current_node.adj_out:
                exp_val+=next_node.value * next_node.prob
            current_node.value = exp_val + current_node.own_value
            print(nid, current_node.value)


    print sum([0.09090909090909091, 0.09090909090909091, 0.0, 0.0, 0.0, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.09090909090909091, 0.0])
