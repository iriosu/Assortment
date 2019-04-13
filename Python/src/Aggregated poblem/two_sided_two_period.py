from gurobipy import *
import sys, os
import numpy, math
import itertools
import copy
numpy.random.seed(1)
import subprocess, multiprocessing, time, gc, json
import matplotlib.pyplot as plt

# ==============================
# AUXILIAR FUNCTIONS
# ==============================
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

def CreateInputs(n,m,delta,boo_m=True):
    def phi(i, A, values, beta=1):
        # Assume uniform distribution for outside option
        if i not in A:
            return 0
        else:
            return max(min(values[i] + beta*sum([values[i]-values[j] for j in A]),1),0)

    U = sorted(list(numpy.random.uniform(0,1,n)), reverse=True)
    U = [round(u,3) for u in U]
    V = sorted(list(numpy.random.uniform(0,1,n)), reverse=True)
    V = [round(v,3) for v in V]

    S = []
    if boo_m:
        for comb in itertools.combinations(range(n), m):
            S.append(comb)
    else:
        for mm in range(1,m+1):
            for comb in itertools.combinations(range(n), mm):
                S.append(comb)

    d, pI, pJ = [], [], []
    for i in range(n):
        d.append([int(i in comb) for comb in S]) # mapping of users to assortments
        pI.append([phi(i, comb, V, delta) for comb in S]) # probabilities that users on side J are liked
        pJ.append([phi(i, comb, U, delta) for comb in S]) # probabilities that users on side I are liked

    DI, DJ, mp, all_pairings = CreatePairsMapping(n)

    return DI, DJ, mp, S, d, pI, pJ, all_pairings, U, V

def ConstructBacklogs(n,m,ls, out,boo_m):
    if len(ls) == n and sum(ls) <= n-m*(t-1) and boo_m: # assumes we must show at least one profile per user
        out.append(copy.copy(ls))
    elif len(ls) == n and sum(ls) <= n-(t-1) and not boo_m: # assumes we must show at least one profile per user
        out.append(copy.copy(ls))
    elif len(ls) >= n:
        pass
    else:
        for j in range(2):
            ls.append(j)
            ConstructBacklogs(n,m,ls, out, boo_m)
            ls.pop()

def FeasibleBacklogsInd(n,m,t,boo_m=True):
    out, ls = [], []
    for i in range(2):
        ls.append(i)
        ConstructBacklogs(n,m,ls, out,boo_m)
        ls.pop()
    return out

def FeasibleBacklogs(n,m,t,pairs,boo_m=True):
    def MergeBacklogs(n,m,ls, out):
        if len(ls) == n:
            if sum([sum([ls[i][j] for i in range(n)])<=m for j in range(n)]) == n:
                out.append(copy.copy(ls))
        else:
            for l1 in inp:
                ls.append(l1)
                MergeBacklogs(n,m,ls, out)
                ls.pop()

    inp = FeasibleBacklogsInd(n,m,t,boo_m)
    jaux = []
    lst = []
    for l in inp:
        lst.append(l)
        MergeBacklogs(n,m,lst, jaux)
        lst.pop()

    jout = []
    for b in jaux:
        # first level, scenario; second level, side of market;
        # third level, specific i in I (or j in J; forth level, specific j for i
        for b1 in jaux:
            if sum([b[pairs[p][0]][pairs[p][1]] + b1[pairs[p][1]][pairs[p][0]] <= 1 for p in range(len(pairs))]) == len(pairs):
                jout.append([b,b1])
    return jout

def ConstructPairings(n,m,ls,out, t,DI,DJ, boo_m):
    if len(ls) == n**2 and sum(DI.dot(ls)<= n-m*(t-1)) == n and sum(DJ.dot(ls)<= n-m*(t-1)) == n and boo_m:
        out.append(copy.copy(ls))
    elif len(ls) == n**2 and sum(DI.dot(ls)<= n-(t-1)) == n and sum(DJ.dot(ls)<= n-(t-1)) == n and not boo_m:
        out.append(copy.copy(ls))
    elif len(ls) >= n**2:
        pass
    else:
        for j in range(2):
            ls.append(j)
            ConstructPairings(n,m,ls,out, t,DI,DJ, boo_m)
            ls.pop()

def FeasiblePairings(n,m,t,DI,DJ,boo_m=True):
    out, ls = [], []
    for i in range(2):
        ls.append(i)
        ConstructPairings(n,m,ls,out,t,DI,DJ,boo_m)
        ls.pop()
    return out

def BuildModelParallel(data):
    # init model
    n, Bs, Ps, S, d, prI, prJ, mp, ct, outdir = data[0],data[1],data[2],data[3],data[4],\
                                                data[5],data[6],data[7], data[8],data[9]
    nS = len(S)
    model = Model('mip1')
    xI,xJ,Z = {},{},{}
    for i in range(n):
        for s in range(nS):
            xI[i,s] = model.addVar(vtype=GRB.CONTINUOUS, name='xI'+str(i)+str(s))
    for j in range(n):
        for s in range(nS):
            xJ[j,s] = model.addVar(vtype=GRB.CONTINUOUS, name='xJ'+str(j)+str(s))
    for i in range(n):
        for j in range(n):
            for r in range(nS):
                for s in range(nS):
                    Z[i,j,r,s] = model.addVar(vtype=GRB.CONTINUOUS, name='Z'+str(i)+str(j)+str(r)+str(s))

    # =====================
    # add constraints
    # =====================
    # if users are not in potentials nor in backlog they cannot be shown
    for i in range(n):
        for j in range(n):
            model.addConstr(sum([xI[i,s] * d[j][s] for s in range(nS)]) <= Bs[0][i][j] + Ps[mp[i,j]], "PI"+str(i)+str(j))
    for j in range(n):
        for i in range(n):
            model.addConstr(sum([xJ[j,s] * d[i][s] for s in range(nS)]) <= Bs[1][j][i] + Ps[mp[i,j]], "PJ"+str(i)+str(j))

    # only one assortment per period
    for i in range(n):
        model.addConstr(sum([xI[i,s] for s in range(nS)]) <= 1, "OAI"+str(i))
    for j in range(n):
        model.addConstr(sum([xJ[j,s] for s in range(nS)]) <= 1, "OAJ"+str(j))

    # simultaneous shows
    for i in range(n):
        for j in range(n):
            for r in range(nS):
                for s in range(nS):
                    model.addConstr(Z[i,j,r,s] <= xI[i,r], "MSI"+str(i)+str(j)+str(r)+str(s))
                    model.addConstr(Z[i,j,r,s] <= xJ[j,s], "MSJ"+str(i)+str(j)+str(r)+str(s))

    # Set objective
    model.setObjective(sum([xI[i,r] * d[j][r] * Bs[0][i][j] * prI[j][r] for i in range(n) for j in range(n) for r in range(nS)]) \
                   + sum([xJ[j,s] * d[i][s] * Bs[1][j][i] * prJ[i][s] for i in range(n) for j in range(n) for s in range(nS)]) \
                   + sum([Z[i,j,r,s]* d[j][r] * d[i][s] * prI[j][r] * prJ[i][s] for i in range(n) for j in range(n) for r in range(nS) for s in range(nS)]) \
                   , GRB.MAXIMIZE)
    model.optimize()
    obj = model.objVal

    xI_opt = {}
    xJ_opt = {}
    for v in model.getVars():
        if v.varName[0] == 'x':
            key = v.varName[2:]
            if 'xI' in v.varName:
                xI_opt[key] = v.x
            else:
                xJ_opt[key] = v.x
        else:
            pass
    scn_data = {'P':Ps, 'B':Bs, 'ct':ct, 'obj':obj, 'xI': xI_opt, 'xJ': xJ_opt}

    with open(os.path.join(outdir, 'scenario_s='+str(ct)+'.txt'), 'w') as outfile:
        json.dump(scn_data, outfile)

def FeasibleSolutions(n,m,t,D,boo_m):
    out, ls = [], []
    for i in range(2):
        ls.append(i)
        ConstructSolution(n,m,ls, out,D, boo_m)
        ls.pop()
    return out

def ConstructSolution(n,m,ls,out, D, boo_m):
    # i.e. every user is offered m profiles
    if len(ls) == n**2 and sum(D.dot(ls)== m) == n and boo_m:
        out.append(copy.copy(ls))
    # i.e. every user is offered at most m profiles, and at least 1
    elif len(ls) == n**2 and sum(D.dot(ls)<= m) == n and sum(D.dot(ls)>= 1) == n and not boo_m:
        out.append(copy.copy(ls))
    elif len(ls) >= n**2:
        pass
    else:
        for j in range(2):
            ls.append(j)
            ConstructSolution(n,m,ls,out, D, boo_m)
            ls.pop()

def GetVars(xi, xj, n, S):
    SI = {i:[sum([(xi[n*i+j] == 0 and j not in s) or (xi[n*i+j] == 1 and j in s) for j in range(n)]) == n\
          for s in S] for i in range(n)}
    SJ = {j:[sum([(xj[n*i+j] == 0 and i not in s) or (xj[n*i+j] == 1 and i in s) for i in range(n)]) == n\
          for s in S] for j in range(n)}
    return SI, SJ

def ComputeTransitionProb(xIS,xJS,prI,prJ,Bsn,n,nS,d):
    probI=1
    probJ=1
    for i in range(n):
        for r in range(nS):
            if xIS[i][r] == 0:
                continue
            for j in range(n):
                if Bsn[1][j][i] == 1 and xIS[i][r]*d[j][r] == 0:
                    return 0 # cannot be in backlog if does not see
                elif Bsn[1][j][i] == 1 and sum([d[i][s]*xJS[j][s] for s in range(nS)]) == 1:
                    return 0 # cannot be in backlog and being seen in same period
                elif Bsn[1][j][i] == 0 and xIS[i][r]*d[j][r] == 0:
                    probI*=1 # no update if not shown and backlog compatible
                elif sum([d[i][s]*xJS[j][s] for s in range(nS)]) == 1:
                    probI*=1 # simultaneous shows
                else: # no simultaneous shows
                    probI*=math.pow(prI[j][r], Bsn[1][j][i]) * math.pow(1-prI[j][r], 1-Bsn[1][j][i])
    for j in range(n):
        for s in range(nS):
            if xJS[j][s] == 0:
                continue
            for i in range(n):
                if Bsn[0][i][j] == 1 and xJS[j][s]*d[i][s] == 0:
                    return 0
                elif Bsn[0][i][j] == 1 and sum([d[j][r]*xIS[i][r] for r in range(nS)]) == 1:
                    return 0
                elif Bsn[0][i][j] == 0 and xJS[j][s]*d[i][s] == 0:
                    probJ*=1
                elif sum([d[j][r]*xIS[i][r] for r in range(nS)]) == 1:
                    probJ*=1
                else:
                    probJ*=math.pow(prJ[i][s], Bsn[0][i][j]) * math.pow(1-prJ[i][s], 1-Bsn[0][i][j])
    prob = probI*probJ
    return prob

# ==============================
# MAIN FUNCTIONS PER STAGE
# ==============================
def SolveSecondStageParallel(n,m,t,delta,outdir='aux', boo_m=True, num_processors=3):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    DI, DJ, mp, S, d, pI, pJ, pairs, U, V = CreateInputs(n,m,delta,False) # these are inputs for second stage model
    setup_data = {'n':n, 'm':m, 't':t, 'delta':delta,'S':S,'d':d,'pI':pI,'pJ':pJ,'U':U,'V':V}
    with open(os.path.join(outdir, 'setup.txt'), 'w') as outfile:
        json.dump(setup_data, outfile)
    # feasible backlogs and potentials in second stage,
    # assuming that in first stage everyones is offered m profiles
    feasible_backlogs = FeasibleBacklogs(n,m,t,pairs,boo_m)
    potential_pairs = FeasiblePairings(n,m,t,DI,DJ, boo_m)
    Bs, Ps = {}, {}
    ct = 0
    indata = []
    for fb in feasible_backlogs:
        for fp in potential_pairs:
            if sum([(fb[0][pairs[p][0]][pairs[p][1]] + fb[1][pairs[p][1]][pairs[p][0]] + fp[p] <= 1) for p in range(len(pairs))]) == len(pairs):
                indata.append((n, fb, fp, S, d, pI, pJ, mp, ct, outdir))
                ct+=1
    np = min(num_processors,multiprocessing.cpu_count())
    # We execute our process for each replication with a pool
    pool = multiprocessing.Pool(processes=min(np, len(indata)))
    pool.map(BuildModelParallel, indata)
    pool.close()
    pool.join()

def SolveFirstStage(n,m,t,outdir, boo_m=True):
    # Create inputs
    DI, DJ, mp, pairs = CreatePairsMapping(n)
    # Read second stage
    dataSS = {}
    for filename in os.listdir(outdir):
        if filename.endswith(".txt"):
            if 'setup' in filename or 'first_stage' in filename:
                continue
            ct = int(filename.split('=')[1].split('.')[0])
            with open(os.path.join(outdir, filename)) as json_file:
                dataSS[ct] = json.load(json_file)
        else:
            continue

    # Initialize
    B1 = [[[0 for i in range(n)] for j in range(n)] for k in range(2)]
    P1 = [1 for i in range(n**2)]
    with open(os.path.join(outdir, 'setup.txt')) as json_file:
        setup = json.load(json_file)
    prI, prJ, S, d = setup['pI'], setup['pJ'], setup['S'], setup['d']
    nS = len(S)
    V_opt,xI_opt,xJ_opt = 0,0,0

    # feasible solutions for first stage
    fxI = FeasibleSolutions(n,m,t,DI, boo_m) # True to force to offer assortment of size m
    fxJ = FeasibleSolutions(n,m,t,DJ, boo_m) # True to force to offer assortment of size m
    for xI in fxI:
        for xJ in fxJ:
            xIS, xJS = GetVars(xI, xJ, n, S)
            mx = [max(xI[k], xJ[k]) for k in range(len(xI))]
            P2 = [P1[k] - mx[k] for k in range(len(mx))]
            exp_val = sum([xIS[i][r] * xJS[j][s] * d[j][r] * d[i][s] * prI[j][r] * prJ[i][s]\
                           for i in range(n) for j in range(n) \
                           for r in range(nS) for s in range(nS)])
            cum_prob = 0
            for scn in dataSS:
                Bsn = dataSS[scn]['B']
                Psn = dataSS[scn]['P']
                if P2 != Psn: # the probability of this scenario is 0 is P2 != Psn
                    continue
                if sum([B1[0][i][j] == 1 and Bsn[0][i][j] != 1-xI[n*i+j] \
                        for i in range(n) for j in range(n)])\
                != sum([B1[0][i][j] == 1 for i in range(n) for j in range(n)]):
                    continue
                prob = ComputeTransitionProb(xIS,xJS,prI,prJ,Bsn,n,nS,d)
                if prob == 0:
                    continue
                cum_prob += prob
                exp_val += dataSS[scn]['obj']*prob
            if cum_prob < 1-1e-4:
                print('Error: transition probabilities do not add up to 1')
                print(xI, xJ)
                print(cum_prob)
                sys.exit(1)
            if exp_val > V_opt:
                V_opt = exp_val
                xI_opt = xI
                xJ_opt = xJ

    fs_data = {'obj':V_opt, 'xI': xI_opt, 'xJ': xJ_opt}
    with open(os.path.join(outdir, 'first_stage.txt'), 'w') as outfile:
        json.dump(fs_data, outfile)

def SolveStages(n,m,t,delta,outdir,boo_m=True,num_processors=3):
    # Solve second stage for every scenario
    SolveSecondStageParallel(n,m,t,delta,outdir,boo_m,num_processors)
    # Solve initial stage by brute force
    SolveFirstStage(n,m,t,outdir,boo_m)

if __name__ == '__main__':
    n,m,t,delta,bind_m = 3,2,2,0.1,True
    num_sim, num_processors = 1, 3

    for sim in range(num_sim):
        if bind_m:
            outdir = 'sim_bind_'+str(sim)
        else:
            outdir = 'sim_nonbind_'+str(sim)
        SolveStages(n,m,t,delta,outdir,bind_m,num_processors)
