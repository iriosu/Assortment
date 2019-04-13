import sys,os
import json
import matplotlib.pyplot as plt







if __name__ == '__main__':

    outdir = 'outputs'
    filename = 'simulated_policies.txt'

    with open(os.path.join(outdir, filename)) as json_file:
        results = json.load(json_file)

    nsim = len(results)
    sks = [str(n) for n in range(nsim)]
    plt.figure()
    plt.plot(sks, [results[k]["symmetric"]["value"] for k in sks], label='symmetric')
    plt.plot(sks, [results[k]["greedy"]["value"] for k in sks], label='greedy')
    plt.plot(sks, [results[k]["random"]["value"] for k in sks], label='random')
    plt.show(())

    plt.figure()
    n, bins, patches = plt.hist([results[k]["symmetric"]["value"] for k in sks], 10, facecolor='blue', alpha=0.5, label='symmetric')
    n, bins, patches = plt.hist([results[k]["greedy"]["value"] for k in sks], 10, facecolor='red', alpha=0.5, label='greedy')
    n, bins, patches = plt.hist([results[k]["random"]["value"] for k in sks], 10, facecolor='green', alpha=0.5, label='random')
    plt.legend()
    plt.show()
