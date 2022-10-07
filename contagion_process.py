import matplotlib.pyplot as plt
import numpy as np
import random
import multiprocessing as mp

def ERmodel(n):
    a = random.randint(0, n-1)
    b = random.randint(0, n-1)
    if a != b:
        if a > b:
            if links.count((b, a)) == 0:
                links.append((b, a))
                neighbors[a] += 1
                neighbors[b] += 1
            else:
                return ERmodel(n)
        if a < b:
            if links.count((a, b)) == 0:
                links.append((a, b))
                neighbors[a] += 1
                neighbors[b] += 1
            else:
                return ERmodel(n)
    else:
        return ERmodel(n)

def CP(p):
    r = 0
    nodes_copy = nodes[:]
    cp = [nodes_copy.count(-1)]
    cp_copy = cp[:]
    while True:
        switches = [0]*n
        for j4 in links:
            (a, b) = j4
            c = nodes_copy[a] + nodes_copy[b]
            if c == -2:
                if random.random() <= p:
                    switches[a] += +1
                if random.random() <= p:
                    switches[b] += +1
            elif c == 0:
                if random.random() <= p:
                    if nodes_copy[a] > 0:
                        switches[a] += 1
                    else:
                        switches[b] += 1
        k = 0
        for j5 in nodes_copy:
            if j5 < 0:
                if random.random() <= q:
                    nodes_copy[k] = -nodes_copy[k]
            k += 1
        k2 = 0
        for j6 in switches:
            if j6 >= threshold[k2] :
                nodes_copy[k2] = -nodes_copy[k2]
            k2 += 1
        r += 1
        cp.append(nodes_copy.count(-1))
        if r > 3:
            if max(cp) - min(cp) <= 1:
                return [cp[-1]/n, r, cp_copy]
                break
            else:
                del cp[0]
                

n = 10000 # 네트워크에 있는 노드의 수
links = [] # 네트워크상의 모든 링크들
nodes = [] # 감염 여부를 저장(-1 감염, 1 정상)
neighbors = [0]*n # 각 노드가 가진 이웃 노드의 수
for i in range(5*n):
    ERmodel(n)
for i2 in range(n):
    if random.random() <= 0.01:
        nodes.append(-1)
    else:
        nodes.append(1)


num_cores = mp.cpu_count()

plt.figure(figsize=(12, 14))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
markershape = ['s', 'o', '^', 'v', 'D']
markercolor = ['r', 'g', 'b', 'm', 'y']


m = 50
p_array = np.linspace(0, 1, num=m+1) # 감염된 노드가 링크된 정상 노드를 감염시킬 확률
q = 1 # 감염된 노드가 정상으로 회복될 확률
F = [3] # 네트워크 내 complex nodes 의 비율(나중에 최댓값을 1로 만들어준다.)
for i3 in F:
    
    threshold = [] # 각 노드의 역치값을 저장
    for i4 in range(n):
        if random.random() <= (i3/4): # 비율의 최댓값을 1로 만들어 줌
            threshold.append(neighbors[i4]/2)
        else:
            threshold.append(1)

    rate = []
    rlist = []
    cp_copy = []
    simple_rate = [0]*(m+1)
    complex_rate = [0]*(m+1)
    runs = 1
    for z in range(0, runs):
        pool = mp.Pool(num_cores)
        rate_r = list(pool.map(CP, p_array))
        for z2 in rate_r:
            rate.append(z2[0])
            rlist.append(z2[1])
            cp_copy.append(z2[2])
        '''
        if i3 == 2:
            number_of_simple = threshold.count(1)
            number_of_complex = n - number_of_simple
            adopted_simple = 0
            adopted_complex = 0
            for j2 in range(n):
                if nodes_copy[j2] == -1:
                    if threshold[j2] == 1:
                        adopted_simple += 1
                    else:
                        adopted_complex += 1
            simple_rate[j] += (adopted_simple/number_of_simple)
            complex_rate[j] += (adopted_complex/number_of_complex)
            ''' # 일단 보류

    print(rate)
    print(rlist)
    print(cp_copy)
    rate2 = []
    for y in rate:
        rate2.append(y/runs)

    if i3 == 0:
        ax1.scatter(p_array, rate2, s = 50, marker=markershape[i3], facecolors='none', edgecolors=markercolor[i3], label='p=0')
        with open("plot1.txt", "w") as f:
            f.write('p=0'+'\n')
            f.write(str(rate2)+'\n')
    elif i3 != 4:
        ax1.scatter(p_array, rate2, s = 50, marker=markershape[i3], facecolors='none', edgecolors=markercolor[i3], label=str(i3)+'/4')
        with open("plot1.txt", "a") as f:
            f.write('p='+str(i3)+'/4'+'\n')
            f.write(str(rate2)+'\n')
            f.write(str(rlist)+'\n')
    else:
        ax1.scatter(p_array, rate2, s = 50, marker=markershape[i3], facecolors='none', edgecolors=markercolor[i3], label='1')
        with open("plot1.txt", "a") as f:
            f.write('p=1'+'\n')
            f.write(str(rate2)+'\n')
    '''
    if i3 == 2:
        simple_rate2 = []
        complex_rate2 = []
        for y2 in simple_rate:
            simple_rate2.append(y2/runs)
        for y3 in complex_rate:
            complex_rate2.append(y3/runs)
        ax2.scatter(p_array, simple_rate2, s = 50, marker='s', facecolors='none', edgecolors='r', label='$R_S$')
        ax2.scatter(p_array, complex_rate2, s = 50, marker='o', facecolors='none', edgecolors='g', label='$R_C$')
        with open("plot2.txt", "w") as f:
            f.write('simple_rate'+'\n')
            f.write(str(simple_rate2)+'\n')
            f.write('complex_rate'+'\n')
            f.write(str(complex_rate2)+'\n')
            '''
    
ax1.set_xlabel(r'$\lambda$', fontsize=21)
ax1.set_ylabel('\n'+'R', fontsize=21)
ax1.tick_params(labelsize=18)
ax1.legend(fontsize=18)
ax2.set_xlabel(r'$\lambda$', fontsize=21)
ax2.set_ylabel('\n'+'R', fontsize=21)
ax2.tick_params(labelsize=18)
ax2.legend(fontsize=18)
plt.savefig('result.png')
