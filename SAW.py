import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
from tqdm import tqdm

random.seed(7)

class Lattice():
    
    def __init__(self, n=10) -> None:
        self.n = n
        self.start = (0,0)
        self.record = []
        self.longest_trail = []
        self.record_at_n_n = []
        self.count_length = defaultdict(default=0)
        self.count_length_n_n = defaultdict(default=0)
        self.longest_trail_n_n = []

    # def walks(self, M, mode='random'):
    #     if mode == 'random':
    #         self.random_walks(M)
    #         self.print_result('random_walk')
    #         self.plot_hist('random_walk')
    #         self.plot_longest_trail('random_walk')
    #         self.random_walks_to_n_n(M)
    #         self.print_result_n_n('random_walk')
    #         self.plot_hist_n_n('random_walk')
    #         self.plot_longest_trail_n_n('random_walk')

   
    def random_walks(self, M, id):
        'walk M times'
        exps = {1:self.random_walk_1, 2:self.random_walk_2, \
                3:self.random_walk_3}
        for m in tqdm(range(M)):
            exps[id]()                 
        self.estimate =  np.cumsum(np.array(self.record)) / np.arange(1,len(self.record)+1)

    def random_walks_to_n_n(self, M):
        'try M times'
        i_s = 0
        for _ in tqdm(range(M)):
            i=1
            while True:
                flag = self.random_walk_to_n_n(i) 
                i += 1
                if flag:
                    i_s += i - 1
                    break

        u = M/i_s # k = 1/u
        k = u             
                    


        self.record_at_n_n = np.array(self.record_at_n_n)
        # np.save('random_walk_n_n', record_array) 
        # all_weights = len(self.record_at_n_n) / M
        # real_weights = record_array / all_weights
        self.estimate_n_n = np.cumsum(1/(self.record_at_n_n/k)) / np.arange(1,len(self.record_at_n_n)+1)
        for i in range(120):
            self.count_length_n_n[i] = self.count_length_n_n.get(i,0)*k

 
            
        

    
                
    
    def random_walk_to_n_n(self, k):
        'Keep walking until arriving at (n,n)'
        # current = (self.n, 0)
        # trail = [(i, 0) for i in range(self.n+1)]
        current = self.start
        trail = [current]
        prob = 1
        i = 0
        while True:
            a, b = current
            choices = [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
            choices = [choice for choice in choices if choice not in trail 
                    and 0 <=choice[0]<= self.n and 0 <= choice[1] <= self.n]
            nt = len(choices)
            if nt == 0:
                return False
            else:
                prob *= 1. / nt
                next = random.choice(choices) # Move on
                trail.append(next)
                current = next
                i += 1 # step ++ 
                
                
                if current[0] == self.n and current[1] == self.n:
                    self.record_at_n_n.append(prob)
                    if i > len(self.longest_trail_n_n):
                        self.longest_trail_n_n = trail
                    self.count_length_n_n[i] = self.count_length_n_n.get(i,0) + 1/prob 
                    return True
                
    def random_walk_2(self):
        current = self.start
        trail = [current]
        prob = 1
        i = 0
        eps=0.1
        while True:
            a, b = current
            choices = [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
            choices = [choice for choice in choices if choice not in trail 
                    and 0 <=choice[0]<= self.n and 0 <= choice[1] <= self.n]
            nt = len(choices)
            if nt == 0:
                self.record.append(1./prob)
                break
            else:
                a = np.random.random(1)
                if a < eps:
                    prob *= eps
                    self.record.append(1./prob)
                    break
                else:
                    prob *= (1-eps)*1. / (nt)
                    #choices = choices + [(a,b)]
                    next = random.choice(choices)
                    # if next[0] == a and next[1] == b:
                    #     self.record.append(1./prob)
                    #     break
                    trail.append(next)
                    current = next
                    i += 1 # step ++ 
        if i > len(self.longest_trail):
            self.longest_trail = trail
        if i in self.count_length:
            self.count_length[i] += 1/prob
        else:
            self.count_length[i] = 1/prob

    def branch(self, current, trail, i, prob, n_branch=5):
        record = []
        count_length = []
        for _ in range(n_branch):
            trail_i = trail
            current_i = current
            i_i = i
            prob_i = prob * 1/n_branch
            
            while True:         
                a, b = current_i
                choices = [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
                choices = [choice for choice in choices if choice not in trail_i 
                        and 0 <=choice[0]<= self.n and 0 <= choice[1] <= self.n]
                nt = len(choices)
                if nt == 0:
                    record.append(1./prob_i)
                    count_length.append(i_i)
                    break
                else:    
                    prob_i *= 1. / (nt)
                    next_i = random.choice(choices)
                    trail_i.append(next_i)
                    current_i = next_i
                    i_i += 1 # step ++ 
            if i_i > len(self.longest_trail):
                self.longest_trail = trail_i
            self.count_length[i_i] = self.count_length.get(i_i, 0) + 1/prob_i

        prob_avg_inv = np.mean(np.array(record))
        self.record.append(prob_avg_inv)
        
        # i_avg = int(np.mean(np.array(count_length)))
        # if i_avg in self.count_length:
        #     self.count_length[i_avg] += prob_avg_inv
        # else:
        #     self.count_length[i_avg] = prob_avg_inv
        # for idx, length in enumerate(count_length):
        #     self.count_length[length] = self.count_length.get(length, 0) +  5*1/record[idx]



            


    def random_walk_3(self):
        current = self.start
        trail = [current]
        prob = 1
        i = 0

        while True:
            if i >= 50:
                self.branch(current, trail, i, prob)
                return
            a, b = current
            choices = [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
            choices = [choice for choice in choices if choice not in trail 
                    and 0 <=choice[0]<= self.n and 0 <= choice[1] <= self.n]
            nt = len(choices)
            if nt == 0:
                self.record.append(1./prob)
                break
            else:
                
                prob *= 1. / (nt)
                next = random.choice(choices)
                trail.append(next)
                current = next
                i += 1 # step ++ 
        if i > len(self.longest_trail):
            self.longest_trail = trail
        if i in self.count_length:
            self.count_length[i] += 1/prob
        else:
            self.count_length[i] = 1/prob

    def random_walk_1(self):
        current = self.start
        trail = [current]
        prob = 1
        i = 0
        while True:
            a, b = current
            choices = [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
            choices = [choice for choice in choices if choice not in trail 
                    and 0 <=choice[0]<= self.n and 0 <= choice[1] <= self.n]
            nt = len(choices)
            if nt == 0:
                self.record.append(1./prob)
                break
            else:
                prob *= 1. / (nt)
                #choices = choices + [(a,b)]
                next = random.choice(choices)
                # if next[0] == a and next[1] == b:
                #     self.record.append(1./prob)
                #     break
                trail.append(next)
                current = next
                i += 1 # step ++ 
        if i > len(self.longest_trail):
            self.longest_trail = trail
        if i in self.count_length:
            self.count_length[i] += 1/prob
        else:
            self.count_length[i] = 1/prob
    
    def plot_hist(self, name):
        plt.figure()
        counts = [self.count_length.get(i, 0) for i in range(120)]
        counts = np.array(counts) 
        counts = counts/counts.sum() * self.estimate[-1]
        plt.plot(counts)
        plt.title('histgram of length N')
        plt.savefig(f'{name}_histgram')
        plt.show()

    def plot_hist_n_n(self, name):
        plt.figure()
        counts = [self.count_length_n_n.get(i, 0) for i in range(120)]
        counts = np.array(counts) 
        counts = counts/counts.sum() * self.estimate_n_n[-1]
        plt.plot(counts)
        plt.title('histgram of length N')
        plt.savefig(f'{name}_histgram_n_n')
        plt.show()



            


    def print_result(self, name):
        plt.figure()
        plt.loglog(range(len(self.estimate)), self.estimate)
        plt.title(f'log(K)~log(m),{self.start} estimate:{self.estimate[-1]:.3e}')
        plt.savefig(f'{name}_loglog.png')
        plt.show()

    def print_result_n_n(self, name):
        plt.figure()
        plt.loglog(range(len(self.estimate_n_n)), self.estimate_n_n)
        plt.title(f'log(K)~log(m), estimate:{self.estimate_n_n[-1]:.3e}')
        plt.savefig(f'{name}_loglog_n_n.png')
        plt.show()


    def plot_longest_trail(self,name):
        plt.figure()
        # Create a 10x10 grid of zeros
        grid = [[0 for _ in range(11)] for _ in range(11)]
        

        # Plot the horizontal lines
        for i in range(11):
            plt.plot([0, 10], [i, i], color='black',alpha=0.5)

        # Plot the vertical lines
        for j in range(11):
            plt.plot([j, j], [0, 10], color='black',alpha=0.5)

        plt.xticks(range(11))
        plt.yticks(range(10,-1,-1))
        # Show the plot
        plt.imshow(grid, cmap='binary')
        plt.box(False)
        plt.title(f'longest trail, step: {len(self.longest_trail)}')
        plt.plot([x for x, y in self.longest_trail], [y for x, y in self.longest_trail], color='blue')
        plt.savefig(f'{name}_longest_trail.png')
        plt.show()
    
    def plot_longest_trail_n_n(self,name):
        # Create a 10x10 grid of zeros
        grid = [[0 for _ in range(11)] for _ in range(11)]
        

        plt.figure()
        # Plot the horizontal lines
        for i in range(11):
            plt.plot([0, 10], [i, i], color='black',alpha=0.5)

        # Plot the vertical lines
        for j in range(11):
            plt.plot([j, j], [0, 10], color='black',alpha=0.5)

        
        # Show the plot
        plt.imshow(grid, cmap='binary')
        plt.xticks(range(11))
        plt.yticks(range(11))
        plt.box(False)
        plt.title(f'longest trail ending at (n,n), step: {len(self.longest_trail_n_n)}')
        plt.plot([x for x, y in self.longest_trail_n_n], [y for x, y in self.longest_trail_n_n], color='blue')
        plt.savefig(f'{name}_longest_trail_n_n.png')
        plt.show()
        
def sanity_check():
    for i in range(1,3):
        lattice = Lattice(n=i)
        lattice.random_walks_to_n_n(int(1e5))
        print(f'n={i}, estimated K of SAWS ending at (n,n) is {lattice.estimate_n_n[-1]}')

random.seed(42)


M = int(1e7)

lattice = Lattice(n=10)
# K = lattice.start_different(M)
# print('estimate K = ', K)

for id in [1,2,3]:
    lattice.random_walks(M, id)
    lattice.print_result('random_walk_design_{}'.format(id))
    lattice.plot_hist('random_walk_design_{}'.format(id))
    lattice.plot_longest_trail('random_walk_design_{}'.format(id))

lattice.random_walks_to_n_n(M)
lattice.print_result_n_n('random_walk_')
lattice.plot_hist_n_n('random_walk_')
lattice.plot_longest_trail_n_n('random_walk_')





#sys.exit()

# 1.56875e24
# 4e23

# 6e12 1e9

        




