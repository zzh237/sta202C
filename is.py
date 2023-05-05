import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

def pi(x,y):
    return 1/(2*np.pi) * np.exp(-1/2*((x-2)**2 + (y-2)**2))

def g(x,y, sigma):
    return 1/(2*np.pi*sigma**2) * np.exp(-1/(2*sigma**2)*((x)**2 + (y)**2))

def compute_theta_1(M):
    samples = np.random.randn(2,M) + 2
    theta = np.sqrt(samples[0]**2 + samples[1]**2)
    #plt.figure()
    esimate = np.cumsum(theta)/np.arange(1,len(theta)+1)
    

    return esimate,theta

def compute_theta_2(M, sigma):
    samples = np.random.randn(2,M)*sigma
    pi_xy = pi(samples[0], samples[1])
    g_xy = g(samples[0], samples[1], sigma)
    theta = np.sqrt(samples[0]**2 + samples[1]**2)
    wi = pi_xy / g_xy
    objective = theta * wi

    # plt.figure()
    esimate = np.cumsum(objective)/np.arange(1,len(objective)+1)
    # plt.loglog(range(1, len(objective)+1), esimate)
    # plt.title(f'sigma={sigma}')
    # plt.savefig(f'sigma={sigma}.png')
    # plt.show()

    # compute ess*



    return esimate, objective, wi


def compute_ess(M, sigma=1):
    # _, theta = compute_theta_1(M)
    # v_pi_I = np.var(theta)/M

    _, theta2 ,wi= compute_theta_2(M, sigma)
    s_w2 = 1/(M-1) * np.sum((wi - 1/M)**2)
    ess = M/(1+s_w2)
    return ess

def compute_ess_star(M, sigma=1):
    _, theta = compute_theta_1(M)
    v_pi_I = np.var(theta)
    I = []
    for n in tqdm(range(10000)):
        _, objective, wi = compute_theta_2(M, sigma)
        I.append(objective.sum()/wi.sum())
    v_g_I = np.var(np.array(I))
    return v_pi_I/v_g_I





np.random.seed(42)
ess_star_1 = []
ess_2 = []
ess_star_2 = []
ess_3 = []
ess_star_3 = []
for M in [10, 100, 1000, 10000]:
    ess_star_1.append(M)
    ess_2.append(compute_ess(M,sigma=1))
    ess_3.append(compute_ess(M,sigma=4))
    ess_star_2.append(compute_ess_star(M,sigma=1))
    ess_star_3.append(compute_ess_star(M,sigma=4))
    
    
    # print(f'M={M}, sigma=1, ess_star:',  compute_ess_star(M, 1))
    # print(f'M={M}, sigma=1, ess:', )
    # print(f'M={M}, sigma=4, ess_star:',  compute_ess_star(M, 4))
    # print(f'M={M}, sigma=4, ess:', compute_ess(M,sigma=4))
plt.figure()
plt.loglog([10, 100, 1000, 10000], ess_star_1, label='theta_1')
plt.loglog([10, 100, 1000, 10000], ess_star_2, label='theta_2_ess_star')
plt.loglog([10, 100, 1000, 10000], ess_star_3, label='theta_3_ess_star')
plt.loglog([10, 100, 1000, 10000], ess_2, label='theta_2_ess')
plt.loglog([10, 100, 1000, 10000], ess_3, label='theta_3_ess')
plt.title('ess comparision')
plt.legend()
plt.savefig('loglog ess_comparision.png')
plt.show()

plt.figure()
plt.loglog([10, 100, 1000, 10000], [1,1,1,1], label='theta_1')
plt.loglog([10, 100, 1000, 10000], np.array(ess_star_2)/np.array(ess_2), label='theta_2')
plt.loglog([10, 100, 1000, 10000], np.array(ess_3)/np.array(ess_star_3), label='theta_3')
plt.title('ess ratio comparision')
plt.legend()
plt.savefig('loglog ess ratio comparision.png')
plt.show()



# theta_1,_ = compute_theta_1(M)
# theta_2,_,_ = compute_theta_2(M,1)
# theta_3,_,_ = compute_theta_2(M,4)
# print('theta_1:', compute_theta_1(M)[-1])
# print('theta_2:', compute_theta_2(M, 1)[-1])
# print('theta_3:', compute_theta_2(M, 4)[-1])





# plt.figure()
# plt.loglog(range(1, M+1), theta_1, 'r', label='theta_1' )
# plt.loglog(range(1, M+1), theta_2, 'b', label='theta_2' )
# plt.loglog(range(1, M+1), theta_3, 'g', label='theta_3' )
# plt.title('comparison')
# plt.legend()
# plt.savefig('comparison.png')
# plt.show()
