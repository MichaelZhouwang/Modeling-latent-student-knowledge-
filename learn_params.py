import numpy as np 
import hmm_algos as hm
import get_data as gd

#Learning parameters using EM 
def baum_welch(y, threshold):
    #Randomly select initial params
    pG_0 = np.random.uniform(low=0.0, high=0.3, size=1)
    pS_0 = np.random.uniform(low=0.0, high=0.1, size=1)
    pT_0 = np.random.uniform(low=0.0, high=1, size=1)
    pL0_0 = np.random.choice(low=0.0, high=1, size=1)
    pi_0, A_0, B_0 = hm.get_matrices(pL0_0, pG_0, pS_0, pT_0)
    theta_0 = [pi_0, A_0, B_0]
    
    def update(theta_0):
        #Do EM updates here
        T = []
        # pi
        sm = hm.smoothing(y, theta_0[0], theta_0[1], theta_0[2])
        pL0 = sm[1,1] #confirm if sm[0,1] or sm[1,1]
        pi = np.array([1-pL0, pL0])
        T[0] = pi
        # pT 
        ds = hm.two_smoothing(y, theta_0[0], theta_0[1], theta_0[2])
        pT = np.sum(ds[:,1])/(np.sum(ds[:,1] + np.sum(ds[:,0])))
        A = np.array([[1-pT,pT], [0,1]])
        T[1] = A
        # pG, pS
        sm0 = sm[1:(sm.shape[0] +1), 0]
        sm1 = sm[1:(sm.shape[0] +1), 1]
        pG = (sm0.dot(y))/(np.sum(sm0))
        pS = (sm1.dot(1-y))/(np.sum(sm1))
        B = np.array([[1-pG, pG], [pS, 1-pS]])
        T[2] = B
        return(T)
    
        
        
    theta = update(theta_0)
    pi = theta[0]
    A = theta[1]
    B = theta[2]
    
    while(np.absolute(np.log(hm.likelihood_obs(y, pi_0, A_0, B_0)) - np.log(hm.likelihood_obs(y, pi, A, B))) > threshold):
        theta_0 = theta
        pi_0 = pi
        A_0 = A
        B_0 = B
        theta = update(theta_0)
        pi = theta[0]
        A = theta[1]
        B = theta[2]
    
    return(theta)
            

#Inefficient grid search 


#Randomize the search
def rand_search(y,rate, combs, dmp, reg):
    
    d ={}
    Md = {}
    Mst = {}
    
    def local_rand_search(G_U, G_D, S_U, S_D, T_U, T_D, IT_U, IT_D,cnt):
        if(cnt==4):
            k= max(Md.keys(), key=(lambda k: Md[k]))
            v= Md[k]
            return({k:v})
            
        else:
            d_t = {}
            for i in range(0,combs):
                gg = np.random.unfiorm(low=G_D, high=G_U, size=1)
                ss = np.random.unfiorm(low=S_D, high=S_U, size=1)
                tt = np.random.unfiorm(low=T_D, high=T_U, size=1)
                itit = np.random.unfiorm(low=IT_D, high=IT_U, size=1)
                pi, A, B = hm.get_matrices(itit,gg,ss,tt)
                d_t.update({(gg,ss,tt,itit):np.log(hm.likelihood_obs(y, pi, A, B))})
            maxm = max(d_t.keys(), key=(lambda k: d_t[k]))
            el_max= d_t[maxm]
            GG, SS, TT, ITIT = maxm
            Md.update({maxm:el_max})
            GG_U = GG + (dmp**(cnt+1))*GG
            GG_D = GG - (dmp**(cnt+1))*GG
            SS_U = SS + (dmp**(cnt+1))*SS
            SS_D = SS - (dmp**(cnt+1))*SS
            TT_U = TT + (dmp**(cnt+1))*TT
            TT_D = TT - (dmp**(cnt+1))*TT
            ITIT_U = ITIT + (dmp**(cnt+1))*ITIT
            ITIT_D = ITIT - (dmp**(cnt+1))*ITIT
            cnt = cnt + 1
            Md.update(local_rand_search(GG_U,GG_D,SS_U,SS_D, TT_U, TT_D, ITIT_U, ITIT_D,cnt))
            
    for i in range(0, combs):
        g = np.random.unfiorm(low=0.0, high=0.3, size=1)
        s = np.random.unfiorm(low=0.0, high=0.1, size=1)
        t = np.random.unfiorm(low=0.0, high=1, size=1)
        it = np.random.unfiorm(low=0.0, high=1, size=1)
        pi, A, B = hm.get_matrices(it,g,s,t)
        d.update({(g,s,t,it):np.log(hm.likelihood_obs(y, pi, A, B))})
        
    for j in range(0, reg):
        maxm = max(d.keys(), key=(lambda k: d[k]))
        el_max= d[maxm]
        if(j==0): Mst.update({maxm:el_max})
        G, S, T, IT = maxm
        G_U = G + dmp*G
        G_D = G - dmp*G
        S_U = S + dmp*S
        S_D = S - dmp*S
        T_U = T + dmp*T
        T_D = T - dmp*T
        IT_U = IT + dmp*IT
        IT_D = IT - dmp*IT
        cnt = 1
        Mst.update(local_rand_search(G_U, G_D, S_U, S_D, T_U, T_D, IT_U, IT_D,cnt))
        del d[maxm]
        
    k= max(Mst.keys(), key=(lambda k: Mst[k]))
    v= Mst[k]
    return({k:v})
   