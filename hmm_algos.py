#HMM based algorithms

import numpy as np
import get_data as gd
#BKT

#Returns matrix of alphas
def alpha_forward(y, pi, A, B):
    ''' P(x_t, y_(1:t))'''
    T = len(y)
    alpha = np.zeros((T+1,2))
    
    for i in range(0,2):
        alpha[0,i] = pi[i]
        
    for t in range(1, T+1):
        for j in range(0,2):
            alpha[t, j] = B[j,y[t-1]]*(A[0,j]*(alpha[t-1,0])+A[1,j]*(alpha[t-1,1]))
    return(alpha)

#Returns likelihood of observed sequence    
def likelihood_obs(y, pi, A, B):
    '''P(y_(1:T))'''
    T = len(y)
    alpha = alpha_forward(y, pi, A, B)
    lik = np.sum(alpha[T,:])
    return(lik)
 
    
#Returns filter matrix    
def filtering(y, pi, A, B):
    '''P(x_t|y_(1:t))'''
    alpha = alpha_forward(y, pi, A, B)
    filt = (alpha.T/(np.sum(alpha, axis = 1))).T
    return(filt)

#Returns matrix of betas    
def beta_backward(y, pi, A, B):
    '''P(y_(t+1:T)|x_t)'''
    T = len(y)
    beta = np.zeros((T+1,2))
    
    for i in range(0,2):
        beta[T,i] = 1
        
    for t in range(1, T):
        for j in range(0, 2):
            beta[T-t, j] = beta[T-t+1,0]*B[0,y[T-t]]*A[j,0] + beta[T-t+1,1]*B[1,y[T-t]]*A[j,1]
            
    return(beta)

  
#Returns matrix of smoothing probabilities    
def smoothing(y, pi, A, B):
    ''' P(x_t|y_(1:T)) '''
    alpha = alpha_forward(y, pi, A, B)
    beta = beta_backward(y, pi, A, B)
    ab = alpha*beta
    smooth = (ab.T/(np.sum(ab, axis = 1))).T
    return(smooth)
    
#Two state smoothing
def two_smoothing(y, pi, A, B):
    
    ''' P(x_(t-1:t)|y_(1:T)) '''
    T = len(y)
    D = np.zeros((T,4))
    alpha = alpha_forward(y, pi, A, B)
    beta = beta_backward(y, pi, A, B)
    for t in range(1,T+1):
        k = 0
        for i in range(0,2):
            for j in range(0,2):
                D[t-1, k] = beta[t, j]*B[j, y[t-1]]*A[i,j]*alpha[t-1,i]
                k = k+1
                
           
    D = (D.T/(np.sum(D, axis = 1))).T
    return(D)
    
#Returns most likely state path of latent variables    
def viterbi(y, pi, A, B):
    ''' most likely state path x_(0:T) '''
    T = len(y)
    M = np.zeros((T,2))
    x_hat = np.zeros(T+1)
    
    for i in range(0,2):
        M[T-1,i]  = 1
        
    for t in range(2, T+1):
        for i in range(0,2):
            M[T-t, i] = max(B[0, y[T-t+1]]*A[i, 0]*M[T-t+1,0], B[1, y[T-t+1]]*A[i, 1]*M[T-t+1,1])
 
    x = [M[0,0]*pi[0], M[0,1]*pi[1]]
    x_hat[0] = x.index(max(x))
    
    for t in range(1, T):
        x = [M[t, 0]*B[0,y[t]]*A[int(x_hat[t-1]), 0], M[t, 1]*B[1,y[t]]*A[int(x_hat[t-1]), 1]]
        x_hat[t] = x.index(max(x))
    
    if(x_hat[T-1] == 1):
        x_hat[T] = 1

    return(x_hat)
    
#Construct matrix of parameters given parameters
def get_matrices(pL0, pG, pS, pT):
    pi = np.array([1-pL0, pL0])
    A = np.array([[1-pT, pT], [0, 1]])   
    B = np.array([[1-pG, pG], [pS, 1-pS]])
    L = [pi, A, B]
    return(L)