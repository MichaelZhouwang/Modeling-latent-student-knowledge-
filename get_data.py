# Generating student data based on BKT assumptions
import numpy as np
# For a single student, and a particular kc
#Generates synthetic data - sequence(s) of observations with specified  params        
def generate_responses( numQuestions, pL0, pT, pG, pS):
    answers = []
    for i in range(0,numQuestions):
        pC = pL0*(1-pS) + (1-pL0)*pG
        cur_answer = np.random.binomial(1, pC)
        answers.append(int(cur_answer))
        
        if i == numQuestions-1: break
        
        #Posterior
        if (cur_answer == 1):
            pL0 = (pL0*(1-pS))/(pL0*(1-pS) + (1-pL0)*pG)
        else:
            pL0 = (pL0*pS)/(pL0*pS+(1-pL0)*(1-pG))
        #Prior for next question    
        pL0 = pL0 + (1-pL0)*pT
    return(answers)

