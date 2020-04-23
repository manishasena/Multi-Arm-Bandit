
# Implementation of MAB with different acquisition functions

#### USER INPUTS ####

# Number of arms
n = 5

# Variance
variance = 1

# Acquisition function
acq_func = "UCB1"
acq_func = "UCB1_Normal"

# Sample Budget
B = 50

# Sampling method
#sample_method = "Uniform"
sample_method = "Adaptive"

# Threshold for prediction
mean_threshold = 0.5

# False Discovery Rate
delta = 0.05

# TPR
TPR = True

# Coefficient in front of confidence function
c = 4

# Initial sampling
initial_loop = 2

######################

import numpy as np
import matplotlib.pyplot as plt

# CLASS: BANDIT to store information about each bandit in the game
class Bandit:

    def __init__(self,n,bandit_number):

            self.true_mean = (bandit_number)/n
            self.true_variance = variance

            self.sample_mean = 0
            self.bandit_trial = 0
            
            self.q = 0


    def Sample_Bandit(self):
        
        # Update number of trials of bandit
        self.bandit_trial += 1

        # Pull arm and return reward
        reward = np.random.normal(self.true_mean, self.true_variance,1)

        # Update bandit sample mean
        self.sample_mean = self.sample_mean + (reward - self.sample_mean)/self.bandit_trial
        
        # Updated sum of squared rewards
        self.q = self.q + reward**2

        return reward


# CLASS: GAME to store results of game (made up of bandits)
class Game:

    def __init__(self,n,acq_func,delta,TPR):
        
        self.n = n
        self.total_trials = 0
        self.St = {}
        
        if TPR == True:
            e_t = 1
            v_t = 1
            
            
        self.bandit_dictionary = {}

        # Create each bandit 
        for i in range(self.n):
            self.bandit_dictionary[str(i+1)] = {}
            self.bandit_dictionary[str(i+1)] = Bandit(self.n,i+1)
        
        # Conduct and initial sample from each bandit
        for il in range(initial_loop):
            for i in range(self.n):
                self.bandit_dictionary[str(i+1)].Sample_Bandit()
                self.total_trials += 1

        # Sample according to sampling method
        if sample_method == "Uniform":
            times = int(B/self.n)
            for j in range(times):
                for i in range(self.n):
                    self.bandit_dictionary[str(i+1)].Sample_Bandit()
                    self.total_trials += 1
        else:
            while self.total_trials < B:
                
                #Select next arm to sample
                acq_value = -1000
                arm_next = 1
                
                for i in range(self.n):
                    if (i not in self.St):

                        bandit_info = self.bandit_dictionary[str(i+1)]
                        
                        I =  bandit_info.sample_mean + Game.Confidence_Interval(self,bandit_info,delta/e_t,c,acq_func)
                        if I > acq_value:
                            acq_value = I
                            arm_next = i+1
                
                # Pull the selected arm
                self.bandit_dictionary[str(arm_next)].Sample_Bandit()
                self.total_trials += 1
                
                # Update St
                self = Game.Update_St(self,delta,acq_func)
                    
    
    def Confidence_Interval(self,bandit_info,d,c,acq_func):
        
        t = bandit_info.bandit_trial
        mean = bandit_info.sample_mean
        
        if acq_func == "UCB1":
            
            Confidence_value = np.sqrt((c*np.log(np.log2(2*t)/d))/(t))
            
        elif acq_func == "UCB1_Normal":
            
            q = bandit_info.q
            term1 = (np.sum(q) - t*mean**2)/(t-1)
            term2 = np.log(self.total_trials - 1)/t
            Confidence_value = np.sqrt(16*term1*term2)
        
        return Confidence_value
    
    def Update_St(self, delta,acq_func):
        
        # Apply Benjamini-Hochberg
        delta_dash = delta/(6.4*np.log(36/delta))
        delta_dash = delta
        
        # Discover St set
        K_LENGTH = 0
        for k in range(self.n):
            
            k_list = list()
            
            for i in range(self.n):
            
                bandit_info = self.bandit_dictionary[str(i+1)]
 
                confidence = Game.Confidence_Interval(self,bandit_info,delta_dash*(k+1)/self.n,c,acq_func)
                I = bandit_info.sample_mean - confidence
                
                    
                if (I >= mean_threshold):
                    k_list.append((i+1))

            len_k = len(k_list)
            if (len_k > K_LENGTH) and (len_k >= (k+1)):
                self.St = k_list
                #print(k)
            
        return self

G = Game(n,acq_func,delta,TPR)

for i in range(n):
    print(i+1)
    print(['True mean:', G.bandit_dictionary[str(i+1)].true_mean])
    print(['Sample mean:', G.bandit_dictionary[str(i+1)].sample_mean])
    print(['Trials:', G.bandit_dictionary[str(i+1)].bandit_trial])
    
print(G.St)
t = np.linspace(1,B)
y = 8*np.log(t)

plt.plot(t,y)
plt.show()