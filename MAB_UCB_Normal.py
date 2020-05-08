import numpy as np
import matplotlib.pyplot as plt
import math

# Number of arms
n = 20

# Variance
variance = 0.3

# Acquisition function
acq_func = "UCB1_Normal_delta"

# Sample Budget
B = 400 #600

# Sampling method
#sample_method = "Uniform"
sample_method = "Adaptive"

# Threshold for prediction
mean_threshold = [0.1,0.3, 0.5,0.7,0.9]

# TPR
TPR = True

# Coefficient in front of confidence function
c = 4

# Visualise plots
visualise = False

# Plot out every 'how  many' iterations
plot_n = 500

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

    def __init__(self,n,acq_func,delta,TPR,mean_threshold):
        
        self.n = n
        self.total_trials = 0
        self.St = {}
        self.FDR = list()
        self.TPR = list()
        
        if TPR == True:
            e_t = 1
            v_t = 1
            
            
        self.H0 = list()
        self.H1 = list()
        self.bandit_dictionary = {}

        # Create each bandit 
        for i in range(self.n):
            self.bandit_dictionary[str(i+1)] = {}
            self.bandit_dictionary[str(i+1)] = Bandit(self.n,i+1)
            
            if self.bandit_dictionary[str(i+1)].true_mean > mean_threshold:
                self.H1.append(i+1)
            else:
                self.H0.append(i+1)
        
        # Conduct and initial sample from each bandit
        
        for il in range(initial_loop):
            for i in range(self.n):
                self.bandit_dictionary[str(i+1)].Sample_Bandit()
                self.total_trials += 1
            self.FDR.append(Game.FDR_function(self))
            self.TPR.append(Game.TPR_function(self))
            
        #Plot distribution
        
        #print('trial' + str(self.total_trials))
        y = np.zeros([1,self.n]).reshape(-1,)
        no_pulls = np.zeros([1,self.n]).reshape(-1,)
        for i in range(self.n):
            y[i] = self.bandit_dictionary[str(i+1)].sample_mean
            no_pulls[i] = self.bandit_dictionary[str(i+1)].bandit_trial 
        
        if visualise == True:
            print('trial: ' + str(self.total_trials))
            print('Number of arm pulls: ' + str(no_pulls))
        
            plt = Game.Visualise(self,y,delta,mean_threshold,acq_func)
            plt.show()

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
                    if ((i+1) not in self.St):

                        bandit_info = self.bandit_dictionary[str(i+1)]
                        
                        I =  bandit_info.sample_mean + Game.Confidence_Interval(self,bandit_info,delta/e_t,c,acq_func)
                        if I > acq_value:
                            acq_value = I
                            arm_next = i+1
                
                # Pull the selected arm
                self.bandit_dictionary[str(arm_next)].Sample_Bandit()
                self.total_trials += 1
                
                # Update St
                self = Game.Update_St(self,delta,acq_func,mean_threshold)
                
                self.FDR.append(Game.FDR_function(self))
                self.TPR.append(Game.TPR_function(self))
                
                #Plot distribution
                if self.total_trials % plot_n == 0:
                    print('trial: ' + str(self.total_trials))
                    print(self.St)
                    y = np.zeros([1,self.n]).reshape(-1,)
                    no_pulls = np.zeros([1,self.n]).reshape(-1,)
                    for i in range(self.n):
                        y[i] = self.bandit_dictionary[str(i+1)].sample_mean
                        no_pulls[i] = self.bandit_dictionary[str(i+1)].bandit_trial 

                    print('Number of arm pulls: ' + str(no_pulls))
                    if visualise == True:
                        plt = Game.Visualise(self,y,delta,mean_threshold,acq_func)
                        plt.show()
                    
    
    def Visualise(self,mean,delta,mu_threshold,acq_func):
    
        plt.style.use('seaborn-whitegrid')
        x = np.linspace(1,self.n,self.n)
        color = ['g','r','b','k','y','m']
        ranges = np.linspace(n,0,n)
        
        if d_dash == True:
            delta_dash = delta/(6.4*np.log(36/delta))
            
            dy = np.zeros([1,self.n]).reshape(-1,)
            for k in ranges:
                k = int(k)
                for i in range(self.n):
                    dy[i] = Game.Confidence_Interval(self,self.bandit_dictionary[str(i+1)],delta_dash*(k+1)/self.n,c,acq_func)

                #plt.errorbar(x, mean, yerr=dy, fmt='o', color='black',ecolor=color[k], elinewidth=20-3*k, capsize=20-3*k)
                plt.errorbar(x, mean, yerr=dy, fmt='o', color='black',ecolor=color[k], elinewidth=3*k+1, capsize=3*k+1)

            plt.axhline(mu_threshold)
        
        else:
            delta_dash = delta
            
            dy = np.zeros([1,self.n]).reshape(-1,)

            for i in range(self.n):
                dy[i] = Game.Confidence_Interval(self,self.bandit_dictionary[str(i+1)],delta_dash,c,acq_func)

                plt.errorbar(x, mean, yerr=dy, fmt='o', color='black',ecolor=color[0], elinewidth=3*k+1, capsize=3*k+1)

            plt.axhline(mu_threshold)

        return plt
    
    def FDR_function(self):
        
        numerator = 0
        for i in self.St:
            if i in self.H0:
                numerator += 1
        
        if len(self.St) != 0:
            FDR = numerator/len(self.St)
        else:
            FDR = 0
            
        return FDR

    def TPR_function(self):
        
        numerator = 0
        for i in self.St:
            if i in self.H1:
                numerator += 1
        
        if len(self.St) != 0:
            TPR = numerator/len(self.H1)
        else:
            TPR = 0
            
        return TPR
    
    def Confidence_Interval(self,bandit_info,d,c,acq_func):
        
        t = bandit_info.bandit_trial
        mean = bandit_info.sample_mean
        
        if acq_func == "UCB1":
            Confidence_value = np.sqrt((c*np.log(np.log2(2*t)/d))/(t))
            
        elif acq_func == "d-PAC":
            e = np.exp(1)
            val = 2*np.log(1/d) + 6*np.log(np.log(1/d)) + 3*np.log(np.log(e*t/2))
            Confidence_value = np.sqrt(val/t)
            
        elif acq_func == "UCB1_Normal":
            
            q = bandit_info.q
            term1 = (np.sum(q) - t*mean**2)/(t-1)
            term2 = np.log(self.total_trials - 1)/t
            Confidence_value = np.sqrt(16*term1*term2)
            
        elif acq_func == "UCB1_Normal_delta":
            
            q = bandit_info.q
            term1 = (np.sum(q) - t*mean**2)/(t-1)
            #term2 = np.log(d**(-1/4) - 1)/t
            term2 = np.log(d**(-1/4))/t
            Confidence_value = np.sqrt(16*term1*term2)

        elif acq_func == "UCB1_derived":
            Confidence_value = np.sqrt((-1*np.log(d))/(2*t))
        
        return Confidence_value
    
    def Update_St(self, delta,acq_func,mean_threshold):
        
        # Apply Benjamini-Hochberg
        if d_dash == True:
            delta_dash = delta/(6.4*np.log(36/delta))
            
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
        else:
            delta_dash = delta
            
            k_list = list()

            for i in range(self.n):

                bandit_info = self.bandit_dictionary[str(i+1)]

                confidence = Game.Confidence_Interval(self,bandit_info,delta_dash,c,acq_func)
                I = bandit_info.sample_mean - confidence

                if (I >= mean_threshold):
                    k_list.append((i+1))
                    
            self.St = k_list
            #print(k)
            
        return self

acq_func = "UCB1_Normal_delta"

# False Discovery Rate
delta = [0.05,0.15,0.3,0.45,0.6]
d_dash = False
no_trials = 50

fig, axs1 = plt.subplots(len(delta), len(mean_threshold))
fig, axs2 = plt.subplots(len(delta), len(mean_threshold))

axs1 = axs1.ravel()
axs2 = axs2.ravel()

pos = 0
for d in range(len(delta)):

    print(d)
    
    # Initial sampling
    initial_loop = math.ceil(8*np.log(delta[d]**(-1/4)))
    
    FDR_holder = np.zeros([no_trials,(B-(n-1)*initial_loop)])
    TPR_holder = np.zeros([no_trials,(B-(n-1)*initial_loop)])

    for m in range(len(mean_threshold)):
        for i in range(no_trials):

            G = Game(n,acq_func,delta[d],TPR,mean_threshold[m])
            FDR_holder[i,:] = G.FDR
            TPR_holder[i,:] = G.TPR
            #print(G.St)

        #print("Arm's that are above threshold:")
        #print(G.St)
        
        
        axs1[pos].plot(np.mean(FDR_holder,axis = 0))
        axs1[pos].set_title('Delta: ' + str(delta[d]) + ' mu_o: ' + str(mean_threshold[m]))
        
        axs2[pos].plot(np.mean(TPR_holder,axis = 0))
        axs2[pos].set_title('Delta: ' + str(delta[d]) + ' mu_o: ' + str(mean_threshold[m]))
        pos += 1

        #axs1[d, m].plot(np.mean(FDR_holder,axis = 0))
        #axs1[d, m].set_title('FDR')
        #plt.xlabel('Experiments')
        #plt.ylabel('FDR')
        #plt.show()

        #ax2.subplot(d+1,m+1,1)
        #axs2[d, m].plot(np.mean(TPR_holder,axis = 0))
        #axs2[d, m].set_title('TPR')
        #ax2.plot(np.mean(TPR_holder,axis = 0))
        #plt.title('TPR')
        #plt.xlabel('Experiments')
        #plt.ylabel('TPR')
plt.show()
        

