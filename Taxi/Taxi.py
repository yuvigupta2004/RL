import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(episodes,viewflag=False, training=True):
    
    
    environment = gym.make('Taxi-v3', render_mode='human' if viewflag else None)
    
    
    if (training):
        q_table=np.zeros((environment.observation_space.n, environment.action_space.n))    
    else:
        f=open('traineddata.pkl','rb')
        q_table = pickle.load(f)
        f.close()
        
    # declaring hyperparameters
    
    learning_rate = 0.9
    discount_factor = 0.9
    
    epsilon = 1
    epsilon_decay = 0.0001
    
    random_number_generator = np.random.default_rng()
    
    reward_of_every_episode = np.zeros(episodes)
    
    for i in range(episodes):
        
        print("Episode Number:",i)
        
        state = environment.reset()[0] 
        terminated = False
        maxmovesreached = False
    
    
        while (not terminated and not maxmovesreached):
            
            if training and random_number_generator.random() < epsilon:
                action = environment.action_space.sample() # this is a random action
            else:
                action = np.argmax(q_table[state,:])
                
                
                # probabilities = np.exp(q_table[state, :]) / np.sum(np.exp(q_table[state, :]))
                # action = np.random.choice(range(len(probabilities)), p=probabilities) # if not to be randomized then choose the action with highest reward
    
            # this chunk above is the epsilon greedy strategy
            
            
            newstate,reward,terminated,maxmovesreached,_ = environment.step(action) #this reutrns the newstate is after the action is made 

            
            if training:
                q_table[state,action] += learning_rate * ( reward + discount_factor * np.max(q_table[newstate,:]) - q_table[state,action])
                
            state = newstate
        

        epsilon = max(epsilon-epsilon_decay,0)  #just so epsilon doesnt become <0
        
        if (epsilon==0):
            
            learning_rate = 0.0001  #make it practically 0 once training is over
        
       
        reward_of_every_episode[i]=reward
    
    
    
    environment.close()
    
    
    # training is over
    
    
    rewards_for_window = np.zeros(episodes)
    
    for episode in range(episodes):
        
        rewards_for_window[episode] = np.sum(reward_of_every_episode[max(0,episode-100):(episode+1)])
        
    plt.plot(rewards_for_window)
    plt.savefig('Rewardplot.png')
    
    if training:
        
        f=open('traineddata.pkl','wb')
        pickle.dump(q_table,f)
        f.close()
        

if __name__ == "__main__":
    run(15000,training=True)
    run(10,training=False,viewflag=True)



