import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

# Define neural network model
def create_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def run(episodes, slipperyflag=True, viewflag=False, training=True):
    environment = gym.make('Taxi-v3', render_mode='human' if viewflag else None)
    
    state_size = environment.observation_space.n
    action_size = environment.action_space.n
    
    if training:
        model = create_model(1, action_size)
        target_model = create_model(1, action_size)
        target_model.set_weights(model.get_weights())
        memory = deque(maxlen=2000)
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.01
        gamma = 0.95
        batch_size = 64
        update_target_freq = 10
    else:
        model = create_model(1, action_size)
        model.load_weights('trained_model.h5')
    
    reward_of_every_episode = np.zeros(episodes)
    
    for i in range(episodes):
        print(f"Episode Number: {i}")
        
        state = environment.reset()[0]
        state = np.reshape([state], [1, 1])
        terminated = False
        maxmovesreached = False
        
        while not terminated and not maxmovesreached:
            if training and np.random.rand() <= epsilon:
                action = environment.action_space.sample()
            else:
                q_values = model.predict(state, verbose=0)
                action = np.argmax(q_values[0])
            
            newstate, reward, terminated, maxmovesreached, _ = environment.step(action)
            newstate = np.reshape([newstate], [1, 1])
            
            if training:
                memory.append((state, action, reward, newstate, terminated))
                
                if len(memory) > batch_size:
                    minibatch = random.sample(memory, batch_size)
                    for state_b, action_b, reward_b, newstate_b, done_b in minibatch:
                        target = reward_b
                        if not done_b:
                            target += gamma * np.amax(target_model.predict(newstate_b, verbose=0)[0])
                        target_f = model.predict(state_b, verbose=0)
                        target_f[0][action_b] = target
                        model.fit(state_b, target_f, epochs=1, verbose=0)
                    
                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay
                
                if i % update_target_freq == 0:
                    target_model.set_weights(model.get_weights())
            
            state = newstate
            
            # Debug information
            print(f"State: {state}, Action: {action}, Reward: {reward}, Terminated: {terminated}")
        
        reward_of_every_episode[i] = reward
        print(f"Episode {i} finished with reward: {reward}")
    
    environment.close()
    
    # Save the trained model
    if training:
        model.save('trained_model.h5')
    
    # Plot rewards
    rewards_for_window = np.zeros(episodes)
    for episode in range(episodes):
        rewards_for_window[episode] = np.sum(reward_of_every_episode[max(0, episode - 100):(episode + 1)])
    
    plt.plot(rewards_for_window)
    plt.savefig('Rewardplot.png')

if __name__ == "__main__":
    run(15000, training=True)
    run(10, training=False, viewflag=True)
