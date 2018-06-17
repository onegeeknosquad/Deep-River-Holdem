#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 10:54:33 2018

@author: mrpotatohead
"""
#First we import some libraries
#Json for loading and saving the model (optional)
import json
#matplotlib for rendering
import matplotlib.pyplot as plt
#numpy for handeling matrix operations
import numpy as np
#time, to, well... keep track of time
import time
#Python image libarary for rendering
from PIL import Image
#iPython display for making sure we can render the frames
from IPython import display
#seaborn for rendering
import seaborn
#Keras is a deep learning libarary
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from experience_replay import *
import gym
import holdem
from player import Player
from Deck import *


def round_setup(env, board, round, totalpot):
	env.self.community = board
	env.self.round = 3
	env.self.totalpot = 55

def baseline_model(grid_size,num_actions,hidden_size):
    #seting up the model with keras
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.1), "mse")
    return model


# parameters
epsilon = .1  # exploration
num_actions = 3  # [move_left, stay, move_right]
max_memory = 500 # Maximum number of experiences we are storing
hidden_size = 100 # Size of the hidden layers
batch_size = 1 # Number of experiences we use for training per batch
grid_size = 10 # Size of the playing field


#Define model
model = baseline_model(grid_size,num_actions,hidden_size)
model.summary()

# Define environment/game
env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)


def train(model,epochs, verbose = 1):
    # Train
    #Reseting the win counter
    win_cnt = 0
    # We want to keep track of the progress of the AI over time, so we save its win count history
    win_hist = []
    #Epochs is the number of games we play
    for e in range(epochs):
        loss = 0.
        #Resetting the game
        env.reset()
        
        #set the board
        ndeck = list(deck)
        np.random.shuffle(ndeck)
        board = ndeck[:5]
        
        for i in range(2):
            p1 = board[5:].pop()
            p2 = board[5:].pop()
        print(p1,p2)
        # start with 2 players
        env.add_player(0, stack=73) # add a player to seat 0 with 2000 "chips"
        env.add_player(1, stack=73) # add another player to seat 1 with 2000 "chips"
        
        game_over = False

        # get initial input
        input_t = env._get_current_state()
        
        while not game_over:
            #The learner is acting on the last observed game screen
            #input_t is a vector containing representing the game screen
            input_tm1 = input_t
            
            """
            We want to avoid that the learner settles on a local minimum.
            Imagine you are eating eating in an exotic restaurant. After some experimentation you find 
            that Penang Curry with fried Tempeh tastes well. From this day on, you are settled, and the only Asian 
            food you are eating is Penang Curry. How can your friends convince you that there is better Asian food?
            It's simple: Sometimes, they just don't let you choose but order something random from the menu.
            Maybe you'll like it.
            The chance that your friends order for you is epsilon
            """
            if np.random.rand() <= epsilon:
                #Eat something random from the menu
                action = np.random.randint(0, num_actions, size=1)
            else:
                #Choose yourself
                #q contains the expected rewards for the actions
                q = model.predict(input_tm1)
                #We pick the action with the highest expected reward
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            #If we managed to catch the fruit we add 1 to our win counter
            if reward == 1:
                win_cnt += 1        
            
            #Uncomment this to render the game here
            #display_screen(action,3000,inputs[0])
            
            """
            The experiences < s, a, r, s’ > we make during gameplay are our training data.
            Here we first save the last experience, and then load a batch of experiences to train our model
            """
            
            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)    
            
            # Load batch of experiences
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
  
            # train model on experiences
            batch_loss = model.train_on_batch(inputs, targets)
            
            #print(loss)
            loss += batch_loss
        if verbose > 0:
            print("Epoch {:03d}/{:03d} | Loss {:.4f} | Win count {}".format(e,epochs, loss, win_cnt))
        win_hist.append(win_cnt)
    return win_hist



epoch = 5000 # Number of games played in training, I found the model needs about 4,000 games till it plays well
# Train the model
# For simplicity of the noteb
hist = train(model,epoch,verbose=0)
print("Training done")

