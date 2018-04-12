#! /usr/local/bin/python3

# Mario Bot
# Reinforcement Learning with Mario

import traceback
import logging
import sys
import random
from collections import deque

import Pyro4
from PIL import Image
import numpy as np

from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam

MAX_ITERATIONS = 3

ACTION_NONE = 0
ACTION_RIGHT = 3
ACTION_LEFT = 7
NUM_ACTIONS = 3

logger = logging.getLogger('CLIENT')

IMG_ROWS, IMG_COLS = 144, 160
# Convert image into Black and white
IMG_CHANNELS = 4  # We stack 4 frames

INITIAL_EPSILON = 0.0001
FRAME_PER_ACTION = 1
OBSERVATIONS_BEFORE_TRAINING = 100
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001  # final value of epsilon
EXPLORE = 3000000.  # frames over which to anneal epsilon

def basic_policy(obs):
    return random_policy(obs)
    # mario_x = obs[0]
    # if mario_x < 4000:
    #     return ACTION_RIGHT
    # return ACTION_LEFT


def random_policy(obs):
    if bool(random.getrandbits(1)):
        return ACTION_RIGHT
    return ACTION_LEFT

policy_func = basic_policy


def blank_screen():
    return np.array([[0] * 160] * 144)


def array_to_png(screen, filename):
    """
    Convert the raw screen array to a png
    """
    # im = Image.fromarray(np.array(screen, dtype=np.int32))
    screen[screen == 1] = 16777215
    im = Image.fromarray(screen)
    im.save('/tmp/marioimage.png')
    flat_list = [item for sublist in screen for item in sublist]
    print(set(flat_list))


def preprocess_observation(obs):
    if len(obs) == 1:
        return blank_screen()

    screen = np.array(obs, dtype=np.int32)

    screen[screen == 10066329] = 0
    screen[screen == 5592405] = 0
    screen[screen == 0] = 0
    screen[screen == 16777215] = 1

    # screen = screen[::2, ::2]
    screen = np.transpose(screen)
    return screen


def buildmodel():
    # Adapted from https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html
    print("Building the model ")

    model = Sequential()
    model.add(layers.Convolution2D(
        32, 8, 8, subsample=(4, 4), border_mode='same',
        input_shape=(IMG_CHANNELS, IMG_ROWS, IMG_COLS)))
    model.add(layers.Activation('relu'))
    model.add(layers.Convolution2D(
        64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Convolution2D(
        64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2))

    adam = Adam(lr=1e-6)
    model.compile(
        loss='mse',
        optimizer=adam)
    print("We finish building the model")
    return model


def main():
    env = None

    try:

        logger.info('Starting environment')
        uri = "PYRO:marioenv@localhost:9999"
        env = Pyro4.Proxy(uri)
        env.start_pyboy()

        model = buildmodel()
        print(model.summary())
        train_network(model, env, args=None)

    except Exception as e:
        logger.error('Failed to start environment')
        logger.error(e)
        traceback.print_exc()
        sys.exit()

    # totals = []
    # try:
    #     for episode in range(1):
    #         episode_rewards = 0
    #         obs = env.reset()

    #         for step in range(52):  # 500 steps max
    #             action = policy_func(obs)
    #             obs, reward, done, _ = env.step(action=action)

    #             if step % 10 == 0:
    #                 logger.debug('reward: {}, obs: {}'.format(reward, obs))
    #             if step == 50:
    #                 # screen = env.get_screen()
    #                 screen = preprocess_observation(obs)
    #                 array_to_png(screen, '/tmp/mario_screen.png')

    #             episode_rewards += reward
    #             if done:
    #                 break
    #         totals.append(episode_rewards)
    #         logger.error('Episode total reward: {}'.format(episode_rewards))

    # except KeyboardInterrupt:
    #     print("Interrupted by keyboard")
    # except Exception as e:
    #     logger.error('Failed during run')
    #     logger.error(e)
    #     traceback.print_exc()

    # logger.debug('Totals: {}'.format(totals))
    env.shutdown()
    env._pyroRelease()


def train_network(model, env, args):
    # open up a game state to communicate with emulator
    # game_state = game.GameState()
    game_state = preprocess_observation(env.reset())

    # store the previous observations in replay memory
    replay_mem = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(NUM_ACTIONS)
    do_nothing[0] = 1
    # x_t, r_0, terminal = game_state.frame_step(do_nothing)

    # x_t = skimage.color.rgb2gray(x_t)
    # x_t = skimage.transform.resize(x_t,(80,80))
    # x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    # s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

    #In Keras, need to reshape
    # s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    state_stack = np.stack((game_state, game_state, game_state, game_state), axis=0)

    state_stack = state_stack.reshape(1, state_stack.shape[0], state_stack.shape[1], state_stack.shape[2])
    print(state_stack.shape)
    # # if args['mode'] == 'Run':
    # #     OBSERVE = 999999999    #We keep observe, never train
    # #     epsilon = FINAL_EPSILON
    # #     print ("Now we load weight")
    # #     model.load_weights("model.h5")
    # #     adam = Adam(lr=1e-6)
    # #     model.compile(loss='mse',optimizer=adam)
    # #     print ("Weight load successfully")
    # # else:                       #We go to training mode
    OBSERVE = OBSERVATIONS_BEFORE_TRAINING
    epsilon = INITIAL_EPSILON

    tick = 0
    while tick < 10:
    # while (True):
        loss = 0
        q_max = 0
        action_index = 0
        reward = 0
        action = np.zeros([NUM_ACTIONS])

        # choose an action epsilon greedy
        if tick % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(NUM_ACTIONS)
                action[action_index] = 1
            else:
                q = model.predict(state_stack)  # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                action[max_Q] = 1

            print(action)

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and tick > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    #     #run the selected action and observed next state and reward
    #     x_t1_colored, reward, terminal = game_state.frame_step(action)

    #     x_t1 = skimage.color.rgb2gray(x_t1_colored)
    #     x_t1 = skimage.transform.resize(x_t1,(80,80))
    #     x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

    #     x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
    #     s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1)

    #     # store the transition in D
    #     D.append((s_t, action_index, reward, s_t1, terminal))
    #     if len(D) > REPLAY_MEMORY:
    #         D.popleft()

    #     #only train if done observing
    #     if t > OBSERVE:
    #         #sample a minibatch to train on
    #         minibatch = random.sample(D, BATCH)

    #         inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
    #         targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

    #         #Now we do the experience replay
    #         for i in range(0, len(minibatch)):
    #             state_t = minibatch[i][0]
    #             action_t = minibatch[i][1]   #This is action index
    #             reward_t = minibatch[i][2]
    #             state_t1 = minibatch[i][3]
    #             terminal = minibatch[i][4]
    #             # if terminated, only equals reward

    #             inputs[i:i + 1] = state_t    #I saved down s_t

    #             targets[i] = model.predict(state_t)  # Hitting each buttom probability
    #             q_max = model.predict(state_t1)

    #             if terminal:
    #                 targets[i, action_t] = reward_t
    #             else:
    #                 targets[i, action_t] = reward_t + GAMMA * np.max(q_max)

    #         # targets2 = normalize(targets)
    #         loss += model.train_on_batch(inputs, targets)

    #     s_t = s_t1
        tick += 1

    #     # save progress every 10000 iterations
    #     if t % 100 == 0:
    #         print("Now we save model")
    #         model.save_weights("model.h5", overwrite=True)
    #         with open("model.json", "w") as outfile:
    #             json.dump(model.to_json(), outfile)

    #     # print info
    #     state = ""
    #     if t <= OBSERVE:
    #         state = "observe"
    #     elif t > OBSERVE and t <= OBSERVE + EXPLORE:
    #         state = "explore"
    #     else:
    #         state = "train"

    #     print("TIMESTEP", t, "/ STATE", state, \
    #         "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
    #         "/ Q_MAX " , np.max(q_max), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()


# https://github.com/rafalrusin/Keras-FlappyBird/blob/45d010f2adeddb50b659692c247153fca084af36/qlearn.py