#! /usr/local/bin/python3

# Mario Bot
# Reinforcement Learning with Mario

import traceback
import logging
import sys
import random
import json
import argparse
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
OBSERVATIONS_BEFORE_TRAINING = 50
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001  # final value of epsilon
GAMMA = 0.99  # decay rate of past observations
EXPLORE = 3000000.  # frames over which to anneal epsilon
REPLAY_MEMORY_SIZE = 50000
BATCH_SIZE = 32
EXPLORE = 3000000  # frames over which to anneal epsilon TODO


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
    model.add(layers.Dense(3))

    adam = Adam(lr=1e-6)
    model.compile(
        loss='mse',
        optimizer=adam)
    print("We finish building the model")
    return model


def translate_action(action_arr):
    """
    Translates from an action array to a single action
    [L, None, R]
    """
    # TODO I don't think we need this

    if action_arr[0]:
        return ACTION_LEFT
    if action_arr[2]:
        return ACTION_RIGHT
    return ACTION_NONE


def main(args):
    env = None

    try:

        logger.info('Starting environment')
        uri = "PYRO:marioenv@localhost:9999"
        env = Pyro4.Proxy(uri)
        env.start_pyboy()

        model = buildmodel()
        print(model.summary())
        train_network(model, env, args=args)

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
    game_state_t0 = preprocess_observation(env.reset())
    # TODO do we need to process this?

    # store the previous observations in replay memory
    replay_mem = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(NUM_ACTIONS)
    do_nothing[1] = 1.
    print(do_nothing)

    obs, _, _, _ = env.step(action=translate_action(do_nothing))
    game_state_t0 = preprocess_observation(obs)

    state_stack = np.stack((game_state_t0, game_state_t0, game_state_t0, game_state_t0), axis=0)

    # TODO why do we need to reshape in Kera?
    state_stack = state_stack.reshape(1, state_stack.shape[0], state_stack.shape[1], state_stack.shape[2])
    print(state_stack.shape)
    if args['mode'] == 'run':
        mode = 'run'
        # We keep observe, never train
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse', optimizer=adam)
        print ("Weight load successfully")
    else:
        # We go to training mode
        mode = 'train'
        OBSERVE = OBSERVATIONS_BEFORE_TRAINING
        epsilon = INITIAL_EPSILON

    tick = 0
    while tick < 500:
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
                # input a stack of 4 images, get the prediction
                q = model.predict(state_stack)
                max_Q = np.argmax(q)
                action_index = max_Q
                action[max_Q] = 1

        # We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and tick > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        obs, reward, done, terminal = env.step(action=translate_action(action))
        game_state_t1 = preprocess_observation(obs)

        game_state_t1 = game_state_t1.reshape(
            1, 1, game_state_t1.shape[0], game_state_t1.shape[1])
        state_stack_t1 = np.append(
            game_state_t1, state_stack[:, :3, :, :], axis=1)

        # store the transition in the replay memory
        replay_mem.append(
            (state_stack, action_index, reward, state_stack_t1, terminal))
        if len(replay_mem) > REPLAY_MEMORY_SIZE:
            replay_mem.popleft()

        # only train if done observing
        if tick > OBSERVATIONS_BEFORE_TRAINING and mode != 'run':
            # sample a minibatch to train on
            minibatch = random.sample(replay_mem, BATCH_SIZE)

            inputs = np.zeros((BATCH_SIZE, state_stack.shape[1], state_stack.shape[2], state_stack.shape[3]))   # 32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], NUM_ACTIONS))  # 32, 2

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]  # This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    # I saved down state_stack

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                q_max = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(q_max)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

        state_stack = state_stack_t1
        tick += 1

        # save progress every 10000 iterations
        # TODO no need to save in run mode
        if tick % 100 == 0:
            print("Saving model")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        state = ""
        if tick <= OBSERVATIONS_BEFORE_TRAINING:
            state = "observe"
        elif tick > OBSERVATIONS_BEFORE_TRAINING and tick <= OBSERVATIONS_BEFORE_TRAINING + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print('T: {} | State: {} | E: {:.8f} | Action: {} | Reward: {} | Q_Max: {:.8f} | Loss: {:.8f}'.format(
            tick, state, epsilon, action_index, reward, np.max(q_max), loss))

    print("Episode finished!")
    print("************************")

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    parser = argparse.ArgumentParser(description='Mario learnings')
    parser.add_argument('-m', '--mode', help='train / run', required=True)
    args = vars(parser.parse_args())

    main(args)


# https://github.com/rafalrusin/Keras-FlappyBird/blob/45d010f2adeddb50b659692c247153fca084af36/qlearn.py