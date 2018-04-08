#! /usr/local/bin/python3

# Mario Bot
# Reinforcement Learning with Mario

import traceback
import logging
import sys
import random

import Pyro4
from PIL import Image
import numpy as np


MAX_ITERATIONS = 3

ACTION_NONE = 0
ACTION_RIGHT = 3
ACTION_LEFT = 7

stop_at_frame = -1
logger = logging.getLogger('CLIENT')


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
    screen = np.array(obs, dtype=np.int32)
    print(screen[0][-10:])
    screen[screen == 10066329] = 0
    screen[screen == 5592405] = 0
    screen[screen == 0] = 0
    screen[screen == 16777215] = 1

    print(screen[0][-10:])
    screen = np.transpose(screen)
    return screen


def main():
    env = None

    try:

        logger.info('Starting environment')
        uri = "PYRO:marioenv@localhost:9999"
        env = Pyro4.Proxy(uri)
        env.start_pyboy()

    except Exception as e:
        logger.error('Failed to start environment')
        logger.error(e)
        traceback.print_exc()
        sys.exit()

    totals = []
    try:
        for episode in range(1):
            episode_rewards = 0
            obs = env.reset()

            for step in range(52):  # 500 steps max
                action = policy_func(obs)
                obs, reward, done, _ = env.step(action=action)

                if step % 10 == 0:
                    logger.debug('reward: {}, obs: {}'.format(reward, obs))
                if step == 50:
                    # screen = env.get_screen()
                    screen = preprocess_observation(obs)
                    array_to_png(screen, '/tmp/mario_screen.png')

                episode_rewards += reward
                if done:
                    break
            totals.append(episode_rewards)
            logger.error('Episode total reward: {}'.format(episode_rewards))

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as e:
        logger.error('Failed during run')
        logger.error(e)
        traceback.print_exc()

    logger.debug('Totals: {}'.format(totals))
    env.shutdown()
    env._pyroRelease()

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()
