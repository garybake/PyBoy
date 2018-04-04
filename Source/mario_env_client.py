# # This is the code that visits the warehouse.
# import Pyro4
# # from person import Person

# uri = input("Enter the uri of the instance: ").strip()
# mario_env = Pyro4.Proxy(uri)

# # psf = Pyro4.Proxy("PYRONAME:MyApp.Factories.ProductFactory")
# print(mario_env.action_space_sample())
# # janet = Person("Janet")
# # henry = Person("Henry")
# # janet.visit(warehouse)
# # henry.visit(warehouse)


#! /usr/local/bin/python3

# Mario Bot
# Reinforcement Learning with Mario

import traceback
import os
import sys

import Pyro4

from PyBoy.Logger import logger

from mario_env import MarioEnv

MAX_ITERATIONS = 3

ACTION_NONE = 0
ACTION_RIGHT = 3
ACTION_LEFT = 7

stop_at_frame = -1


def basic_policy(obs):
    mario_x = obs[0]
    if mario_x < 4000:
        return ACTION_RIGHT
    return ACTION_LEFT


def main():
    # rom_file = os.path.join('ROMs', 'mario.gb')
    # state_file = os.path.join('saveStates', 'mario_save')
    env = None

    try:

        logger.info('Starting environment')
        # env = MarioEnv(rom_file, state_file)
        # uri = 'PYRO:example.marioenv@localhost:43047'
        uri = "PYRO:marioenv@localhost:9999"
        env = Pyro4.Proxy(uri)
        # env.reset()
    except Exception as e:
        logger.error('Failed to start environment')
        logger.error(e)
        traceback.print_exc()
        sys.exit()

    totals = []
    try:
        for episode in range(4):
            episode_rewards = 0
            obs = env.reset()

            for step in range(100):  # 500 steps max
                action = basic_policy(obs)
                obs, reward, done, _ = env.step(action=action)

                if step % 10 == 0:
                    logger.debug('reward: {}, obs: {}'.format(reward, obs))

                episode_rewards += reward
                if done:
                    break
            totals.append(episode_rewards)

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as e:
        logger.error('Failed during run')
        logger.error(e)
        traceback.print_exc()

    logger.info('Totals: {}'.format(totals))
    env.stop()

if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()')
    main()
