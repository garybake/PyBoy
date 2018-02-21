#! /usr/local/bin/python3

# Mario Bot
# Reinforcement Learning with Mario

import traceback
import os

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
    rom_file = os.path.join('ROMs', 'mario.gb')
    state_file = os.path.join('saveStates', 'mario_save')

    try:

        logger.info('Starting environment')
        env = MarioEnv(rom_file, state_file)
        # env.reset()
    except Exception as e:
        logger.error('Failed to start environment')
        logger.error(e)
        traceback.print_exc()

    totals = []
    try:
        for episode in range(3):
            episode_rewards = 0
            obs = env.reset()

            for step in range(500):  # 500 steps max
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

if __name__ == "__main__":
    main()
