#! /usr/local/bin/python3

# Mario Bot
# Reinforcement Learning with Mario

import traceback
# import time
import os.path
import os
import sys
from PyBoy.Logger import logger

from mario_env import MarioEnv

MAX_ITERATIONS = 3

ACTION_NONE = 0
ACTION_RIGHT = 3
ACTION_LEFT = 7

stop_at_frame = -1


def main():
    rom_file = os.path.join('ROMs', 'mario.gb')
    state_file = os.path.join('saveStates', 'mario_save')

    try:

        iteration = 1

        logger.info('Starting iteration: {}'.format(iteration))
        env = MarioEnv(rom_file, state_file)

        env.reset()

        done = False
        while not done:

            next_action = ACTION_RIGHT
            # next_action = env.action_space_sample()

            obs, reward, done, _ = env.step(action=next_action)

            frame = env.frame
            if (frame % 10 == 0):
                logger.info("Frame: {} \t x: {}".format(frame, (obs, reward, done)))

            if frame == 500:
                done = True

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as ex:
        traceback.print_exc()


if __name__ == "__main__":
    main()
