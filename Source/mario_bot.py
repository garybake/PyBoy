#! /usr/local/bin/python3

# Mario Bot
# Reinforcement Learning with Mario

import traceback
# import time
import os.path
import os
import sys
# import numpy as np
# import platform
from PyBoy.Logger import logger
from PyBoy.WindowEvent import WindowEvent

# from Debug import Debug
from PyBoy import PyBoy
from PyBoy.GameWindow import SdlGameWindow as Window

from mario_env import MarioEnv

MAX_ITERATIONS = 3

ACTION_NONE = 0
ACTION_RIGHT = 3
ACTION_LEFT = 7

stop_at_frame = -1



def getROM(rom_dir):
    filename = os.path.join(rom_dir, 'mario.gb')
    return filename


if __name__ == "__main__":
    boot_rom = None
    rom_dir = "ROMs"
    save_file = os.path.join('saveStates', 'mario_save')
    scale = 2

    try:
        # Check if the ROM is given through argv
        if len(sys.argv) > 1:  # First arg is SDL2/PyGame
            filename = sys.argv[1]
        else:
            filename = getROM(rom_dir)

        # Start PyBoy and run loop
        pyboy = PyBoy(Window(scale=scale), filename, boot_rom)
        iteration = 1
        frame = 0
        view = pyboy.getTileView(False)

        logger.info('Starting iteration: {}'.format(iteration))
        env = MarioEnv(pyboy, save_file)

        env.reset()

        for x in range(500):

            # next_action = ACTION_RIGHT
            next_action = env.action_space_sample()

            obs, reward, done, _ = env.step(action=next_action)

            frame = env.frame
            if (frame % 10 == 0):
                logger.info("Frame: {} \t x: {}".format(frame, obs))

        #     if frame == 10:
        #         pyboy.mb.loadState(save_file)

        #     if frame > 30:

        #          pyboy.sendInput([WindowEvent.PressArrowRight])
        #         # elif frame % 2 == 1:
        #         #     pyboy.sendInput([WindowEvent.ReleaseArrowRight])

        #         if frame > 10:
        #             if (frame % 10 == 0):
        #                     pyboy.sendInput([WindowEvent.PressButtonA])
        #             elif (frame % 10 == 5):
        #                     pyboy.sendInput([WindowEvent.ReleaseButtonA])

        #         # Check for death
        #         if not pyboy.getSprite(3).is_on_screen():
        #             if iteration >= MAX_ITERATIONS:
        #                 pyboy.stop()

        #             frame = reset_env(pyboy, save_file)
        #             iteration += 1
        #             logger.info('Starting iteration: {}'.format(iteration))

        #             # time.sleep(0.1)

        #     if frame == 500:
        #         pyboy.stop()

            # env.frame
        # pyboy.stop()

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as ex:
        traceback.print_exc()
