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


MEM_MARIO_X = 0xc0aa
MAX_ITERATIONS = 3

stop_at_frame = -1


def reset_env(pyboy, save_file):
    pyboy.sendInput([WindowEvent.ReleaseButtonA])
    pyboy.sendInput([WindowEvent.ReleaseButtonB])
    pyboy.sendInput([WindowEvent.ReleaseArrowUp])
    pyboy.sendInput([WindowEvent.ReleaseArrowRight])
    pyboy.sendInput([WindowEvent.ReleaseArrowDown])
    pyboy.sendInput([WindowEvent.ReleaseArrowLeft])
    pyboy.mb.loadState(save_file)
    frame = 11
    return frame


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
        while not pyboy.tick():

            if (frame % 10 == 0):
                logger.info("Frame: {} \t x: {}".format(frame, pyboy.mb.read_word(MEM_MARIO_X)))

            if frame == 10:
                pyboy.mb.loadState(save_file)

            if frame > 30:

                pyboy.sendInput([WindowEvent.PressArrowRight])
                # elif frame % 2 == 1:
                #     pyboy.sendInput([WindowEvent.ReleaseArrowRight])

                if frame > 10:
                    if (frame % 10 == 0):
                            pyboy.sendInput([WindowEvent.PressButtonA])
                    elif (frame % 10 == 5):
                            pyboy.sendInput([WindowEvent.ReleaseButtonA])

                # Check for death
                if not pyboy.getSprite(3).is_on_screen():
                    if iteration >= MAX_ITERATIONS:
                        pyboy.stop()

                    frame = reset_env(pyboy, save_file)
                    iteration += 1
                    logger.info('Starting iteration: {}'.format(iteration))

                    # time.sleep(0.1)

            if frame == 500:
                pyboy.stop()

            frame += 1
        pyboy.stop()

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as ex:
        traceback.print_exc()
