#! /usr/local/bin/python3
# -*- encoding: utf-8 -*-
#
# Authors: Asger Anders Lund Hansen, Mads Ynddal and Troels Ynddal
# License: See LICENSE file
# GitHub: https://github.com/Baekalfen/PyBoy
#

import traceback
import time
import os.path
import os
import sys
import numpy as np
import platform
from PyBoy.Logger import logger
from PyBoy.WindowEvent import WindowEvent

# from Debug import Debug
from PyBoy import PyBoy
from PyBoy.GameWindow import SdlGameWindow as Window


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
        frame = 0
        view = pyboy.getTileView(False)
        while not pyboy.tick():

            if (frame % 10 == 0):
                # print("frame: {}".format(frame))
                print("X_{}: {} {}".format(frame, pyboy.mb[0xc20b], pyboy.mb[0xe20b]))

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

                # print('Screen pos: {}'.format(pyboy.getScreenPosition()))

                # As an example, it could be useful to know the coordinates
                # of the sprites on the screen and which they look like.
                # for n in range(40):
                #     sprite = pyboy.getSprite(n)
                #     if sprite.is_on_screen():
                #         print('Sprite: {} {} {} {}'.format(n, sprite.get_x(), sprite.get_y(), sprite.get_tile()))

                # Check for death
                # Run for 10 more frames
                if not pyboy.getSprite(3).is_on_screen():
                    if stop_at_frame == -1:
                        stop_at_frame = frame + 50
                        print('Stopping at frame {}'.format(stop_at_frame))

                # time.sleep(0.1)


            # if frame == stop_at_frame:
            #     # pyboy.stop()
            #     print('Resetting here *****')
            #     time.sleep(0.5)
            #     stop_at_frame = -1
            #     pyboy.mb.loadState(save_file)

            frame += 1
        pyboy.stop()

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as ex:
        traceback.print_exc()
