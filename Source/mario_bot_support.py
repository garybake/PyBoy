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

if platform.system() != "Windows":
    from Debug import Debug
from PyBoy import PyBoy
from PyBoy.GameWindow import SdlGameWindow as Window


boot_rom = None
rom_dir = 'ROMs'
save_file = os.path.join('saveStates', 'mario_save')
scale = 2


def getROM(rom_dir):
    filename = os.path.join(rom_dir, 'mario.gb')
    return filename


def create_save_state():
    try:
        # Check if the ROM is given through argv
        if len(sys.argv) > 1:  # First arg is SDL2/PyGame
            filename = sys.argv[1]
        else:
            filename = getROM(rom_dir)

        # Start PyBoy and run loop
        pyboy = PyBoy(Window(scale=scale), filename, boot_rom)
        frame = 0
        while not pyboy.tick():

            if (frame % 10 == 0):
                print("frame: {}".format(frame))

            if frame == 100:
                pyboy.sendInput([WindowEvent.PressButtonStart])
            elif frame == 102:
                pyboy.sendInput([WindowEvent.ReleaseButtonStart])

            if frame == 110:
                pyboy.mb.saveState(save_file)

            if frame == 120:
                pyboy.stop()

            frame += 1
        pyboy.stop()

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as ex:
        traceback.print_exc()


def filter_mem_locs(prev_vals, curr_vals):
    # prev_vals = dict
    # curr_cals = array
    for i in prev_vals:
        if curr_vals[i] > prev_vals[i]:
            prev_vals[i] = curr_vals[i]
        else:
            prev_vals[i] = None

    new_vals = {k: v for k, v in prev_vals.iteritems() if v is not None}
    return new_vals


def find_increasing_mem_locations():
    """
    Find memory locations that have values that increase over time
    """
    try:
        filename = getROM(rom_dir)

        # Start PyBoy and run loop
        pyboy = PyBoy(Window(scale=scale), filename, boot_rom)
        frame = 0
        candidate_mem_locs = {}
        while not pyboy.tick():

            if (frame % 10 == 0):
                print("frame: {}".format(frame))

            if frame == 10:
                pyboy.mb.loadState(save_file)
                pyboy.sendInput([WindowEvent.PressArrowRight])

            elif frame == 50:
                pyboy.sendInput([WindowEvent.PressButtonA])
            elif frame == 52:
                pyboy.sendInput([WindowEvent.ReleaseButtonA])

            if frame == 55:
                candidate_mem_locs = pyboy.mb.get_mem_array()

            if frame == 100:
                pyboy.sendInput([WindowEvent.PressButtonB])

            if frame > 20:
                if (frame % 10 == 0):
                        pyboy.sendInput([WindowEvent.PressButtonA])
                elif (frame % 10 == 5):
                        pyboy.sendInput([WindowEvent.ReleaseButtonA])

            # if (frame % 20 == 0):
            #     pyboy.window.dump(filename='/dev/shm/mario_{}.bmp'.format(frame), dump_all=False)
            #     # for n in range(40):
            #     #     sprite = pyboy.getSprite(n)
            #     #     if sprite.is_on_screen():
            #     #         print('Sprite: {} {}'.format(n, sprite.get_attributes()))

            if (frame > 79) and (frame % 10 == 0):
                mem_list = pyboy.mb.get_mem_array(bits=16)
                candidate_mem_locs = filter_mem_locs(candidate_mem_locs, mem_list)
                print('-- Mem Pass {}: {}'.format(frame, len(candidate_mem_locs)))
                for k in candidate_mem_locs:
                    print('{}: {}'.format(hex(k), hex(candidate_mem_locs[k])))

            if frame == 500:
                pyboy.stop()

            frame += 1
        pyboy.stop()

    except KeyboardInterrupt:
        print("Interrupted by keyboard")
    except Exception as ex:
        traceback.print_exc()

if __name__ == "__main__":
    # create_save_state()
    find_increasing_mem_locations()


# Increasing:
# 0xc20b: 0x77 r+jump
# 0xcff1: 0xf4
# 0xe20b: 0x77 r+jump
# 0xeff1: 0xf4
# 0xffac: 0xfc -- stay still
# 0xffae: 0xa3

# 16 bit inc =)
# 0xc0aa: 0x2698
# 0xe0aa: 0x2698