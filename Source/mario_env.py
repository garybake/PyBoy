"""
Basic environment
Mimic OpenAI gym (minimally)
"""
import random

# from Debug import Debug
from PyBoy import PyBoy
from PyBoy.GameWindow import SdlGameWindow as Window

from PyBoy.WindowEvent import WindowEvent
from PyBoy.Logger import logger

MEM_MARIO_X = 0xc0aa

ACTION_NONE = 0
ACTION_RIGHT = 3
ACTION_LEFT = 7


def sample(moves):
    return random.choice(moves)


class MarioEnv:
    """
    Environment to play super mario land on the gb
    """

    boot_rom = None

    def __init__(self, rom_file, state_file, scale=2):
        """
        Initialise the environment
        TODO: This should create the pyboy instance

        :param string rom_file: Cartridge rom file
        :param string state_file: State file to load
        :param int scale: Scale size of window
        """

        self.pyboy = PyBoy(Window(scale=scale), rom_file, self.boot_rom)
        self.state_file = state_file
        self.frame = 0
        self.ctrl_left = False
        self.ctrl_right = False
        self.prev_x = 0  # x of previous step
        self.action_space = self._get_action_space()

    def action_space_sample(self):
        """
        TODO how to add this func to action space?
        Should be actionspace.sample()?
        """
        return random.choice(self.action_space)

    def _get_action_space(self):
        """
        Returns a tuple of list of actions
        TODO: add more actions
        """
        return (0, 3, 7)

    def _clear_inputs(self):
        """
        Clears all key presses
        TODO: Start/select?
        """

        pyboy = self.pyboy
        pyboy.sendInput([WindowEvent.ReleaseButtonA])
        pyboy.sendInput([WindowEvent.ReleaseButtonB])
        pyboy.sendInput([WindowEvent.ReleaseArrowUp])
        pyboy.sendInput([WindowEvent.ReleaseArrowRight])
        pyboy.sendInput([WindowEvent.ReleaseArrowDown])
        pyboy.sendInput([WindowEvent.ReleaseArrowLeft])

    def reset(self):
        """
        Reset the environment
        Clears all key presses and loads a save state

        :return: Frame number of the save state
        :rtype: int
        """
        self._clear_inputs()

        self.pyboy.mb.loadState(self.state_file)

        self.frame = 0
        return [0]

    def obs(self):
        """
        Get current state of the environment
        Currently just marios x val

        :return: Array of observation values
        :rtype: array
        """
        mario_x = self.pyboy.mb.read_word(MEM_MARIO_X)
        return [mario_x]

    def _get_action_outcome(self):
        """
        Get the outcome of the last action

        :return: Outcome - state, reward, done, info
        :rtype: tuple
        """
        game_over = False
        if (self.frame > 10) and not self.pyboy.getSprite(3).is_on_screen():
            game_over = True
            logger.debug('Mario death')

        outcome = [
            self.obs(),  # state
            1.0,  # reward
            game_over,  # Game over
            None  # Debug info
        ]
        return outcome

    def step(self, action=None):
        """
        Step the environment forward 1 cycle
        Handles input

        None = clear all input
        Need to reward pressing right?

        :return: Outcome - state, reward, done, info
        :rtype: tuple
        """

        if action:
            self._clear_inputs()
            if action == ACTION_NONE:
                pass
            elif action == ACTION_RIGHT:
                self.pyboy.sendInput([WindowEvent.PressArrowRight])
                self.ctrl_right = True
            elif action == ACTION_LEFT:
                self.pyboy.sendInput([WindowEvent.PressArrowLeft])
                self.ctrl_left = True

        stop = self.pyboy.tick()
        outcome = self._get_action_outcome()

        if stop:
            outcome[2] = True

        self.frame += 1
        self.prev_x = outcome[0][0]
        return outcome

    def render(self, mode=None):
        """
        Returns and image of the env/screen
        TODO
        """
        return None

    def shape(self):
        """
        Returns the shape of the observation
        TODO
        """
        return (1)
