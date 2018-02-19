"""
Basic environment
Mimic OpenAI gym (minimally)
"""

from PyBoy.WindowEvent import WindowEvent
from PyBoy.Logger import logger

MEM_MARIO_X = 0xc0aa


class MarioEnv:
    """
    Environment to play super mario land on the gb
    """

    def __init__(self, pyboy, state_file):
        """
        Initialise the environment
        TODO: This should create the pyboy instance

        :param PyBoy pyboy: PyBoy instance to run
        :param string state_file: State file to load
        """
        self.pyboy = pyboy
        self.state_file = state_file
        self.frame = 0
        self.ctrl_left = False
        self.ctrl_right = False

    def reset(self):
        """
        Reset the environment
        Clears all key presses and loads a save state

        :return: Frame number of the save state
        :rtype: int
        """
        pyboy = self.pyboy

        pyboy.sendInput([WindowEvent.ReleaseButtonA])
        pyboy.sendInput([WindowEvent.ReleaseButtonB])
        pyboy.sendInput([WindowEvent.ReleaseArrowUp])
        pyboy.sendInput([WindowEvent.ReleaseArrowRight])
        pyboy.sendInput([WindowEvent.ReleaseArrowDown])
        pyboy.sendInput([WindowEvent.ReleaseArrowLeft])

        pyboy.mb.loadState(self.state_file)

        self.frame = 0
        return self.frame

    def obs(self):
        """
        Get current state of the environment
        Currently just marios x val

        :return: Array of observation values
        :rtype: array
        """
        mario_x = self.pyboy.mb.read_word(MEM_MARIO_X)
        return [mario_x]

    def get_action_outcome(self):
        """
        Get the outcome of the last action

        :return: Outcome - state, reward, done, info
        :rtype: tuple
        """
        game_over = False
        # if not self.pyboy.getSprite(3).is_on_screen():
        #     game_over = True
        #     logger.debug('Mario death')

        outcome = [
            self.obs(),  # state
            1.0,  # reward
            game_over,  # Game over?
            None  # Debug info
        ]
        return outcome

    def step(self, action=None):
        """
        Step the environment forward 1 cycle
        Handles input

        TODO: handling no input?
        None = clear all input
        Diff = clear inputs and do new
        Same as last = continue
        Need to reward pressing right?

        :return: Outcome - state, reward, done, info
        :rtype: tuple
        """

        if action == 'right':
            if self.ctrl_left:
                self.pyboy.sendInput([WindowEvent.ReleaseArrowLeft])
            self.pyboy.sendInput([WindowEvent.PressArrowRight])
            self.ctrl_right = True
        elif action == 'left':
            if self.ctrl_right:
                self.pyboy.sendInput([WindowEvent.ReleaseArrowRight])
            self.pyboy.sendInput([WindowEvent.PressArrowLeft])
            self.ctrl_left = True

        stop = self.pyboy.tick()
        outcome = self.get_action_outcome()

        if stop or outcome[2]:
            raise StopIteration

        self.frame += 1
        return outcome
