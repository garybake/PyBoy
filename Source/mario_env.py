"""
Basic environment
Mimic OpenAI gym (minimally)
"""
import random

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

    def _get_action_outcome(self):
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
            game_over,  # Game over
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

        if stop or outcome[2]:
            raise StopIteration

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
