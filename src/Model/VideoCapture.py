import numpy as np
import torch

from DataCapture.capture_data import DataCapture


class Video_Capture(DataCapture):
    '''
    The main change in this inherited class is it also returns the frame that is processed when the screenshot is taken. This is primarily useful for rendering the game when we are testing the environment with our agent.
    '''

    def __init__(self):
        super().__init__()

    # Capture one frame, process it and return the processed tensor array and the done status
    def capture_frame(self):
        frame = self.sc.grab(self.bbox)
        raw_frame = frame.raw
        if self.prev_frame == raw_frame:
            # if self.prev_frame==self.prev2_frame:
            self.done = True
        # self.prev2_frame=self.prev_frame
        self.prev_frame = raw_frame
        frame_tensor = torch.from_numpy(np.where((255-np.array(frame)[:, :, 0] if np.sum(np.array(
            frame)[:, :, 0]) < self.check_night else np.array(frame)[:, :, 0]) < 85, 0, 1)).expand(1, -1, -1)
        return frame_tensor, frame
