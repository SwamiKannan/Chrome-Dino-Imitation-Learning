import numpy as np
import torch

from DataCapture.capture_data import DataCapture


class Video_Capture(DataCapture):
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
        '''The command above does three things:
        1. Convert night to day (i.e. inverts the image if the number of white pixels of the crop is less than 50% of total pixels)
        2. Make a binary numpy with values less than 85 in the RGB values as 0 and the rest as 1. This removes clouds, moons and stars which are irrelevant
        3. Convert the numpy array to torch.tensor. 
            numpy is faster than torch on where() and sum() commands. 
            Further, we can also convert an mss screenshot to numpy but not pytorch tensor.
            Hence, while we need an output of a pytorch tensor, we use numpy as an intermediary
        4. Expand the dims to allow stacking of this observation on the observation stack for the game

        '''
        return frame_tensor, frame
