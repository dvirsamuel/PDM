# Code adapted from https://prompt-to-prompt.github.io/
import abc


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def between_steps_inject(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0 #self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            attn = self.forward(attn, is_cross, place_in_unet)
        return attn

    def check_next_step(self):
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            if self.is_inject:
                self.between_steps_inject()
            else:
                self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.is_inject = False


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        #return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #        "down_self": [], "mid_self": [], "up_self": []}
        return {}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn is None:
            attn = self.attention_store[self.cur_step][key]
        else:
            self.step_store[key] = attn.cpu()
        #if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        #    self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store[self.cur_step - 1] = self.step_store
        # if len(self.attention_store) == 0:
        #     self.attention_store[self.cur_step-1] = self.step_store
        # else:
        #     for key in self.attention_store:
        #         for i in range(len(self.attention_store[key])):
        #             self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def between_steps_inject(self):
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}





