import torch
import torch.nn as nn
import torch.nn.functional as F

class NoteEventPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_size=256, num_layers=2, dropout=0.1):
        super(NoteEventPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc_1 = nn.Linear(hidden_size, 2)  # for mu_t, sigma_t
        self.fc_2 = nn.Linear(hidden_size, 2)  # for mu_d, sigma_d
        self.fc_3 = nn.Linear(hidden_size, 2)  # for mu_v, sigma_v
        self.fc_4 = nn.Linear(hidden_size, 128)  # logits for pi0, ..., pi127

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  

        t_param = self.fc_1(last_hidden)
        d_param = self.fc_2(last_hidden)
        v_param = self.fc_3(last_hidden)
        pi_logits = self.fc_4(last_hidden)

        # For positive standard deviations
        t_param[:, 1] = torch.exp(t_param[:, 1])
        d_param[:, 1] = torch.exp(d_param[:, 1])
        v_param[:, 1] = torch.exp(v_param[:, 1])

        # Taking softmax, then log of softmax and adding small epsilon to avoid log(0) and overflow.
        pi_prob = F.softmax(pi_logits, dim=1)
        log_pi_prob = torch.log(pi_prob + 1e-10)  

        pred = torch.cat([t_param, d_param, log_pi_prob, v_param], dim=1)
        return pred