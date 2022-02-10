
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, time_feat, seq_len, start_token_len, pred_len):
        self.data = data
        self.time_feat = time_feat

        self.seq_len = seq_len
        self.start_token_len = start_token_len
        self.pred_len = pred_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # Find indices of input x
        x_start = index
        x_end = x_start + self.seq_len

        # Find indices of output y (includes buffer of length start token)
        y_start = x_end - self.start_token_len
        y_end = y_start + self.start_token_len + self.pred_len

        # Extract x and y sequences
        x = self.data[x_start:x_end]
        y = self.data[y_start:y_end]

        # Extract Time Features for x and y
        x_time_feat = self.time_feat[x_start:x_end]
        y_time_feat = self.time_feat[y_start:y_end]

        return x, y, x_time_feat, y_time_feat