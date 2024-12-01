import torch
import torch.nn as nn
from torch import fft
from torch.nn import init
from torch.nn import functional as F
from torch import optim
import os
import cv2
import glob
import torch
import yaml
import argparse
import random
import numpy as np
from torch.optim import Adam
import pickle
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
# from spikingjelly.clock_driven.neuron import MultiStepLIFNode
# from spikingjelly.clock_driven import functional
from scipy.signal import cont2discrete

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MnistLSTM(nn.Module):
    r"""
    Very Simple 2 layer LSTM with an fc layer on last steps hidden dimension
    """
    def __init__(self, input_size, hidden_size, codebook_size):
        super(MnistLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size // 4),
                                nn.ReLU(),
                                nn.Linear(hidden_size // 4, codebook_size))
        # Add pad and start token to embedding size
        self.word_embedding = nn.Embedding(codebook_size+2, input_size)
    
    def forward(self, x):
        x = self.word_embedding(x)
        output, _ = self.rnn(x)
        output = output[:, -1, :]
        return self.fc(output)

class LMUFFTCell(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, seq_len, theta):
        super(LMUFFTCell, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.seq_len = seq_len
        self.theta = theta

        self.W_u = nn.Linear(in_features = input_size, out_features = 1)
        self.f_u = nn.ReLU()
        self.W_h = nn.Linear(in_features = memory_size + input_size, out_features = hidden_size)
        self.f_h = nn.ReLU()

        A, B = self.stateSpaceMatrices()
        self.register_buffer("A", A) # [memory_size, memory_size]
        self.register_buffer("B", B) # [memory_size, 1]

        H, fft_H = self.impulse()
        self.register_buffer("H", H) # [memory_size, seq_len]
        self.register_buffer("fft_H", fft_H) # [memory_size, seq_len + 1]

    def stateSpaceMatrices(self):
        """ Returns the discretized state space matrices A and B """

        Q = np.arange(self.memory_size, dtype = np.float64).reshape(-1, 1)
        R = (2*Q + 1) / self.theta
        i, j = np.meshgrid(Q, Q, indexing = "ij")

        # Continuous
        A = R * np.where(i < j, -1, (-1.0)**(i - j + 1))
        B = R * ((-1.0)**Q)
        C = np.ones((1, self.memory_size))
        D = np.zeros((1,))

        # Convert to discrete
        A, B, C, D, dt = cont2discrete(
            system = (A, B, C, D), 
            dt = 1.0, 
            method = "zoh"
        )

        # To torch.tensor
        A = torch.from_numpy(A).float() # [memory_size, memory_size]
        B = torch.from_numpy(B).float() # [memory_size, 1]
        
        return A, B

    def impulse(self):
        """ Returns the matrices H and the 1D Fourier transform of H (Equations 23, 26 of the paper) """

        H = []
        A_i = torch.eye(self.memory_size).to(self.A.device) 
        for t in range(self.seq_len):
            H.append(A_i @ self.B)
            A_i = self.A @ A_i

        H = torch.cat(H, dim = -1) # [memory_size, seq_len]
        fft_H = fft.rfft(H, n = 2*self.seq_len, dim = -1) # [memory_size, seq_len + 1]

        return H, fft_H

    def forward(self, x):
        """
        Parameters:
            x (torch.tensor): 
                Input of size [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, input_size = x.shape
        # print("batch_size, seq_len, input_size", batch_size, seq_len, input_size)

        # Equation 18 of the paper
        u = self.f_u(self.W_u(x)) # [batch_size, seq_len, 1]

        # Equation 26 of the paper
        fft_input = u.permute(0, 2, 1) # [batch_size, 1, seq_len]
        fft_u = fft.rfft(fft_input, n = 2*seq_len, dim = -1) # [batch_size, seq_len, seq_len+1]

        # Element-wise multiplication (uses broadcasting)
        # [batch_size, 1, seq_len+1] * [1, memory_size, seq_len+1]
        temp = fft_u * self.fft_H.unsqueeze(0) # [batch_size, memory_size, seq_len+1]

        m = fft.irfft(temp, n = 2*seq_len, dim = -1) # [batch_size, memory_size, seq_len+1]
        m = m[:, :, :seq_len] # [batch_size, memory_size, seq_len]
        m = m.permute(0, 2, 1) # [batch_size, seq_len, memory_size]

        # Equation 20 of the paper (W_m@m + W_x@x  W@[m;x])
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_size + input_size]
        h = self.f_h(self.W_h(input_h)) # [batch_size, seq_len, hidden_size]

        h_n = h[:, -1, :] # [batch_size*T, hidden_size]

        return h, h_n
    
    def forward_recurrent(self, x, m_last):
        u = self.f_u(self.W_u(x)) # [batch_size, seq_len, 1]
        # A: torch.Size([512, 512]), m_last: torch.Size([256, 512]), B: torch.Size([512, 1]), u: torch.Size([256, 1])
        m = m_last @ self.A.T + u @ self.B.T  # [batch_size, memory_size]
        input_h = torch.cat((m, x), dim = -1) # [batch_size, seq_len, memory_size + input_size]
        h = self.f_h(self.W_h(input_h)) # [batch_size, seq_len, hidden_size]

        return h, m

class LMU(nn.Module):
    def __init__(self, dim, T, use_all_h=True):
        super().__init__()
        self.dim = dim
        self.hidden_size = dim
        self.memory_size = dim
        self.use_all_h = use_all_h
        self.lmu = LMUFFTCell(input_size=dim, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=T, theta=T)
        # self.lmu = LMUFFTCell(input_size=dim, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=64, theta=64)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        x = x.transpose(-1,-2).contiguous() # B, C, N -> B, N, C
        h, _ = self.lmu(x) # B, N, C; B, C
        
        x = h.transpose(-1,-2).contiguous() #if self.use_all_h else h_n.unsqueeze(-1) # h or h_n

        x = self.proj_conv(x)
        x = self.proj_bn(x)

        return x

class LinearFFN(nn.Module):
    def __init__(self, in_features, pre_norm=False, hidden_features=None, out_features=None, drop=0., act_type='spike'):
        super().__init__()
        self.pre_norm = pre_norm
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_linear  = nn.Linear(in_features, hidden_features)
        self.fc1_ln = nn.LayerNorm(hidden_features)
        self.fc1_lif = get_act(act_type if act_type == 'spike' else 'gelu', tau=2.0, detach_reset=True)

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_ln = nn.LayerNorm(out_features)
        self.fc2_lif = get_act(act_type, tau=2.0, detach_reset=True)
 
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        B,C,N = x.shape
        # 
        x = x.permute(0,2,1) # B, N, C
        # x = x.reshape(B*N, C)
        if self.pre_norm:
            x = self.fc1_ln(x)
            x = self.fc1_lif(x)
            x = self.fc1_linear(x)
            
            x = self.fc2_ln(x)
            x = self.fc2_lif(x)
            x = self.fc2_linear(x)

        else:
            x = self.fc1_linear(x)
            x = self.fc1_ln(x)
            x = self.fc1_lif(x)

            x = self.fc2_linear(x)
            x = self.fc2_ln(x)
            x = self.fc2_lif(x)

        # x = x.reshape(B, N, self.c_output)
        x = x.permute(0,2,1) # B, C, N
        return x
    
class Block(nn.Module):
    def __init__(self, dim, T, mlp_ratio=4., act_type='spike'):
        super().__init__()

        self.attn = LMU(dim=dim, T=T)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_type=act_type)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class perm(nn.Module):
    def __init__(self, a, b, c) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        return x.permute(self.a,self.b,self.c).contiguous()

def get_act(act_type = 'spike', **act_params):
    '''
    act_type :- spike, gelu, relu, identity

    output :- class <act_type>
    '''
    act_type = act_type.lower()
    # if act_type == 'spike':
    #     return MultiStepLIFNode(**act_params, backend='cupy')
    #     # act_params['init_tau'] = act_params.pop('tau')
    #     # return MultiStepParametricLIFNode(**act_params, backend="cupy")
    if act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'identity':
        return nn.Identity()
    
def get_conv_block(T, dim, act_type, kernel_size=3, padding=1, groups=1):
    return [
        perm(0,2,1),
        nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, padding=padding, groups=groups, bias=False),
        nn.BatchNorm1d(dim),
        perm(1,2,0),
        get_act(act_type, tau=2.0, detach_reset=True),
        perm(2,1,0)
]

class Conv1d4EB(nn.Module):
    def __init__(self, T=128, vw_dim=256, act_type='spike'):
        super().__init__()

        kernel_size = 3
        padding = 1
        groups = 1
        self.proj_conv = nn.ModuleList(
            [perm(0,2,1)]+\
            get_conv_block(T, vw_dim, act_type)+\
            get_conv_block(T, vw_dim, act_type, kernel_size=kernel_size, padding=padding, groups=groups)+\
            get_conv_block(T, vw_dim, act_type, kernel_size=kernel_size, padding=padding, groups=groups)+\
            get_conv_block(T, vw_dim, act_type, kernel_size=kernel_size, padding=padding, groups=groups)+\
            [perm(0,2,1)]
        )
        self.rpe_conv = nn.ModuleList(
            [perm(0,2,1)]+\
            get_conv_block(T, vw_dim, act_type, kernel_size=kernel_size, padding=padding, groups=groups)+\
            [perm(0,2,1)]
        )
        self.act_loss = 0.0
        
    def forward(self, x):

        for ele in self.proj_conv:
            x = ele(x)

        x_rpe = x.clone()
        for ele in self.rpe_conv:
            x_rpe = ele(x_rpe)

        x = x + x_rpe
        
        return x 

class LMU_RNN(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, act_type='relu', T=784, test_mode='all_seq',with_head_lif=False):
        super().__init__()
        self.with_head_lif = with_head_lif
        self.test_mode = test_mode

        self.in_layer = nn.Linear(input_size, hidden_size)

        self.patch_embed = Conv1d4EB(T=T, vw_dim=hidden_size, act_type=act_type)

        self.block = nn.ModuleList([
            Block(dim=hidden_size, T=T, act_type=act_type)
            for j in range(num_layers)
        ])

        # classification head
        if self.with_head_lif:
            self.head_bn = nn.BatchNorm1d(hidden_size)
            self.head_lif = get_act(act_type, tau=2.0, detach_reset=True)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)
        return x

    def forward(self, x):
        self.act_loss = 0.0
        x = self.in_layer(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.forward_features(x)    # b, d, t -> b, d, t

        if self.with_head_lif:
            x = self.head_bn(x)         # b, d, t 
            x = self.head_lif(x)        # b, d, t

        x = x.permute(0, 2, 1).contiguous()
        
        return x

class MnistLMU(nn.Module):
    r"""
    Very Simple 2 layer LSTM with an fc layer on last steps hidden dimension
    """
    def __init__(self, input_size, hidden_size, codebook_size, T=32):
        super(MnistLMU, self).__init__()
        # self.rnn = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True)
        self.rnn = LMU_RNN(input_size=2, hidden_size=128, num_layers=2, T=T)

        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size//4),
                                nn.ReLU(),
                                nn.Linear(hidden_size//4, codebook_size))
        # Add pad and start token to embedding size
        self.word_embedding = nn.Embedding(codebook_size+2, input_size)
    
    def forward(self, x):
        x = self.word_embedding(x)
        output = self.rnn(x)
        output = output[:, -1, :]
        return self.fc(output)

class MnistSeqDataset(Dataset):
    r"""
    Dataset for training of LSTM. Assumes the encodings are already generated
    by running vqvae inference
    """
    def __init__(self, config):
        self.codebook_size = config['model_params']['codebook_size']
        
        # Codebook tokens will be 0 to codebook_size-1
        self.start_token = self.codebook_size
        self.pad_token = self.codebook_size+1
        # Fix context size
        self.context_size = 64
        self.sents = self.load_sents(config)
    
    def load_sents(self, config):
        assert os.path.exists(os.path.join(config['train_params']['task_name'],
                                           config['train_params']['output_train_dir'],
                                           'mnist_encodings.pkl')), ("No encodings generated for lstm."
                                                                     "Run save_encodings method in inference script")
        mnist_encodings = pickle.load(open(os.path.join(config['train_params']['task_name'],
                                           config['train_params']['output_train_dir'],
                                           'mnist_encodings.pkl'), 'rb'))
        mnist_encodings = mnist_encodings.reshape(mnist_encodings.size(0), -1)
        num_encodings = mnist_encodings.size(0)
        padded_sents = []
        
        for encoding_idx in tqdm(range(num_encodings)):
            # Use only 10% encodings.
            # Uncomment this for getting some kind of output quickly validate working
            if random.random() > 0.1:
                continue
            enc = mnist_encodings[encoding_idx]
            encoding_length = enc.shape[-1]
            
            # Make sure all encodings start with start token
            enc = torch.cat([torch.ones((1)).to(device) * self.start_token, enc.to(device)])
            
            # Create batches of context sized inputs(if possible) and target
            sents = [(enc[:i], enc[i]) if i < self.context_size else (enc[i - self.context_size:i], enc[i])
                   for i in range(1, encoding_length+1)]
            
            for context, target in sents:
                # Pad token if context not enough
                if len(context) < self.context_size:
                    context = torch.nn.functional.pad(context, (0, self.context_size-len(context)), "constant", self.pad_token)
                padded_sents.append((context, target))
        return padded_sents
    
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, index):
        context, target = self.sents[index]
        return context, target

def train_lstm(args):
    ############ Read the config #############
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    #########################################
    
    ############## Create dataset ###########
    mnist = MnistSeqDataset(config)
    mnist_seq_loader = DataLoader(mnist, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=0)
    #########################################
    
    ############## Create LSTM ###########
    default_lstm_config = {
            'input_size' : 2,
            'hidden_size' : 128,
            'codebook_size' : config['model_params']['codebook_size']
    }
    if config['model_params']['rnn_type']=='lstm':
        model = MnistLSTM(input_size=default_lstm_config['input_size'],
                        hidden_size=default_lstm_config['hidden_size'],
                        codebook_size=default_lstm_config['codebook_size']).to(device)
    elif config['model_params']['rnn_type']=='lmu':
        model = MnistLMU(input_size=default_lstm_config['input_size'],
                        hidden_size=default_lstm_config['hidden_size'],
                        codebook_size=default_lstm_config['codebook_size'],
                        T=64).to(device)
    model.to(device)
    model.train()
    
    ############## Training Params ###########
    num_epochs = 10
    optimizer = Adam(model.parameters(), lr=1E-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        losses = []
        for sent, target in tqdm(mnist_seq_loader):
            sent = sent.to(device).long()
            target = target.to(device).long()
            optimizer.zero_grad()
            pred = model(sent)
            loss = torch.mean(criterion(pred, target))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Epoch {} : {}'.format(epoch, np.mean(losses)))
        print('=' * 50)
        torch.save(
            model.state_dict(), 
            os.path.join(
                config['train_params']['task_name'],
                f'best_mnist_{config['model_params']['rnn_type']}.pth',
            )
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for lstm training')
    parser.add_argument('--config', dest='config_path',
                        default='./config/vqvae_colored_mnist.yaml', type=str)
    args = parser.parse_args()
    train_lstm(args)