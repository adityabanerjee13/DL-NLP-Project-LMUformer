import torch
import torch.nn as nn
import torch.nn.functional as F
from lmu_cell import SpikingLMUFFTCell, LMUFFTCell

from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepParametricLIFNode



def get_act(act_type = 'spike', **act_params):
    act_type = act_type.lower()
    if act_type == 'spike':
        return MultiStepLIFNode(**act_params, backend='cupy')
        # act_params['init_tau'] = act_params.pop('tau')
        # return MultiStepParametricLIFNode(**act_params, backend="cupy")
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'gelu':
        return nn.GELU()
    elif act_type == 'identity':
        return nn.Identity()
    

class LMU(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_all_h=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.hidden_size = dim
        self.memory_size = dim
        self.use_all_h = use_all_h
        self.lmu = LMUFFTCell(input_size=dim, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=128, theta=128)
        # self.lmu = LMUFFTCell(input_size=dim, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=64, theta=64)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.act_loss = torch.tensor(0.0)

    def forward(self, x):
        x = x.transpose(-1,-2).contiguous() # B, C, N -> B, N, C
        h, h_n = self.lmu(x) # B, N, C; B, C

        x = h.transpose(-1,-2).contiguous() if self.use_all_h else h_n.unsqueeze(-1) # h or h_n

        x = self.proj_conv(x)
        x = self.proj_bn(x)

        return x
    def forward_recurrent(self, x, state = None):
        batch_size = x.size(0) # B, C, N
        seq_len = x.size(-1)

        # Initial state (h_0, m_0)
        if state == None:
            m_0 = torch.zeros(batch_size, self.memory_size).to(x.device)
            state = m_0

        # Iterate over the timesteps
        output = []
        for t in range(seq_len):
            x_t = x[:, :, t] # [batch_size, input_size]
            h_t, m_t = self.lmu.forward_recurrent(x_t, state)
            state = m_t
            output.append(h_t)

        output = torch.stack(output) # [seq_len, batch_size, hidden_size]

        x = output.permute(1, 2, 0) if self.use_all_h else state[0].unsqueeze(-1) # state is (h_n, m_n) where n = seq_len
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        return x 

class SLMU(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_all_h=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.use_all_h = use_all_h
        self.lmu = SpikingLMUFFTCell(input_size=dim, hidden_size=dim, memory_size=dim*2, seq_len=128, theta=128)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = get_act('spike', tau=2.0, detach_reset=True)

    def forward(self, x):
        x = x.transpose(-1,-2).contiguous()
        h, h_n = self.lmu(x) # B, N, C; B, C, 1

        x = h.transpose(-1,-2).contiguous() if self.use_all_h else h_n.unsqueeze(-1) # h or h_n


        x = self.proj_conv(x)
        x = self.proj_lif(self.proj_bn(x).permute(2,1,0).contiguous()).permute(2,1,0).contiguous()

        return x


class SLMUMs(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, use_all_h=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.hidden_size = dim
        self.memory_size = int(dim*2.0)
        self.use_all_h = use_all_h
        self.lmu = SpikingLMUFFTCell(input_size=dim, hidden_size=self.hidden_size, memory_size=self.memory_size, seq_len=128, theta=128) 

        # v_threshold=0.5 is slightly better
        self.prev_bn = nn.BatchNorm1d(dim)
        self.prev_lif = get_act('spike', tau=2.0, v_threshold=0.5, detach_reset=True)

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.act_loss = 0.0

    def cal_act_loss(self, x):
        return torch.sum(torch.abs(x))
    
    def forward(self, x):
        B, C, N = x.shape
        x = self.prev_bn(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.prev_lif(x).permute(2,0,1).contiguous()# N, C, B -> B, N, C 
        self.act_loss = self.cal_act_loss(x)
        
        # x = F.relu(x)
        h, h_n = self.lmu(x) # B, N, C; B, C, 1
        self.act_loss += self.lmu.act_loss
        x = h.transpose(-1,-2).contiguous() if self.use_all_h else h_n.unsqueeze(-1) # h or h_n


        x = self.proj_conv(x)
        x = self.proj_bn(x)

        return x
    # def forward(self, x, state = None):
    def forward_recurrent(self, x, state = None):
        batch_size = x.size(0) # B, C, N
        seq_len = x.size(-1)
        x = self.prev_bn(x).permute(2,1,0).contiguous() # B, C, N -> N, C, B
        x = self.prev_lif(x).permute(2,0,1).contiguous()# N, C, B -> B, N, C
        # Initial state (h_0, m_0)
        if state == None:
            m_0 = torch.zeros(batch_size, self.memory_size).to(x.device).unsqueeze(1)
            state = m_0

        # Iterate over the timesteps
        output = []
        for t in range(seq_len):
            x_t = x[:, 0:t+1, :] # [batch_size, input_size]
            h_t, m_t = self.lmu.forward_recurrent(x_t, state)
            state = torch.cat([m_0, m_t], dim=1)

        output = h_t

        x = output.permute(0, 2, 1).contiguous() if self.use_all_h else output[:,-1,:].permute(0, 2, 1).contiguous() # state is (h_n, m_n) where n = seq_len
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        return x 
