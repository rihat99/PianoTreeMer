import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.distributions import Normal
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import random
import numpy as np
from model import VAE as VAE_base


class AbsolutePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=16):
        super(AbsolutePositionalEncoder, self).__init__()
        self.position = torch.arange(max_position).unsqueeze(1)

        self.positional_encoding = torch.zeros(1, max_position, emb_dim)

        _2i = torch.arange(0, emb_dim, step=2).float()

        # PE(pos, 2i) = sin(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 0::2] = torch.sin(self.position / (10000 ** (_2i / emb_dim)))

        # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        self.positional_encoding[0, :, 1::2] = torch.cos(self.position / (10000 ** (_2i / emb_dim)))

    def forward(self, x):
        # batch_size, input_len, embedding_dim
        batch_size, seq_len, _ = x.size()

        return self.positional_encoding[:batch_size, :seq_len, :]


class VAETR(nn.Module):
    def __init__(self, 
                 max_simu_note=16, 
                 max_pitch=127, 
                 min_pitch=0,
                 pitch_sos=128, 
                 pitch_eos=129, 
                 pitch_pad=130,
                 dur_pad=2, 
                 dur_width=5, 
                 device=None, 
                 num_step=32,
                 note_emb_size=128,
                 note_embed_hid_size=32,
                 enc_notes_hid_size=256,
                 enc_time_hid_size=512, 
                 z_size=512,  
                 num_enc_note_layers = 8, 
                 num_enc_time_layers = 4
                 ):
        
        super(VAETR, self).__init__()

        # Parameters
        # note and time
        self.max_pitch = max_pitch  # the highest pitch in train/val set.
        self.min_pitch = min_pitch  # the lowest pitch in train/val set.
        self.pitch_sos = pitch_sos
        self.pitch_eos = pitch_eos
        self.pitch_pad = pitch_pad
        self.note_embed_hid_size = note_embed_hid_size
        self.pitch_range = max_pitch - min_pitch + 3  # 88, not including pad.
        self.dur_pad = dur_pad
        self.dur_width = dur_width
        self.note_size = self.pitch_range + dur_width
        self.max_simu_note = max_simu_note  # the max # of notes at each ts.
        self.num_step = num_step  # 32

        # device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device


        # model_size
        # Used for both encoder & decoder
        self.note_emb_size = note_emb_size
        self.z_size = z_size
        # encoder
        self.enc_notes_hid_size = enc_notes_hid_size
        self.enc_time_hid_size = enc_time_hid_size

        # self.pos_encoder_note = AbsolutePositionalEncoder(emb_dim = self.note_embed_hid_size, max_position = self.max_simu_note)
        self.pos_encoder_time = AbsolutePositionalEncoder(emb_dim = self.enc_time_hid_size, max_position = self.num_step)      

        # Modules
        # For both encoder and decoder
        self.note_embedding = nn.Linear(self.note_size, self.note_emb_size)
            
        # self.transformer_projection = nn.Linear(self.note_emb_size + self.note_embed_hid_size, self.note_embed_hid_size)
        self.transformer_projection = nn.Linear(self.note_emb_size, self.note_embed_hid_size)
        
        #changed
        self.encoder_layer_note = TransformerEncoderLayer(
            # d_model=self.note_embed_hid_size + self.note_emb_size, 
            d_model=self.note_emb_size, 
            nhead=4, batch_first=True, 
            dim_feedforward=self.note_embed_hid_size + self.note_emb_size
            # dim_feedforward=self.note_emb_size
        )
        self.encoder_layer_time = TransformerEncoderLayer(d_model=self.enc_time_hid_size, nhead=8, batch_first=True, dim_feedforward=self.enc_time_hid_size)
        self.enc_notes_tr = TransformerEncoder(self.encoder_layer_note, num_layers=num_enc_note_layers)
        self.enc_time_tr = TransformerEncoder(self.encoder_layer_time, num_layers=num_enc_time_layers)

        self.enc_note_token = nn.Parameter(torch.randn(32))
        # self.enc_time_token = nn.Parameter(torch.randn(2, self.enc_time_hid_size))
        self.linear_enc_time = nn.Linear(self.enc_time_hid_size, 32)

        #finish change
        self.linear_mu = nn.Linear(2 * enc_time_hid_size, z_size)
        self.linear_std = nn.Linear(2 * enc_time_hid_size, z_size)

        # self.decoder = self.model.decoder_

        # decoder
        self.dec_time_token = nn.Parameter(torch.randn(32, 512))
        self.decoder_layer_time = TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True, dim_feedforward=512)
        self.dec_time_tr = TransformerDecoder(self.decoder_layer_time, num_layers=4)

        self.dec_note_token = nn.Parameter(torch.randn(16, 32))
        self.decoder_layer_note = TransformerDecoderLayer(d_model=32, nhead=4, batch_first=True, dim_feedforward=32)
        self.dec_notes_tr = TransformerDecoder(self.decoder_layer_note, num_layers=8)

        self.linear_pitch = nn.Linear(32, self.pitch_range)
        self.linear_dur = nn.Linear(32, self.dur_width * 2)

    def encoder(self, x, lengths):

        embedded = self.note_embedding(x)
        # projection = self.transformer_projection(embedded)

        x = embedded.view(-1, self.max_simu_note, self.note_emb_size)
        lengths_cpu = lengths.view(-1).cpu()
        mask = torch.arange(self.max_simu_note).expand(len(lengths_cpu), self.max_simu_note)>=lengths_cpu.unsqueeze(1)
        mask = mask.to(self.device) # mask such that padded notes do not attend in attention
       
        # x = x + self.pos_encoder_note(x).to(self.device)
        # token_note = self.enc_note_token.unsqueeze(0).unsqueeze(0).repeat(x.size(0),x.size(1),1).to(self.device)
        # x = torch.cat([x, token_note], dim = 2)

        
        x = self.enc_notes_tr(src = x, src_key_padding_mask = mask)

        x = self.transformer_projection(x)

        x = x.view(-1, self.num_step, self.enc_time_hid_size)
        x = x + self.pos_encoder_time(x).to(self.device)
        # token_time = self.enc_time_token.unsqueeze(0).repeat(x.size(0),1,1).to(self.device)
        # x = torch.cat([x, token_time], dim = 1)

        # mask_att = (torch.triu(torch.ones(32, 32)) == 1).transpose(0, 1)
        # mask_att = mask_att.float().masked_fill(mask_att == 0, float('-inf')).masked_fill(mask_att == 1, float(0))
        # mask_att = mask_att.to(self.device)

        x = self.enc_time_tr(x)
        # x = x[:, [0, -1], :].view(x.size(0), -1)
        # x = x[:, -2:, :].view(x.size(0), -1)
        x = self.linear_enc_time(x)
        x = x.view(-1, 1024)
 
        mu = self.linear_mu(x)  # (B, z_size)
        std = self.linear_std(x).exp_()  # (B, z_size)

        dist = Normal(mu, std)
        return dist, embedded
    
    def decoder(self, z):
        z = z.view(-1, 1, self.z_size)

        z = z.repeat(1, 32, 1)
        time_input = self.dec_time_token.unsqueeze(0).repeat(z.size(0),1,1).to(self.device)

        # pass through decoder
        # matrix size of mask (B, 32, 32) where upper triangular matrix for each batch

        mask = (torch.triu(torch.ones(32, 32)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        mask = mask.to(self.device)

        z = self.dec_time_tr(tgt=time_input, memory=z, memory_mask = mask)

        # z = self.dec_time_tr(tgt=time_input, memory=z)

        z = z.view(-1, 512)
        z = z.view(-1, 16, 32)

        note_input = self.dec_note_token.unsqueeze(0).repeat(z.size(0),1,1).to(self.device)

        # pass through decoder

        mask = (torch.triu(torch.ones(16, 16)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        mask = mask.to(self.device)

        z = self.dec_notes_tr(tgt=note_input, memory=z, memory_mask = mask)
        # z = self.dec_notes_tr(tgt=note_input, memory=z)

        z = z.view(-1, 32, 16, 32)
        z = z[:, :, 1:, :]

        pitch_outs = self.linear_pitch(z)
        dur_outs = self.linear_dur(z)
        # print(dur_outs.size())
        dur_outs = dur_outs.view(-1, 32, 15, 5, 2)

        return pitch_outs, dur_outs
    
    def get_len_index_tensor(self, ind_x):
        """Calculate the lengths ((B, 32), torch.LongTensor) of pgrid."""
        with torch.no_grad():
            lengths = self.max_simu_note - \
                      (ind_x[:, :, :, 0] - self.pitch_pad == 0).sum(dim=-1)
        return lengths

    def index_tensor_to_multihot_tensor(self, ind_x):
        """Transfer piano_grid to multi-hot piano_grid."""
        # ind_x: (B, 32, max_simu_note, 1 + dur_width)
        with torch.no_grad():
            dur_part = ind_x[:, :, :, 1:].float()
            out = torch.zeros([ind_x.size(0) * self.num_step * self.max_simu_note,
                               self.pitch_range + 1],
                              dtype=torch.float).to(self.device)

            out[range(0, out.size(0)), ind_x[:, :, :, 0].view(-1)] = 1.
            out = out.view(-1, 32, self.max_simu_note, self.pitch_range + 1)
            # print("Shape of out: ", out.size())
            out = torch.cat([out[:, :, :, 0: self.pitch_range], dur_part],
                            dim=-1)
        return out
    
    def forward(self, 
                x, 
                inference=False, 
                sample=True,
                teacher_forcing_ratio1=0.5, 
                teacher_forcing_ratio2=0.5
        ):
        lengths = self.get_len_index_tensor(x)
        x = self.index_tensor_to_multihot_tensor(x)
        dist, embedded_x = self.encoder(x, lengths)
        if sample:
            z = dist.rsample()
        else:
            z = dist.mean
        
        pitch_outs, dur_outs = self.decoder(z)

        return pitch_outs, dur_outs, dist



