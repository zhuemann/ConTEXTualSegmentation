import torch
import torch.nn as nn

class LangCrossAtt(nn.Module):
    "add documentaiton"

    def __init__(self, emb_dim):
        super(LangCrossAtt, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=1) #vdim=vdimension
        self.tanh = nn.Tanh()

    def forward(self, lang_rep, vision_rep):

        # gets all dimensions to be used in the attention
        input_batch   = vision_rep.size()[0]
        input_channel = vision_rep.size()[1]
        input_width   = vision_rep.size()[2]
        input_height  = vision_rep.size()[3]

        # puts the vision representation into the right shape for attention mechanism
        vision_rep = torch.swapaxes(vision_rep, 0, 1)
        vision_rep_flat = torch.flatten(vision_rep, start_dim=2)
        vision_rep = torch.swapaxes(vision_rep_flat, 2, 0)

        # puts the language rep into the right shape for attention
        lang_rep = torch.swapaxes(lang_rep, 0, 1)

        # does cross attention between vision and language
        att_matrix, attn_output_weights = self.multihead_attn(query=vision_rep, key=lang_rep, value=lang_rep)

        att_matrix = self.tanh(att_matrix)

        vision_rep = vision_rep * att_matrix
        vision_rep = vision_rep.contiguous()

        # rearanges the output matrix to be the dimensions of the input
        out = vision_rep.view(input_width, input_height, input_batch, input_channel)
        out = torch.swapaxes(out, 0, 2)
        out = torch.swapaxes(out, 1, 3)

        return out
        #return out, att_matrix
