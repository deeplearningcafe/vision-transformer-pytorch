import torch
import torch.nn as nn
import omegaconf
import re
from safetensors.torch import load_file


class LinearEmbedding(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(LinearEmbedding, self).__init__()
        
        self.patch_dim = conf.vit.patch_dim
        self.seq_len = (conf.vit.dim**2 // self.patch_dim**2)
        self.input_channels = conf.vit.input_channels
        self.hidden_dim = conf.vit.hidden_dim
        self.embedding = nn.Conv2d(in_channels=self.input_channels, out_channels=self.hidden_dim,
                                   kernel_size=self.patch_dim, stride=self.patch_dim)
        
        self.class_token = nn.Parameter(torch.randn((1, 1, self.hidden_dim)))
        # we use broadcasting here, embedding output is [batch, seq_len, hidden_dim]
        self.positional_embeds = nn.Parameter(torch.zeros((1, self.seq_len + 1, self.hidden_dim)))

        #  Dropout, when used, is applied after every dense layer except for the the qkv-projections and directly after adding positional- to patch embeddings
        self.dropout = nn.Dropout(p=conf.vit.dropout)
        
    def forward(self, x: torch.tensor):
        # [B, C, H, W]
        out = self.embedding(x)
        # [B, D, H*W/PATCH, H*W/PATCH]
        out = out.view(out.shape[0], out.shape[1], -1).transpose(1, 2)

        # [B, SEQ, D]
        out = torch.concat([self.class_token.repeat(out.shape[0], 1, 1), out], dim=1)

        out += self.positional_embeds
        
        return self.dropout(out)
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(MultiHeadSelfAttention, self).__init__()
    
        self.num_heads = conf.vit.num_heads
        self.head_dim = conf.vit.hidden_dim//self.num_heads
        self.hidden_dim = conf.vit.hidden_dim
        self.q = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.k = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.v = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True)
        
        self.projection = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(p=conf.vit.dropout)
        self.attention_dropout = nn.Dropout(p=conf.vit.attention_dropout)
        
    def forward(self, x: torch.tensor, attention_mask: torch.tensor=None):
        B, N, D = x.shape
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)
        
        # apply multi head, we want [B, num_heads, N, head_dim]
        query = query.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = torch.matmul(query, key.transpose(3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if attention_mask != None:
            attention += attention_mask
        attention_weights = nn.functional.softmax(attention, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # now we need to go back to the normal shape
        out = torch.matmul(attention_weights, value)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.projection(out)
        return self.dropout(out)
        
class LayerNormImplementation(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(LayerNormImplementation, self).__init__()
        
        self.hidden_dim = conf.vit.hidden_dim
        self.gamma = nn.Parameter(torch.ones(self.hidden_dim))
        self.beta = nn.Parameter(torch.zeros(self.hidden_dim))
        self.eps = conf.vit.eps
        
    def forward(self, x: torch.tensor):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)
        
        out = (x-mean) / torch.sqrt(var+self.eps)
        out = out * self.gamma + self.beta

        return out
    
class MLP(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(MLP, self).__init__()
        
        self.hidden_dim = conf.vit.hidden_dim
        self.ff_dim = conf.vit.ff_dim
        
        self.projection_in = nn.Linear(in_features=self.hidden_dim, out_features=self.ff_dim, bias=True)
        self.projection_out = nn.Linear(in_features=self.ff_dim, out_features=self.hidden_dim, bias=True)
        self.activation = nn.GELU()
        
        self.dropout = nn.Dropout(p=conf.vit.dropout)
        
    def forward(self, x: torch.tensor):
        x = self.dropout(self.activation(self.projection_in(x)))
        
        out = self.activation(self.projection_out(x))
        
        return self.dropout(out)
        
class ViTBlock(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(ViTBlock, self).__init__()
        
        self.mhsa = MultiHeadSelfAttention(conf)
        self.ln1 = LayerNormImplementation(conf)
        
        self.mlp = MLP(conf)
        self.ln2 = LayerNormImplementation(conf)
        
    def forward(self, x: torch.tensor, attention_mask: torch.tensor=None):
        
        x = self.mhsa(self.ln1(x), attention_mask) + x
        
        out = self.mlp(self.ln2(x)) + x
        return out
        
class MLPHead(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(MLPHead, self).__init__()

        self.hidden_dim = conf.vit.hidden_dim
        self.num_classes = conf.vit.num_classes
        
        self.predictor = nn.Linear(in_features=self.hidden_dim, out_features=self.num_classes, bias=True)
        self.ln = LayerNormImplementation(conf)
    
    def forward(self, x: torch.tensor):
        # when passing a tensor to a function, the tensor is not copied, it is just a pointer so if we change the tensor
        # inside the function the original tensor outside the function also changes
        out = self.predictor(self.ln(x[:, 0]))

        return out
    
class ViT(nn.Module):
    def __init__(self, conf: omegaconf.DictConfig) -> None:
        super(ViT, self).__init__()
        
        self.num_layers = conf.vit.num_layers
        
        self.embeddings = LinearEmbedding(conf)
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(ViTBlock(conf))
            
        self.mlp_head = MLPHead(conf)
        
        
    def forward(self, x: torch.tensor, attention_mask: torch.tensor=None):
        
        x = self.embeddings(x)
        
        for i in range(len(self.blocks)):
            # we don't need attention masks as we pay attention to all the tokens and always same sequence length
            # so no padding 
            x = self.blocks[i](x, attention_mask)
        
        out = self.mlp_head(x)
        
        return out
    
    def from_pretrained(self, state_dict_path:str, load_head:bool=False):
        state_dict = load_file(state_dict_path)
        
        # Patterns for matching source and target keys
        layer_patterns = [
            (re.compile(r'^vit\.embeddings\.cls_token$'), 'embeddings.class_token'),
            (re.compile(r'^vit\.embeddings\.position_embeddings$'), 'embeddings.positional_embeds'),
            (re.compile(r'^vit\.embeddings\.patch_embeddings\.projection\.weight$'), 'embeddings.embedding.weight'),
            (re.compile(r'^vit\.embeddings\.patch_embeddings\.projection\.bias$'), 'embeddings.embedding.bias'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.attention\.key\.weight$'), 'blocks.{}.mhsa.k.weight'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.attention\.key\.bias$'), 'blocks.{}.mhsa.k.bias'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.attention\.query\.weight$'), 'blocks.{}.mhsa.q.weight'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.attention\.query\.bias$'), 'blocks.{}.mhsa.q.bias'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.attention\.value\.weight$'), 'blocks.{}.mhsa.v.weight'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.attention\.value\.bias$'), 'blocks.{}.mhsa.v.bias'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\.weight$'), 'blocks.{}.mhsa.projection.weight'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\.bias$'), 'blocks.{}.mhsa.projection.bias'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.weight$'), 'blocks.{}.mlp.projection_in.weight'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\.bias$'), 'blocks.{}.mlp.projection_in.bias'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.output\.dense\.weight$'), 'blocks.{}.mlp.projection_out.weight'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.output\.dense\.bias$'), 'blocks.{}.mlp.projection_out.bias'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.layernorm_before\.weight$'), 'blocks.{}.ln1.gamma'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.layernorm_before\.bias$'), 'blocks.{}.ln1.beta'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.layernorm_after\.weight$'), 'blocks.{}.ln2.gamma'),
            (re.compile(r'^vit\.encoder\.layer\.(\d+)\.layernorm_after\.bias$'), 'blocks.{}.ln2.beta'),
        ]
        if load_head:
            layer_patterns.extend([(re.compile(r'^classifier\.weight$'), 'mlp_head.predictor.weight'),
            (re.compile(r'^classifier\.bias$'), 'mlp_head.predictor.bias'),
            (re.compile(r'^vit\.layernorm\.weight$'), 'mlp_head.ln.weight'),
            (re.compile(r'^vit\.layernorm\.bias$'), 'mlp_head.ln.bias'),]
            )

        target_model_state_dict = self.state_dict()

        for source_key, source_value in state_dict.items():
            for pattern, target_template in layer_patterns:
                match = pattern.match(source_key)
                if match:
                    layer_num = match.group(1) if match.groups() else ''
                    target_key = target_template.format(layer_num)
                    if target_key in target_model_state_dict:
                        # print(target_key)
                        assert source_value.shape == target_model_state_dict[target_key].shape, f"bad shapes {source_value.shape, target_model_state_dict[target_key].shape}"
                        target_model_state_dict[target_key] = source_value
                        break
        self.load_state_dict(target_model_state_dict)


