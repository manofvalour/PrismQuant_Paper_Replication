from __future__ import annotations
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import math
from prism_util import PrismQuantQuantizer, PrismQuantKVCacheQuantizer

class PrismQuantLlamaLinear(nn.Module):
  def __init__(self, original_linear:nn.Linear, 
               quantizer:PrismQuantQuantizer,
               group_size:int=128,
               precision:int=4):
    super().__init__()
    self.in_feature = original_linear.in_features
    self.out_feature = original_linear.out_features
    self.group_size = group_size
    self.quantizer = quantizer
    self.precision = precision

    self.register_buffer('weight', original_linear.weight.data)
    self.bias = (original_linear.bias 
                 if original_linear.bias is not None else None)
    
  def forward(self, x:torch.Tensor)-> torch.Tensor:
    quantized_x = self.quantizer(x,precision=self.precision,
                                 group_size=self.group_size)
    return F.linear(quantized_x, self.weight, self.bias)



class PrismQuantLlamaAttention(nn.Module):
  def __init__(self, original_attn:nn.Module,
               kv_quantizer:PrismQuantKVCacheQuantizer,
               group_size:int=128,
               precision:int=4
               ):
    super().__init__()
    self.hidden_size= original_attn.hidden_size
    self.kv_quantizer = kv_quantizer
    self.num_heads = original_attn.num_heads
    self.num_key_value_heads = original_attn.num_key_value_heads
    self.head_dim = original_attn.head_dim
    self.num_key_value_group = self.num_heads//self.numb_key_value_heads
    self.group_size = group_size
    self.precision - precision

    ##wraping in linear projections
    self.q_proj = PrismQuantLlamaLinear(
        original_attn.q_proj,group_size=group_size,
        precision=precision
    )
    self.k_proj = PrismQuantLlamaLinear(
        original_attn.k_proj,group_size=group_size,
        precision=precision
    )
    self.v_proj = PrismQuantLlamaLinear(
        original_attn.v_proj,group_size=group_size,
        precision=precision
    )
    self.o_proj = PrismQuantLlamaLinear(
        original_attn.o_proj,group_size=group_size,
        precision=precision
    )

  def forward(
      self, hidden_states: torch.Tensor,
      positional_embedding:tuple[torch.Tensor, torch.Tensor],
      past_key_value = None,
      **kwargs
    ):

    batch, seq_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(
        batch, seq_len, self.num_heads, self.head_dim
    )
    k = self.k_proj(hidden_states).view(
        batch, seq_len, self.num_key_heads, self.head_dim
    )
    v = self.v_proj(hidden_states).view(
        batch, seq_len, self.num_key_heads, self.head_dim
    )

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    ## applying Rotary Positional Encoding
    cos,sin = positional_embedding
    q,k = apply_rotary_pos_emb(q,k,cos,sin)

    #KV cache handling and KIVi Quant
    if past_key_value is not None:
      k,v = past_key_value.update(k,v,layer_idx=0)
    
    #quantizing the key_value
    quantized_key = self.kv_quantizer.quantize_keys(k,self.precision)
    quantized_val = self.kv_quantizer.quantize_values(v, self.precision)

    #handling GQA
    if self.num_key_value_groups >1:
      quantized_key = (quantized_key.unsqueeze(2).
                       expand(batch,
                              self.num_key_value_heads,
                              self.num_key_value_groups,
                              -1,
                              self.head_dim)).reshape(
                                  batch, self.num_heads, 
                                  -1, self.head_dim
                              )

      quantized_value = (quantized_value.unsqueeze(2).
                         expand(batch,
                                self.num_key_value_heads,
                                self.num_key_value_groups,
                                -1,
                                self.head_dim)).reshape(
                                    batch,self.num_heads,
                                    -1,
                                    self.head_dim
                                )

      attn_weights = torch.matmul(q,quantized_key.transpose(2,3))/math.sqrt(self.head_dim)
      attn_weights = F.softmax(attn_weights, dim=-1,dtype = torch.float32).to(q.dtype)
      attn_output = torch.matmul(attn_weights, quantized_value)

      attn_output = attn_output.transpose(1,2).contiguous().view(batch,seq_len,-1)

      return self.o_proj(attn_output),past_key_value


class PrismQuantDownProjWrapper(nn.Module):
  def __init__(
      self,
      original_down_proj: nn.Linear,
      W_tilde: torch.Tensor,
      Y_tilde: torch.Tensor,
      group_size: int = 128
  ):
    super().__init__()
    self.down_proj = original_down_proj
    self.register_buffer("W_tilde",W_tilde) #dim,k
    self.register_buffer('Y_tilde',Y_tilde) #dim,k
    self.group_size = group_size

  def apply_block_hadamard(self, x:torch.Tensor,group_size)-> torch.Tensor:
    m = x.shape[-1]//group_size
    n = math.log2(m)
    def _walsh_hadamard(n:int):
      if n==0:
        return torch.tensor([[1]], device=x.device)

      else:
        walsh_hadamard_base = _walsh_hadamard(n-1)
        return torch.cat(
            [torch.cat([walsh_hadamard_base, walsh_hadamard_base], dim=1),
            torch.cat([walsh_hadamard_base, -walsh_hadamard_base], dim=1)],
            dim=0
        )

    #applying walsh_hadamard and Range Absorption
    normalized_walsh = (group_size**-0.5) * _walsh_hadamard(n)
    block_walsh = torch.block_diag(*[normalized_walsh for _ in range(m)]).to(x.dtype)

    return x@block_walsh


  def forward(self, x:torch.Tensor)->torch.Tensor:
    U = torch.matmul(x,self.Y_tilde) #U = xY
    x_house = x - torch.matmul(U, self.W_tilde.T) # x-UW -> X-(XY)W_transpose

    #block_hadamard mixing: x_rotation = x_household_refleciton * H_g
    x_rot = self.apply_block_hadamard(x_house)

    return self.down_proj(x_rot)


## fold rotation and signed permutation offline
def fold_prismquant_into_model(
    model: AutoModelForCausalLM,
    R1: torch.Tensor,
    layer_rotations:list[dict],# consist of W,Y,pi_order, D per layer
    quantizer: PrismQuantQuantizer,
    kv_quantizer: PrismQuantKVCacheQuantizer,
    group_size:int,
    precision:int
    ):
  
  ## folding R1 into the embedding and lM layer
  model.model.embed_tokens.weight.data = torch.matmul(
      model.model.tokens.weight.data, R1.T)
  
  model.lm_head.weight.data = torch.matmul(
      model.lm_head.weight.data, R1)
  
  ## folding the rotation into the decode layer
  for layer_idx, layer in enumerate(model.model.layers):
    rot_data = layer_rotations[f"layer_{layer_idx}"]
    pi_order = rot_data['pi_order']
    xi = rot_data['xi'] # the seeded sign

    ## handling the attention block folding R1 into q_proj, k_proj, and v_proj
    layer.self_attn.q_proj.weight.data = torch.matmul(
        layer.self_attn.q_proj.weight.data, R1.T
    )

    layer.self_attn.k_proj.weight.data = torch.matmul(
        layer.self_attn.k_proj.weight.data, R1.T
    )

    layer.self_attn.v_proj.weight.data = torch.matmul(
        layer.self_attn.v_proj.weight.data, R1.T
    )
    layer.self_attn.o_proj.weight.data = torch.matmul(
       R1, layer.self_attn.o_proj.weight.data
    )

    ## handling the MLP block: Folding R1 and signed permutation(T=D@P_i) into the gate and the up projection
    W_gate_R1 = torch.matmul(
        layer.mlp.gate_proj.weight.data, R1.T)
    W_gate_signed = W_gate_R1 * xi.unsqueeze(0)
    layer.mlp.gate_proj.weight.data = W_gate_R1[:pi_order]

    W_up_R1 = torch.matmul(
        layer.mlp.up_proj.weight.data,R1.T)
    W_up_signed = W_up_R1 * xi.unsqueezed(0)
    layer.mlp.up_proj.weight.data = W_up_signed[:,pi_order]

    layer.mlp.down_proj.weight.data = torch.matmul(
        R1, layer.mlp.down_proj.weight.data
    )

    ## ingecting online quantized submodules
    layer.self_attn = PrismQuantLlamaAttention(
        layer.self_attn, 
        kv_quantizer=kv_quantizer,
        group_size= group_size
    )
    layer.mlp.gate_proj = PrismQuantLlamaLinear(
        layer.mlp.gate_proj, 
        quantizer,
        group_size=group_size,
        precision=precision
    )
    layer.mlp.up_proj = PrismQuantLlamaLinear(
        layer.mlp.up_proj, quantizer,
        group_size=group_size,
        precision=precision
    )
    layer.mlp.down_proj = PrismQuantDownProjWrapper(
        original_down_proj=layer.mlp.down_proj,
        W_tilde = rot_data["w_tilde"],
        Y_tilde = rot_data["Y_tilde"],
        group_size=group_size
    )

def prepare_prismquant_model(model_id:str,
               calib_data:DataLoader,
               prism_class: PrismQuant,
               k:int=8, seed=42, group_size=32
               ):
    #launching the prism quant
    #1. load the model
    tokenizer= AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map='cuda')
    
    #2. load the calibration moment
    calib_moments = collect_callibration_moments(model, calib_data)

    #3. compute R1 with the other parameters (W,Y,Pi_order, D)
    layer_rotations = prism_class.prism_quant_params(calib_moments, 
                                            model, group_size=group_size,
                                            k=k, seed=seed)
    #4. fold prismquant into model
    R1 = layer_rotations['global_R1']
    fold_prismquant_into_model(model,R1,layer_rotations, 
                                group_size=group_size)

    return model, tokenizer
    