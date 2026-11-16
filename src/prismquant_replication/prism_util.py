from __future__ import annotations
#implementing in llama transformer models
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def collect_callibration_moments(
    model: AutoModelForCausalLM,
    dataloader: torch.utils.data.DataLoader,
    device:str
    ) -> dict[str,torch.Tensor]:

    """Extracting the uncentered second moments (sigma) for HF model submodule
    """
    moments = {}
    handles =[]

    def get_hook(name:str):
      def hook(module, input, output):

        x = input[0] if isinstance(input, tuple) else input
        x_flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32) #flatten the input to (-1,d)

        if name not in moments:
          moments[name] = torch.zeros(
              x_flat.shape[-1], x_flat.shape[-1],
              device=device)
        
        moments[name] += torch.matmul(x_flat.T, x_flat)

      return hook

    #Global Resiudal stream hook
    handles.append(
          model.model.layers[0].input_layernorm.register_forward_hook(
              get_hook(f"global_residual")
          )
      )
   
    #registering hooks on pe_layer residual_reading RMSNorms and MLP down-proj inputs
    for layer_idx, layer in enumerate(model.model.layers):
      handles.append(
          layer.mlp.down_proj.register_forward_hook(
              get_hook(f"layer_{layer_idx}_down_input")
          )
      )

    ## runing the calibration forward pass
    model.eval()
    with torch.no_grad():
      for batch in dataloader:
        batch = batch.to(device)
        input = batch.reshape(-1,2048)
        #inputs = inputs["input_ids"].to(device)
        #inputs = inputs.reshape(-1,2048)
        model(input)

    # removing the hooks
    for handle in handles:
      handle.remove()
    return moments



#implementing activation quantization
class PrismQuantQuantizer(nn.Module):
  "Assymetric Int4 group quantizer"

  def __init__(self):
    super().__init__()
    pass

  #assymetric group quantization
  def quantizer_dequantizer(self, data:torch.Tensor,
                        precision:int,
                        group_size:int)->torch.Tensor:
    data_shape = data.shape
    dim = data_shape[-1]
   # group_size=self.group_size
    num_groups = dim//group_size
    normalizer = ((2**precision) -1)**-1
    
    ## reshaping to contigious feature groups (...,num_groups, group_size)
    data_grouped = data.view(-1,num_groups, group_size)

    ##computing the min and max
    min_val = torch.amin(data_grouped, dim=-1, keepdim=True)
    max_val = torch.amax(data_grouped, dim=-1, keepdim=True)

    ## deriving the scale and the zeropoint(z) at fp16
    zeropoint = min_val.to(torch.float16)
    scaling = ((max_val - min_val)*normalizer).to(torch.float16)

    ## setting up the positve fallback scale for constant groups
    scaling = torch.where(scaling<=0, torch.ones_like(scaling), scaling)

    ## convert the parameters back to the original activation dtype
    zeropoint_f = zeropoint.to(data.dtype)
    scaling_f = scaling.to(data.dtype)

    ## mapping the value into precision_bit grid using zeropoint and scaling factor
    quantized = torch.clamp(torch.round((
        data_grouped-zeropoint_f)/scaling_f),min=0, max=normalizer).to(torch.uint8)
    
    #dequantizing the values back to floating point
    dequantized = scaling_f * quantized + zeropoint_f

    return dequantized.view(data_shape)

  def forward(self,data:torch.Tensor, precision:int=4)->torch.Tensor:
    return self.quantizer_dequantizer(data, precision)


## implementing the offline weight quantization
class PrismQuantWeightQuantizer:
    """offline GPTQ weight quantizer"""

    @staticmethod
    def quantize_weight_matrix(
        weight_data:torch.Tensor,
        group_size:int=128,
        precision:int=4
    )->torch.Tensor:
        data_shape = weight_data.shape
        dim = data_shape[-1]
        num_groups = dim//group_size #-> m = d/g
        normalizer = ((2**precision)-1)**-1
        grouped_weight = weight_data.view(-1,num_groups, group_size)

        #computing min and max
        min_val = torch.amin(grouped_weight,dim=-1, keepdim=True)
        max_val = torch.amax(grouped_weight,dim=-1, keepdim=True)

        #computing the scaling factor and zeropoint in float16
        zeropoint = min_val.to(torch.float16)
        scaling = ((max_val-min_val)*normalizer).to(torch.float16)

        #fallback value when scaling factor <=0
        scaling = torch.where(scaling<=0, torch.ones_like(scaling), scaling)

        ##move to data dtype
        zeropoint_f = zeropoint.to(weight_data.dtype)
        scaling_f = scaling.to(weight_data.dtype)

        ## quantizing 
        quantized = torch.clamp(torch.round((grouped_weight-zeropoint_f)/scaling_f),
                                min=0, max=normalizer).to(torch.uint8)

        ##dequantizing weight
        dequantized = (scaling_f* quantized.to(weight_data.dtype)+zeropoint_f).view(
            data_shape
        )

        return dequantized, quantized, (scaling, zeropoint)


class PrismQuantKVCacheQuantizer():
  def __init__(
      self,
      key_group_size:int =32,
      value_group_size:int = 128,
      residual_window:int = 32,
      ):
      super().__init__()
      self.key_group_size = key_group_size
      self.value_group_size = value_group_size
      self.residual_window = residual_window

  def quantize_keys(self, key_rot: torch.Tensor,
                      precision:int =4)->torch.Tensor:
      # shape of the key => (batch, num_heads, seq_len,head_dim)
      seq_len = key_rot.shape[-2]
      normalizer = ((2**precision)-1)**-1


      if seq_len <= self.residual_window:
        return key_rot ## because the recent residual_window tokens is unquantized
      
      key_past = key_rot[:,:,:-self.residual_window,:]
      key_recent = key_rot[:,:, -self.residual_window :, :]

      past_len = key_past.shape[-2]
      num_blocks = past_len//self.key_group_size

      if num_blocks>0:
        usable_len = num_blocks * self.key_group_size
        key_block = key_past[:,:,:usable_len,:].view(
            key_rot.shape[0], key_rot.shape[1],
            num_blocks, self.key_group_size,
            key_rot.shape[-1]
        )

        min_val = torch.amin(key_block, dim=-2, keepdim=True)
        max_val = torch.amin(key_block, dim=-2, keepdim=True)

        zeropoint = min_val.to(torch.float16)
        scaling = ((max_val - min_val)*normalizer).to(torch.float16)
        scaling = torch.where(scaling<=0,torch.ones_like(scaling),scaling)

        zeropoint_f = zeropoint.to(key_block.dtype)
        scaling_f = scaling.to(key_block.dtype)

        quantized = torch.clamp((key_block - zeropoint_f)*scaling_f**-1, 
                                min=0, max=normalizer)
        dequantized_usable = (scaling_f * quantized + zeropoint_f).view(
            key_rot.shape[0],key_rot.shape[1], usable_len, key_rot.shape[-1]
        )

        remainder_key = key_past[:,:,usable_len:, :]
        key_past_hat = torch.cat([dequantized_usable, remainder_key])

      else:
        key_past_hat = key_past
      
      return torch.cat([key_past_hat, key_recent], dim=2)

  def quantize_values(self, 
                      value_rot:torch.Tensor,
                    precision:int=4)->torch.Tensor:
    """Value Per token quantization  """

    seq_len = value_rot.shape[-2]
    normalizer = ((2**n)-1)**-1
    if seq_len <= self.residual_window:
      return value_rot

    value_past = value_rot[:,:,:-self.residual_window,:]
    value_recent = value_rot[:,:,-self.residual_window,:]

    min_val = torch.amin(value_past, dim=-1, keepdim=True)
    max_val = torch.amax(value_past, dim=-1, keepdim=True)

    zeropoint = min_val.to(torch.float16)
    scaling = ((max_val-min_val)*normalizer).to(torch.float16)
    scaling = torch.where(scaling<=0, torch.ones_like(scaling),scaling)

    zeropoint_f, scaling_f = zeropoint.to(value_rot.dtype), scaling.to(value_rot.dtype)
    quantized = torch.clamp(torch.round(value_past-zeropoint_f)*scaling**-1, 
                            min=0, max=normalizer)
    
    dequantized = scaling_f * quantized + zeropoint_f

    return torch.cat([dequantized, value_recent], dim=2)

