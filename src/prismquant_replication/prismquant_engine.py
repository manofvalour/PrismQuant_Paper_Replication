from __future__ import annotations

import math
import torch.nn as nn
import torch
class PrismQuant:
  def __init__(
      self):
    pass

  ## computing rotation
  def prism_quant_params(self, 
                          calibration_moment:dict[str,torch.Tensor],
                          model:nn.Module,
                          group_size:int,
                          k:int = 8,
                          seed:int=42):
    
    device= next(model.parameters()).device
    num_layers = len(model.model.layers)
   # n = math.log2(self.group_size) #n =log2(grp_size)
    results ={}

    #computing global rotation (R1)
    sigma_global = calibration_moment['global_residual']
    d_model = sigma_global.shape[0]
    
    ##cord_matrix
    cord_anchor = self._cord_anchor_vec(dim=d_model,
                                        group_size=group_size,
                                        k=k, device=device,
                                        dtype=torch.float16)
    m = d_model//group_size

    #ky_fan
    _, eigenvec = self._ky_fan(sigma_global, k=k,device=device)
    ##household_ref_matrix
    H_list = self._household_ref_matrix(eigenvec, cord_anchor,
                                  dim=d_model, k=k,
                                   group_size = group_size)
    #building compact wy
    W_g,Y_g = self._compact_wy(H_list)
    R1 = torch.eye(d_model, device=device) - torch.matmul(W_g, Y_g.T)
    results['global_R1']=R1

    #computing the per_layer down projetion (R4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for idx in range(num_layers):
      sigma_down = calibration_moment[f"layer_{idx}_down_input"]
      dim_mlp = sigma_down.shape[0]

      ##cord_matrix
      cord_anchor = self._cord_anchor_vec(dim=dim_mlp,
                                        group_size=group_size,
                                        k=k, device=device,
                                        dtype=torch.float16)

      #eigen decomp of MLP activations
      _,eigenvec = self._ky_fan(sigma_down,k=k, device=device)
      #household_ref_matrix
      mlp_h_list = self._household_ref_matrix(eigenvec, cord_anchor,
                                              dim=dim_mlp, k=k,
                                              group_size=group_size)
      
      #compact_wy
      W_tilde, Y_tilde = self._compact_wy(mlp_h_list)

      #computing post reflection coordinate energy
      post_ref_energy = self.post_ref_energy(W_tilde,Y_tilde,
                                      sigma_down, dim=dim_mlp)

      #Residual Balancing
      permutate = self._residual_balance(post_ref_energy,
                              group_size,dim=dim_mlp, k=k)
      #seeded sign flips
      seeded_sign = self._seeded_sign(dim_mlp, generator=generator).to(device)

      results[f"layer_{idx}"]={
          "W_tilde": W_tilde,
          "Y_tilde": Y_tilde,
          "pi_order": permutate,
          "xi": seeded_sign}
    
    return results

  #E = 1/seq_lentotal_size(calib.T*calib)
  def _ky_fan(self, 
              calibration_moment:dict[str,torch.Tensor],
              k:int,
              device:str):
    
    #ncal = self.calibration_dataset.size(0)
    #E = (1/ncal*(self.calibration_dataset.T@self.calibration_dataset)).to(torch.float32) # 1/N(X.T@X)
    e_val, e_vec = torch.linalg.eig(calibration_moment) #eigenval, eigenvec

    ##sorting eigenval and eigenvec
    sort_e_val,indices = torch.sort(e_val.real, descending=True)[:k]
    sorted_e_vec = e_vec[:, indices].to(device)

    return sort_e_val.to(torch.float32), sorted_e_vec.to(torch.float32)


  def _cord_anchor_vec(self,dim:int,
                       group_size:int,
                       k:int,
                       device:str,
                       dtype:str)->torch.Tensor:

    cord_anchor = torch.zeros([dim,k],
                              device =device,
                              dtype=dtype)
    for i in range(k):
      cord_anchor[i*group_size, i]=1
    return cord_anchor

  def _household_ref_matrix(self,
        eigenvector:torch.Tensor,
        cord_anchor:torch.Tensor,
        k:int,
        dim:int,
        group_size:int,
        )->list[torch.Tensor]:

      G = torch.eye(dim,
                    dtype=eigenvector.dtype,
                    device=eigenvector.device)
      H_list= []
      for i in range(k):
        b = torch.matmul(G, eigenvector[:,i]) # dimxdim @dimx1 ->dimx1

        #computing target vectors
        anchor_idx = i*group_size

        sign = -1.0 if b[anchor_idx]>0 else 1.0
        euclidean_norm = torch.linalg.vector_norm(b, dim=0)
        target_anchor = sign* (euclidean_norm)*cord_anchor[:,i]

        #computing difference vector
        diff_vector = b - target_anchor
        normed_diff_vector = torch.linalg.norm(diff_vector, axis=0)

        if normed_diff_vector < 1e-7:
          h = torch.zeros_like(b)

        else:
          h = (diff_vector/normed_diff_vector) #dimx1

        #H_i = I - (2*h)@h.T
        H_i= torch.eye(dim, device=eigenvector.device,
                       dtype=eigenvector.dtype)-2.0*torch.outer(
                           h,h)

        G = H_i@G
        H_list.append(h)

      return H_list

  def _compact_wy(self, h_list:list[torch.Tensor])->torch.Tensor:
    dim = h_list[0].shape[0]
    k= len(h_list)

    Y = torch.zeros([dim,k],
                      device= h_list[0].device,
                      dtype= h_list[0].dtype)
    W = torch.zeros([dim,k],
                      device= h_list[0].device,
                      dtype= h_list[0].dtype)

    #computing compact WY
    for i, h in enumerate(h_list):
      Y[:,i] = h

      if i>0:
        h_col = h.unsqueeze(1)
        proj = torch.matmul(h_col.T, W[:,:i])

        W[:,:i] = W[:,:i]-2.0*torch.matmul(h_col,proj)
      W[:,i]= 2.0*h
    return torch.round(W, decimals=4), torch.round(Y, decimals=4)

  def _online_transform(self,X,W,Y):

    U = torch.matmul(X.to(Y.dtype),Y)
    Xgt = X- U@W.T
    return Xgt

  #computing the post reflection energy
  def post_ref_energy(self,
                      W,Y,E,
                      dim)->torch.Tensor:

    I = torch.eye(dim, device=W.device, dtype=W.dtype)
    G = I-(W@Y.T)
    return torch.diag(G@E@G.T)

  ## permutating nonanchor features
  def _residual_balance(self, residual_alloc,
                          group_size,
                          dim,
                          k):

    num_groups = int(dim/group_size)
    group_layouts = [[] for _ in range(num_groups)] #for storing the assigned index
    used_idx = set() #for keeping track of used

    #assign aligned anchors to slot 0 of the first k groups
    for i in range(k):
      anchor_idx = i*group_size
      group_layouts[i].append(anchor_idx)
      used_idx.add(anchor_idx)

    ## Assigning low energy filler to slot 0 of unanchored groups:
    if k<num_groups:
      all_candidates = [idx for idx in range(dim) if idx not in used_idx]
      all_candidates.sort(key=lambda idx: residual_alloc[idx]) #sorting ascending

      num_unanchored = num_groups-k
      filler_idx = all_candidates[:num_unanchored]

      for idx_pos, group_idx in enumerate(range(k,group_size)):
        filler_feature = filler_idx[idx_pos]
        group_layouts[group_idx].append(filler_feature)
        used_idx.add(filler_feature)

    ## setting up the greedy allocation of the remaining non_anchor features
    group_residual_energies = [0.0]*num_groups

    #sorting remaining candidates in descending by energy
    remaining_candidates = [idx for idx in range(dim) if idx not in used_idx]
    remaining_candidates.sort(key=lambda idx: residual_alloc[idx],reverse=True) #sort descending
    non_anchor_capacity= group_size-1

    for cand_idx in remaining_candidates:
      cand_energy = residual_alloc[cand_idx]

      elig_candid = [
          j for j in range(num_groups) if len(group_layouts[j])-1<non_anchor_capacity
      ]
      best_group = min(elig_candid, key=lambda j: group_residual_energies[j])
      group_layouts[best_group].append(cand_idx)
      group_residual_energies[best_group]+=cand_energy


    ##flaatten the group layouts into final permutations array
    permute_order = torch.tensor([feature for group in group_layouts for feature in group], dtype=torch.int)

    return permute_order


  def _seeded_sign(self, dim:int, generator,
                   dtype=torch.float32)->torch.Tensor:
    binary_rand = torch.randint(0,2,(dim,),
                                generator=generator,
                                dtype=dtype)
    xi = 2.0*binary_rand-1.0

    return xi