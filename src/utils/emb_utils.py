import torch
import pandas as pd
import numpy as np
from src.utils.logits_utils import compute_logit_diff_2

def group_emb(gradients):
    
    # Step 1: Flatten all gradients to compute global mean and std
    all_gradients = np.concatenate(
        [np.array(grad.cpu()).reshape(-1, grad.shape[-1]) for grad in gradients], axis=0
    )  # Shape: [total_elements, embedding_dim]

    # Compute global mean and std
    global_mean = np.mean(all_gradients, axis=0, keepdims=True)  # Shape: [1, embedding_dim]
    global_std = np.std(all_gradients, axis=0, keepdims=True) + 1e-8  # Shape: [1, embedding_dim]

    # Step 2: Normalize gradients globally
    mean_grad = []

    for it in range(len(gradients)):
        grad = np.array(gradients[it].cpu())  # Convert to numpy for easier manipulation

        # Normalize gradients globally
        grad_norm = (grad - global_mean) / global_std  # Shape: [batch_size, token_length, embedding_dim]

        # Mean over embeddings for each token across the batch
        grad_mean_per_token = np.mean(abs(grad_norm), axis=2)  # Shape: [batch_size,token_length]

        mean_grad.append(grad_mean_per_token)
        
    return mean_grad    
  
def AdvMarginLoss(margin=1.0):
    """
    Create the adversarial Margin Loss
    """
    def loss_fn(logits_all, tokens_all, y, average=True, needed_tokens = [0, 1]):
        """
        Return the adversarial margin loss used to generate adversarial samples.

        Parameters:
        - `logits_all`: Tensor of shape (batch_size, seq_length, num_classes) containing the logits.
        - `y`: Tensor of shape (batch_size,) containing the index of the ground truth.
        """
        # gather the logits of the ground truth
        loss = compute_logit_diff_2(logits_all = logits_all,
                            tokens_all = tokens_all, 
                            correct_answers = y, 
                            needed_tokens = needed_tokens,
                            average=False)
        
        loss = loss + margin 
        loss = torch.where(loss < 0, torch.zeros_like(loss), loss)

        return loss.mean() if average else loss

    return loss_fn



def create_mask(model, sample_tokens, names):
    mask = torch.zeros_like(sample_tokens)

    for j, name in enumerate(names):
      # Find the sequence in the larger tensor
      sequence = model.to_tokens(name, prepend_bos=False)[0]
      large_tensor = sample_tokens[j]
      seq_len = sequence.size(0)
      for i in range(len(large_tensor)- seq_len + 1):
          
          if torch.equal(large_tensor[i:i + seq_len], sequence):
              mask[j,i:i + seq_len] = seq_len  # Update mask with value of seq_len
              break


    #check the correctness of the mask
    for i,mas in enumerate(mask):
        if (mas != 0).any():
           continue
        else:
            print(f"Error: Mask at index {i} is entirely zero.")
            
    return mask

def group_grad(grads, masks):
    """
    Group gradients based on the mask

    Parameters:
    - `grads`: List of gradients for each token
    - `masks`: List of masks for each token
   
    Returns:
    - `full_masked_mean`: Mean of the masked gradients across iterations
    - `full_unmasked_mean`: Mean of the unmasked gradients across iterations
    - `full_masked_max`: Max of the masked gradients across iterations
    - `full_unmasked_max`: Max of the unmasked gradients across iterations
    
    """
    full_masked_mean = []
    full_unmasked_mean = []
    full_masked_max = []
    full_unmasked_max = []

    for mask, grad in zip(masks, grads):

        mask = mask==0
        
        iter_masked_mean = []
        iter_unmasked_mean = []
        iter_masked_max = []
        iter_unmasked_max = []

        for iter_grad in grad:

            # Separate masked and unmasked gradients
            masked_grad = iter_grad[mask]
            unmasked_grad = iter_grad[~mask]

            # Compute means for masked and unmasked
            masked_mean = np.mean(masked_grad) if masked_grad.size > 0 else 0
            unmasked_mean = np.mean(unmasked_grad) if unmasked_grad.size > 0 else 0

            # Compute means for masked and unmasked
            masked_max = np.max(abs(masked_grad)) if masked_grad.size > 0 else 0
            unmasked_max = np.max(abs(unmasked_grad)) if unmasked_grad.size > 0 else 0

            iter_masked_mean.append(masked_mean)
            iter_unmasked_mean.append(unmasked_mean)

            iter_masked_max.append(masked_max)
            iter_unmasked_max.append(unmasked_max)

        full_masked_mean.append(iter_masked_mean)
        full_unmasked_mean.append(iter_unmasked_mean)

        full_masked_max.append(iter_masked_max)
        full_unmasked_max.append(iter_unmasked_max)

    
    return np.array(full_masked_mean), np.array(full_unmasked_mean), np.array(full_masked_max), np.array(full_unmasked_max)

def process_tokens_and_gradients(tokens, gradients, start, end, start_middle):
    """
    Process tokens and gradients to extract tokens and gradients within the boundary
    and calculate the sum of the gradients for tokens outside the boundary
    - used for plot making

    Parameters:
    - `tokens`: List of tokens
    - `gradients`: List of gradients
    - `start`: Start index of the boundary
    - `end`: End index of the boundary
    - `start_middle`: Start index of the middle boundary

    Returns:
    - `tokens_in_boundary`: Tokens within the boundary
    - `gradients_in_boundary`: Gradients within the boundary
    """

    # Summing scores for tokens outside the boundaries
    start_score = np.sum(gradients[:start]) / start
    middle_score = np.sum(gradients[start_middle:end+1]) / (end+1-start_middle)
    end_score = np.sum(gradients[end+1:]) / (len(gradients) - end - 1)
    
    # Extract tokens and gradients within the boundary
    tokens_in_boundary = tokens[start:start_middle-1]
    gradients_in_boundary = gradients[start:start_middle-1]
    # Add special tokens for the start and end sums
    tokens_in_boundary = ["INSTRUCTION\nTOKENS"] + tokens_in_boundary + ["SENTENCE\nTOKENS"] + ["PADDING\nTOKENS"]
    gradients_in_boundary = [start_score] + list(gradients_in_boundary) + [middle_score] + [end_score]
    
    return tokens_in_boundary, gradients_in_boundary