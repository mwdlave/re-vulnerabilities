"""
This module contains the function to project the adversarial embeddings
to the closest token in the embedding space of chosen vocabulary.
"""
import torch
import logging
from src.adv_sample.vocab import FlexibleVocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def project_embeddings(sample_embeddings: torch.tensor,
                       sample_tokens: torch.tensor, 
                       vocab: FlexibleVocab, 
                       mask: torch.tensor,
                       device: str = 'cuda'
                       #compare_function: str
                      ) -> torch.tensor:
    """
    Given a batch of sample adversarial embeddings, project
    them into their closest token in the embedding space.

    Input:
    - `sample_embeddings`: Tensor of shape (batch_size, seq_length, embedding_dim) containing the embeddings.
    - `sample_tokens`: Tensor of shape (batch_size, seq_length) containing the tokens.
    - `vocab`: The vocabulary to use for the adversarial samples.
    - `mask`: Tensor of shape (batch_size, seq_length) containing the mask, it can have values from 0 to n depending on the number of tokens specific phrase will be switched to.
    - `compare_function`: The function to use to compare the embeddings.
    
    Output:
    - `batch_emb`: Tensor of shape (batch_size, seq_length, embedding_dim) containing the projected embeddings.
    - `batch_tokens`: Tensor of shape (batch_size, seq_length) containing the projected tokens.

    """

    batch_emb = sample_embeddings.clone()
    batch_tokens = sample_tokens.clone()

    logger.debug(f"project_embeddings: batch_emb.shape: {batch_emb.shape}")

    # Iterate over the unique token length of words/phrases we want to switch in the mask
    # excluding 0 - tokens that are not to be switched
    for i in torch.unique(mask).tolist():
        if i==0:
            continue
        logger.debug(f"project_embeddings: i: {i}")
        # Get the indices where mask == i
        indices = mask == i

        # Expand indices to match the embedding dimensions
        expanded_indices = indices.unsqueeze(-1).expand_as(batch_emb)

        logger.debug(f"project_embeddings: expanded_indices.shape: {expanded_indices.shape}")
        # Select embeddings where mask == i
        selected_embeddings = batch_emb[expanded_indices].view(-1, i, batch_emb.size(-1))

        results, token_list = vocab.compare_strict_batch(input_embeddings=selected_embeddings)

        idx_closest = results.argmax(dim=-1)

        token_closest = torch.tensor([token_list[idx] for idx in idx_closest.tolist()]).view(-1)
                                
        batch_tokens[indices] = token_closest.to(device)
        
        batch_emb = vocab.embedding_matrix[batch_tokens]

    logger.debug(f"project_embeddings: batch_emb.shape: {batch_emb.shape}")
    logger.debug(f"project_embeddings: batch_tokens.shape: {batch_tokens.shape}")

    return batch_emb, batch_tokens    

