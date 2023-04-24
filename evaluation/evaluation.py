import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import trange


def eval_edge_prediction(model, args, negative_edge_sampler, data, n_neighbors,
                         batch_size=200, last_update_timestamp=None,
                         latest_node_embeddings=None, node_states=None):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []

  # if latest_node_embeddings is None or last_update_timestamp is None:
  #   FLAG_VISUAL = False
  # else:
  #   FLAG_VISUAL = True

  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in trange(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob, embeds = model.compute_edge_probabilities(
        sources_batch, destinations_batch,
        negative_samples, timestamps_batch,
        edge_idxs_batch, n_neighbors, return_embeds=args.do_visual)

      if args.do_visual:
        (src_emb, dst_emb, neg_dst_emb) = embeds

        mask = data.labels[s_idx:e_idx] == 1

        source_nodes_with_state_changes = sources_batch[mask]

        node_states[source_nodes_with_state_changes] = 1

        node_indices = np.concatenate(
          [sources_batch, destinations_batch, negative_samples])
        latest_node_embeddings[
          torch.tensor(node_indices, dtype=torch.long,
                       device=args.device)] = torch.concat(
          [src_emb.detach(), dst_emb.detach(), neg_dst_emb.detach()],
          dim=0)

        last_update_timestamp[sources_batch] = torch.tensor(
          timestamps_batch,
          dtype=torch.long,
          device=args.device)
        last_update_timestamp[destinations_batch] = torch.tensor(
          timestamps_batch, dtype=torch.long, device=args.device)

      pred_score = np.concatenate(
        [(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, args, decoder, data, edge_idxs, batch_size,
                             n_neighbors, last_update_timestamp=None,
                             latest_node_embeddings=None, node_states=None):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in trange(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
        sources_batch,
        destinations_batch,
        destinations_batch,
        timestamps_batch,
        edge_idxs_batch,
        n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

      if args.do_visual:
        mask = data.labels[s_idx:e_idx] == 1

        source_nodes_with_state_changes = sources_batch[mask]

        node_states[source_nodes_with_state_changes] = 1

        node_indices = np.concatenate(
          [sources_batch, destinations_batch])
        latest_node_embeddings[
          torch.tensor(node_indices, dtype=torch.long,
                       device=args.device)] = torch.concat(
          [source_embedding.detach(), destination_embedding.detach()],
          dim=0)

        last_update_timestamp[sources_batch] = torch.tensor(
          timestamps_batch,
          dtype=torch.long,
          device=args.device)
        last_update_timestamp[destinations_batch] = torch.tensor(
          timestamps_batch, dtype=torch.long, device=args.device)

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
