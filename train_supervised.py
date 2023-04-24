import argparse
import logging
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import openTSNE
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from tqdm import trange

from evaluation.evaluation import eval_node_classification
from model.tgn import TGN
from utils.data_processing import compute_time_statistics, \
  get_data_node_classification
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP, set_seed, \
  rgb_to_hex

DIM = 100
MEMORY_DIM = 172

### Argument and global variables
parser = argparse.ArgumentParser('MyTGN self-supervised training')
parser.add_argument('-d', '--data', type=str,
                    help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='',
                    help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10,
                    help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2,
                    help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1,
                    help='Number of network layers')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--patience', type=int, default=5,
                    help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1,
                    help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=MEMORY_DIM,
                    help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100,
                    help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1,
                    help='Every how many batches to '
                         'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention",
                    choices=[
                      "graph_attention", "graph_sum", "identity", "time"],
                    help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity",
                    choices=[
                      "mlp", "identity"], help='Type of message function')
parser.add_argument('--aggregator', type=str, default="last",
                    help='Type of message '
                         'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=DIM,
                    help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=MEMORY_DIM,
                    help='Dimensions of the memory for '
                         'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message',
                    action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')
parser.add_argument('--new_node', action='store_true', help='model new node')

parser.add_argument('--do_visual', action='store_true',
                    help='Whether to visualize node embeddings')

parser.add_argument('--seed', type=int, default=42)

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

set_seed(args.seed)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 1
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}' + '\
  node-classification.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}' + '\
  node-classification.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter(
  '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

full_data, node_features, edge_features, train_data, val_data, test_data = \
  get_data_node_classification(DATA, use_validation=args.use_validation)

max_idx = max(full_data.unique_nodes)

train_ngh_finder = get_neighbor_finder(train_data, uniform=UNIFORM,
                                       max_node_idx=max_idx)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
args.device = device

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations,
                          full_data.timestamps)

colormap = sns.color_palette('hls', n_colors=2)
colormap = [rgb_to_hex(color) for color in colormap]

for i in range(args.n_runs):
  results_path = "results/{}_node_classification_{}.pkl".format(args.prefix,
                                                                i) if i > 0 else "results/{}_node_classification.pkl".format(
    args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator, n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src,
            std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst,
            std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message)

  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.debug('Num of training instances: {}'.format(num_instance))
  logger.debug('Num of batches per epoch: {}'.format(num_batch))

  logger.info('Loading saved MyTGN model')
  model_path = f'./saved_models/{args.prefix}-{DATA}.pth'

  # Added for debug
  model_path = f'./saved_checkpoints/{args.prefix}-{DATA}-10.pth'

  tgn.load_state_dict(torch.load(model_path))
  tgn.eval()
  logger.info('MyTGN models loaded')
  logger.info('Start training node classification task')

  decoder = MLP(node_features.shape[1], drop=DROP_OUT)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
  decoder = decoder.to(device)
  decoder_loss_criterion = torch.nn.BCELoss()

  val_aucs = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)

  all_node_embeddings = None

  tsne_model = openTSNE.TSNE(
    initialization="pca",
    n_components=2,
    perplexity=30,
    metric="cosine",
    n_jobs=32,
    random_state=args.seed,
    verbose=True, n_iter=1000
  )

  for epoch in range(args.n_epoch):
    start_epoch = time.time()

    # Initialize memory of the model at each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    tgn = tgn.eval()
    decoder = decoder.train()
    loss = 0

    # The last updated time of each node in the memory
    last_update_timestamp = -torch.ones((full_data.n_unique_nodes + 1,),
                                        dtype=torch.long, device=device)

    node_states = torch.zeros((full_data.n_unique_nodes + 1,),
                              dtype=torch.long, device=device)

    latest_node_embeddings = torch.zeros(
      (full_data.n_unique_nodes + 1, NODE_DIM), dtype=torch.float,
      device=device)


    @torch.no_grad()
    def get_all_node_embeddings(step):
      assert not tgn.training
      decoder.eval()
      latest_timestamp_in_batch = max(
        tgn.memory.get_last_update(list(range(tgn.n_nodes))))
      latest_timestamp_in_batch = latest_timestamp_in_batch.item()

      nodes = np.arange(tgn.n_nodes)
      emb = tgn.embedding_module.compute_embedding(
        memory=tgn.memory.get_memory(
          list(range(tgn.n_nodes))),
        source_nodes=nodes,
        timestamps=np.full(nodes.shape[0], latest_timestamp_in_batch),
        n_layers=tgn.n_layers,
        n_neighbors=tgn.n_neighbors).detach().cpu().numpy()

      return step, latest_timestamp_in_batch, emb


    if args.do_visual:
      emb_li = [get_all_node_embeddings(0)]

    for k in trange(num_batch):
      s_idx = k * BATCH_SIZE
      e_idx = min(num_instance, s_idx + BATCH_SIZE)

      sources_batch = train_data.sources[s_idx: e_idx]
      destinations_batch = train_data.destinations[s_idx: e_idx]
      timestamps_batch = train_data.timestamps[s_idx: e_idx]
      edge_idxs_batch = full_data.edge_idxs[s_idx: e_idx]
      labels_batch = train_data.labels[s_idx: e_idx]

      size = len(sources_batch)

      decoder_optimizer.zero_grad()
      with torch.no_grad():
        source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(
          sources_batch,
          destinations_batch,
          destinations_batch,
          timestamps_batch,
          edge_idxs_batch,
          NUM_NEIGHBORS)

      labels_batch_torch = torch.from_numpy(labels_batch).float().to(
        device)

      pred = decoder(source_embedding).sigmoid()

      if args.do_visual:
        tgn.eval()
        if (k % 100 == 0) or (k == num_batch - 1):
          emb_li.append(get_all_node_embeddings(k + 1))

        # Set the state of the source nodes whose state has changed at current timestamp
        # We assume only source nodes can be infected
        mask = labels_batch == 1
        source_nodes_with_state_changes = sources_batch[mask]
        node_states[source_nodes_with_state_changes] = 1

        node_indices = np.concatenate(
          [sources_batch, destinations_batch])

        latest_node_embeddings[
          torch.tensor(node_indices, dtype=torch.long,
                       device=args.device)] = torch.concat(
          [source_embedding, destination_embedding],
          dim=0)

        last_update_timestamp[sources_batch] = torch.tensor(
          timestamps_batch,
          dtype=torch.long,
          device=device)
        last_update_timestamp[destinations_batch] = torch.tensor(
          timestamps_batch, dtype=torch.long, device=device)

      decoder_loss = decoder_loss_criterion(pred, labels_batch_torch)
      decoder_loss.backward()
      decoder_optimizer.step()
      loss += decoder_loss.item()

    train_losses.append(loss / num_batch)

    val_auc = eval_node_classification(tgn, args, decoder, val_data,
                                       full_data.edge_idxs, BATCH_SIZE,
                                       n_neighbors=NUM_NEIGHBORS,
                                       last_update_timestamp=last_update_timestamp,
                                       latest_node_embeddings=latest_node_embeddings,
                                       node_states=node_states,
                                       )
    val_aucs.append(val_auc)

    if args.do_visual and epoch > 5:
      emb = emb_li[-1][2]
      sum_of_rows = np.linalg.norm(emb, axis=1)
      normalized_source_embedding = emb / sum_of_rows[:,
                                          np.newaxis]

      embedding_train = tsne_model.fit(normalized_source_embedding)
      df_visual = pd.DataFrame(embedding_train, columns=['x', 'y'])

      """
      mask = (last_update_timestamp.cpu().numpy() >= 0)
      embedding_train = tsne_model.fit(
          all_node_embeddings[mask].cpu().numpy())

      df_visual = pd.DataFrame(embedding_train, columns=['x', 'y'])
      df_visual.index = np.where(mask)[0]
      
      """
      df_visual['label'] = 0
      nodes_wth_state_changes = np.where(node_states.cpu().numpy() == 1)[
        0].tolist()

      df_visual.loc[nodes_wth_state_changes, 'label'] = 1
      df_visual['index'] = df_visual.index
      df_visual['color'] = df_visual['label'].apply(
        lambda x: 'red' if x else 'blue')

      # df_visual['latest_update_time'] = last_update_timestamp.cpu().numpy()[mask]

      fig = px.scatter(df_visual, x="x", y="y", color="color",
                       hover_name="index", labels='label')

      fig.write_html(f"tgn_node_classification_ep{epoch}.html")

      # fig = px.scatter(df_visual, x="x", y="y", size="node_size",
      #                  color='node_color', text="display_name",
      #                  hover_name="name", title=snapshot_name,
      #                  log_x=False,
      #                  opacity=0.5)

    pickle.dump({
      "val_aps": val_aucs,
      "train_losses": train_losses,
      "epoch_times": [0.0],
      "new_nodes_val_aps": [],
    }, open(results_path, "wb"))

    logger.info(
      f'Epoch {epoch}: train loss: {loss / num_batch}, val auc: {val_auc}, time: {time.time() - start_epoch}')

  if args.use_validation:
    if early_stopper.early_stop_check(val_auc):
      logger.info('No improvement over {} epochs, stop training'.format(
        early_stopper.max_round))
      break
    else:
      torch.save(decoder.state_dict(), get_checkpoint_path(epoch))

  if args.use_validation:
    logger.info(
      f'Loading the best model at epoch {early_stopper.best_epoch}')
    best_model_path = get_checkpoint_path(early_stopper.best_epoch)
    decoder.load_state_dict(torch.load(best_model_path))
    logger.info(
      f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
    decoder.eval()

    test_auc = eval_node_classification(tgn, args, decoder, test_data,
                                        full_data.edge_idxs, BATCH_SIZE,
                                        n_neighbors=NUM_NEIGHBORS,
                                        last_update_timestamp=last_update_timestamp,
                                        latest_node_embeddings=latest_node_embeddings)
  else:
    # If we are not using a validation set, the test performance is just the performance computed
    # in the last epoch
    test_auc = val_aucs[-1]

  pickle.dump({
    "val_aps": val_aucs,
    "test_ap": test_auc,
    "train_losses": train_losses,
    "epoch_times": [0.0],
    "new_nodes_val_aps": [],
    "new_node_test_ap": 0,
  }, open(results_path, "wb"))

  logger.info(f'test auc: {test_auc}')
