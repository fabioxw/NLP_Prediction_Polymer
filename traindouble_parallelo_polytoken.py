import multiprocessing as mp
from tqdm import tqdm
import pickle
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,median_absolute_error
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from torch.cuda.amp import GradScaler, autocast
from datetime import datetime
import pandas as pd
from rdkit import Chem
import numpy as np  
from typing import Tuple
import os
import time
import math
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.nn import (
    BatchNorm, GCNConv, GraphConv, SGConv, LayerNorm, NNConv, GATConv,
    global_mean_pool, global_add_pool
)
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataListLoader as DLGeometric

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from rdkit import Chem

from transformers import (
    RobertaConfig, RobertaModel, RobertaTokenizer, RobertaTokenizerFast,
    RobertaForSequenceClassification, TrainingArguments, Trainer,
    EarlyStoppingCallback, IntervalStrategy
)

from torch_geometric.data import Batch
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.cuda.amp import GradScaler, autocast
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm
import os
import pickle
from rdkit import Chem
import psutil
import gc
from copy import deepcopy
import tokenizer_utils

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

free_memory()

def print_memory_usage(message):
    process = psutil.Process(os.getpid())
    print(f"{message} - Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

print_memory_usage("Before data loading")

def process_smile(smile, input_ids):
    mol = GraphMol(smile)
    graph = Data(x=mol.node_features, edge_index=mol.adjacency_info, edge_attr=mol.edge_features)
    graph.smile = smile
    graph.input_ids = input_ids
    return graph






print_memory_usage("After tokenizing train data")

def create_graph_data_batch(smiles_list, input_ids_list, batch_size):
    graph_data_list = []
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Creating graph data in batches"):
        batch_smiles = smiles_list[i:i + batch_size]
        batch_input_ids = input_ids_list[i:i + batch_size]
        batch_graph_data = create_graph_data_parallel(batch_smiles, batch_input_ids)
        graph_data_list.extend(batch_graph_data)
    return graph_data_list

'''
def create_graph_data_parallel(smiles_list, input_ids_list):
    pool_size = min(mp.cpu_count(), 2)
    with mp.Pool(pool_size) as pool:
        graph_data_list = list(tqdm(pool.starmap(process_smile, zip(smiles_list, input_ids_list)), total=len(smiles_list)))
    print(f"Length of graph_data_list: {len(graph_data_list)}")
    return graph_data_list
    
'''
def create_graph_data_parallel(smiles_list, input_ids_list):
    pool_size = min(mp.cpu_count(), 2)
    with mp.Pool(pool_size) as pool:
        # Uso di tqdm all'interno della creazione parallela
        graph_data_list = []
        for graph_data in tqdm(pool.starmap(process_smile, zip(smiles_list, input_ids_list)), total=len(smiles_list), desc="Creating Graph Data"):
            graph_data_list.append(graph_data)
    print(f"Length of graph_data_list: {len(graph_data_list)}")
    return graph_data_list




def cache_graph_data(file_path, func, *args):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        data = func(*args)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return data

def collate_fn(batch):
    smiles = [item['smiles'] for item in batch]
    labels = torch.stack([item['labels'] for item in batch])
    graph_data = Batch.from_data_list([process_smile(smile, item['input_ids']) for smile, item in zip(smiles, batch)])
    token_data = {key: torch.stack([item[key] for item in batch]) for key in batch[0] if key not in ['smiles', 'labels', 'graph']}
    return {**token_data, 'smiles': smiles, 'labels': labels, 'graph': graph_data}



class NewFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, smiles, graph_data):
        self.encodings = encodings
        self.labels = labels
        self.smiles = smiles
        self.graph_data = graph_data
        
        # Verifica della lunghezza dei dati
        assert len(self.encodings['input_ids']) == len(self.labels), f"Le lunghezze degli encodings e delle labels non corrispondono: {len(self.encodings['input_ids'])} vs {len(self.labels)}"
        assert len(self.labels) == len(self.smiles), f"Le lunghezze delle labels e delle smiles non corrispondono: {len(self.labels)} vs {len(self.smiles)}"
        assert len(self.smiles) == len(self.graph_data), f"Le lunghezze delle smiles e dei dati del grafo non corrispondono: {len(self.smiles)} vs {len(self.graph_data)}"
        
        # Debug
        print(f"Length of encodings: {len(next(iter(self.encodings.values())))}")
        print(f"Length of labels: {len(self.labels)}")
        print(f"Length of smiles: {len(self.smiles)}")
        print(f"Length of graph_data: {len(self.graph_data)}")

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        item['smiles'] = self.smiles[idx]
        item['graph'] = self.graph_data[idx]
        return item

    def __len__(self):
        return len(self.labels)

class GraphMol():
    def __init__(self, smile):
        self.mol = Chem.MolFromSmiles(smile)
        self.node_features = self._get_node_features()
        self.edge_features = self._get_edge_features()
        self.adjacency_info = self._get_adjacency_info()

    def _get_node_features(self):
        all_node_feats = []
        for atom in self.mol.GetAtoms():
            node_feats = []
            node_feats.append(atom.GetAtomicNum())
            node_feats.append(atom.GetDegree())
            node_feats.append(atom.GetFormalCharge())
            node_feats.append(atom.GetHybridization())
            node_feats.append(atom.GetIsAromatic())
            node_feats.append(atom.GetTotalNumHs())
            node_feats.append(atom.GetNumRadicalElectrons())
            node_feats.append(atom.IsInRing())
            node_feats.append(atom.GetChiralTag())
            all_node_feats.append(node_feats)
        all_node_feats = np.asarray(all_node_feats)
        all_node_feats_star = []
        for node in all_node_feats:
            if node[0] == 0:
                all_node_feats_star.append([0, -1, -1, -1, -1, -1, -1, -1, -1])
            else:
                all_node_feats_star.append(node)
        all_node_feats_star = np.array(all_node_feats_star)
        return torch.tensor(all_node_feats_star, dtype=torch.float)

    def _get_edge_features(self):
        all_edge_feats = []
        for bond in self.mol.GetBonds():
            edge_feats = []
            edge_feats.append(bond.GetBondTypeAsDouble())
            edge_feats.append(bond.IsInRing())
            all_edge_feats += [edge_feats, edge_feats]
        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(self):
        edge_indices = []
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

class GCNEmbeddingModel(nn.Module):
    def __init__(self, feature_node_dim=9):
        super(GCNEmbeddingModel, self).__init__()
        self.conv1 = GraphConv(feature_node_dim, 32)
        self.conv2 = GraphConv(32, 64)
        self.conv3 = GraphConv(64, 128)
        self.conv4 = GraphConv(128, 256)
        self.ln1 = LayerNorm([32])
        self.ln2 = LayerNorm([64])
        self.ln3 = LayerNorm([128])
        self.ln4 = LayerNorm([256])

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = self.ln4(x)
        x = F.relu(x)
        x_embed = global_mean_pool(x, batch)
        return x_embed

class NewPredModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.chemBERTA = RobertaForSequenceClassification.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250", num_labels=1).to(device)
        self.wdMPNN = GCNEmbeddingModel().to(device)

    def forward(self, input_ids, attention_masks, graph_data):
        graph_data = graph_data.to(self.device)
        graph_vector = self.wdMPNN(graph_data.x, graph_data.edge_index, graph_data.batch)
        print(f"input_ids shape: {input_ids.shape}")
        print(f"graph_vector shape: {graph_vector.shape}")
        if graph_vector.size(0) != input_ids.size(0):
            graph_vector = graph_vector.view(input_ids.size(0), -1)
        print(f"graph_vector shape after adjustment: {graph_vector.shape}")
        cat_tensor = torch.cat((input_ids, graph_vector), dim=1).long()
        print(f"cat_tensor shape: {cat_tensor.shape}")
        attention_masks = torch.cat((attention_masks.to(self.device), torch.ones(attention_masks.size(0), graph_vector.size(1)).to(self.device)), dim=1)
        attention_masks = attention_masks.long()
        out = self.chemBERTA(input_ids=cat_tensor, attention_mask=attention_masks)
        logits = out.logits
        return logits

class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y))
        return loss
    
class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Calcola la differenza tra le previsioni e i valori reali
        diff = y_pred - y_true
        # Eleva al quadrato la differenza
        squared_diff = diff ** 2
        # Calcola la media dei quadrati delle differenze
        loss = torch.mean(squared_diff)
        return loss

    

def train_one_epoch(device: torch.device, model: NewPredModel, loss_fn: MSELoss, optimizer: torch.optim.Optimizer, scaler, epoch_index: int, batch_size: int, token_loader: DataLoader) -> float:
    running_loss = 0
    last_loss = 0.
    all_labels = []
    all_outputs = []
    model.train()
    start_time = time.time()
    for (i, token_data) in enumerate(tqdm(token_loader, desc=f"Training Epoch {epoch_index+1}")):
        print(token_data.keys())
        tokens = token_data['input_ids'].to(device)
        attention_masks = token_data['attention_mask'].to(device)
        labels = token_data['labels'].to(device)
        graph_data = token_data['graph'].to(device)
        labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids=tokens, attention_masks=attention_masks, graph_data=graph_data)
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        all_labels.extend(labels.detach().cpu().numpy())
        all_outputs.extend(outputs.detach().cpu().numpy())
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch_index}, Batch {i}, Loss: {loss.item()}")
    last_loss = running_loss / len(token_loader)
    print(f'Epoch {epoch_index} completed. Average Loss: {last_loss:.4f}')
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    train_rmse = np.sqrt(mean_squared_error(all_labels, all_outputs))
    train_mae = mean_absolute_error(all_labels, all_outputs)
    train_medae = median_absolute_error(all_labels, all_outputs)
    train_r2 = r2_score(all_labels, all_outputs)
    print(f'Training Metrics - Epoch: {epoch_index} RMSE: {train_rmse:.5f} MAE: {train_mae:.5f} MedAE: {train_medae:.5f} R2: {train_r2:.5f}')
    return last_loss, train_rmse, train_mae, train_medae, train_r2

def train_epochs(train_data_dict: dict, val_data_dict: dict, num_epochs: int, train_dict: dict, device: torch.device, patience_threshold: int) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = float('inf')
    best_model_path = None  # Variabile per memorizzare il percorso del miglior modello
    stop_training = False
    metrics_list = []
    scaler = GradScaler()

    for epoch in range(num_epochs):
        if stop_training:
            print(f"Early stopping bloccato in epoca: {epoch + 1}")
            break
        
        start_time = time.time()
        model = train_dict["model"].train(True)
        avg_loss, train_rmse, train_mae, train_medae, train_r2 = train_one_epoch(device, model, train_dict["loss_fn"], train_dict["optimizer"], scaler, epoch, train_dict["batch_size"], train_data_dict)
        
        curr_vloss = 0.0
        all_vlabels = []
        all_voutputs = []
        model.eval()
        
        with torch.no_grad():
            for (_, vtoken_data) in enumerate(val_data_dict):
                tokens = vtoken_data['input_ids'].to(device)
                attention_masks = vtoken_data['attention_mask'].to(device)
                vlabels = vtoken_data['labels'].to(device)
                graph_data = vtoken_data['graph'].to(device)
                with autocast():
                    voutputs = model(input_ids=tokens, graph_data=graph_data, attention_masks=attention_masks)
                    loss = train_dict["loss_fn"]
                    vloss = loss(voutputs, vlabels.unsqueeze(1))
                    curr_vloss += vloss.item()
                all_vlabels.extend(vlabels.detach().cpu().numpy())
                all_voutputs.extend(voutputs.detach().cpu().numpy())
        
        end_time = time.time()
        epoch_time = end_time - start_time
        #avg_vloss = curr_vloss / (epoch + 1)
        avg_vloss = curr_vloss / len(val_data_dict)
        
        all_vlabels = np.array(all_vlabels)
        all_voutputs = np.array(all_voutputs)
        val_rmse = np.sqrt(mean_squared_error(all_vlabels, all_voutputs))
        val_mae = mean_absolute_error(all_vlabels, all_voutputs)
        val_medae = median_absolute_error(all_vlabels, all_voutputs)
        val_r2 = r2_score(all_vlabels, all_voutputs)
        
        print(f'Epoch: {epoch} Train Loss: {avg_loss:.5f} Validation Loss: {avg_vloss:.5f}')
        print(f'Validation Metrics - Epoch: {epoch} RMSE: {val_rmse:.5f} MAE: {val_mae:.5f} MedAE: {val_medae:.5f} R2: {val_r2:.5f}')
        
        metrics_list.append({
            'Epoch': epoch,
            'Train Loss': avg_loss,
            'Train RMSE': train_rmse,
            'Train MAE': train_mae,
            'Train MedAE': train_medae,
            'Train R2': train_r2,
            'Validation Loss': avg_vloss,
            'Validation RMSE': float(val_rmse),
            'Validation MAE': float(val_mae),
            'Validation MedAE': float(val_medae),
            'Validation R2': float(val_r2),
            'Epoch Time (s)': epoch_time
        })
        
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv('/home/fvilla/prova/Encoder/FILE_DOUBLEBERTa/FILEDOUBLE_POLYTOKEN.csv', index=False)
        
        if avg_vloss < best_vloss:
            patience_counter = 0
            best_vloss = avg_vloss
            best_model_path = '/home/fvilla/prova/Encoder/checkpoint_PARALLELODOUBLE/checkpointPOLYTOKEN_{}.pth'.format(timestamp)
            torch.save(model.state_dict(), best_model_path)
            print(f'Checkpoint salvato per epoca {epoch+1} a {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience_threshold:
                stop_training = True
    
    print('Risultati salvati in FILEDOUBLE_POLYTOKEN.csv')

    # Carica il miglior modello al termine dell'addestramento
    if best_model_path is not None:
        train_dict["model"].load_state_dict(torch.load(best_model_path))
        print(f'Miglior modello caricato da {best_model_path}')
    return best_model_path  

def early_stopping(current_loss, validation_loss, patience_counter, patience_threshold, stop_training=False):
    if validation_loss < current_loss:
        current_loss = validation_loss
        stop_training = False
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience_threshold:
            stop_training = True
    return current_loss, stop_training, patience_counter

#-------------------------

def load_best_model(model, best_model_path):
    print(f"Caricamento dei pesi dal file: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    return model

def evaluate_model(device: torch.device, model: NewPredModel, loss_fn: MSELoss, test_loader: DataLoader, best_model_path: str) -> dict:
    print(f"Valutazione del modello con i pesi caricati da: {best_model_path}")
    model.eval()
    all_labels = []
    all_outputs = []
    all_smiles = []
    total_loss = 0.0

    with torch.no_grad():
        for token_data in tqdm(test_loader, desc="Evaluating"):
            tokens = token_data['input_ids'].to(device)
            attention_masks = token_data['attention_mask'].to(device)
            labels = token_data['labels'].to(device)
            graph_data = token_data['graph'].to(device)
            smiles = token_data['smiles']
            
            labels = labels.unsqueeze(1)
            with autocast():
                outputs = model(input_ids=tokens, attention_masks=attention_masks, graph_data=graph_data)
                loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            all_labels.extend(labels.detach().cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_smiles.extend(smiles)

    avg_loss = total_loss / len(test_loader)
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    rmse = np.sqrt(mean_squared_error(all_labels, all_outputs))
    mae = mean_absolute_error(all_labels, all_outputs)
    medae = median_absolute_error(all_labels, all_outputs)
    r2 = r2_score(all_labels, all_outputs)

    print(f'Test Loss: {avg_loss:.5f}')
    print(f'Test RMSE: {rmse:.5f}')
    print(f'Test MAE: {mae:.5f}')
    print(f'Test MedAE: {medae:.5f}')
    print(f'Test R2: {r2:.5f}')
     
    # Salva le SMILES e gli output in un file CSV
    results_df = pd.DataFrame({'smiles': all_smiles, 'predicted_value': all_outputs.flatten()})
    results_df.to_csv('/home/fvilla/prova/Encoder/FILE_DOUBLEBERTa/risultato_ptoken.csv', index=False)
    print('Risultati salvati in risultato_ptoken.csv')
     
    return {
        'Test Loss': avg_loss,
        'Test RMSE': rmse,
        'Test MAE': mae,
        'Test MedAE': medae,
        'Test R2': r2
    }


#--------------------------------
#MAIN
#---------------

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#dataset = pd.read_csv("/home/fvilla/prova/polyset/SMILES_Ei.csv", delimiter=',', encoding="utf-8-sig")
dataset = pd.read_csv("/home/fvilla/prova/DATI/smiles_with_values.csv", delimiter=',', encoding="utf-8-sig")
#dataset = pd.read_csv("/home/fvilla/prova/polyset/SMILES_Ei.csv", delimiter=',', encoding="utf-8-sig")
#dataset = pd.read_csv("/home/fvilla/prova/polyset/smiles100k_less_than_61.csv", delimiter=',', encoding="utf-8-sig")
dataset = dataset[['smiles','value']]
print("sono 1")
train, test = train_test_split(dataset, train_size=0.8)
print("sono 2")
print("sono 3")
#tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer= tokenizer_utils.PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=256)
max_length = 256
print("sono 4")
batch_size1 = 350
encodings_list = []
print_memory_usage("Before data loading")
print("sono 5")
for i in tqdm(range(0, len(train), batch_size1), desc="Tokenizzazione Train"):
    batch_smiles = train['smiles'].iloc[i:i + batch_size1].tolist()
    batch_encodings = tokenizer(batch_smiles, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    encodings_list.append(batch_encodings)
print_memory_usage("After tokenizing train data")
print("sono 7")
train_encodings = {key: torch.cat([batch[key] for batch in encodings_list], dim=0) for key in encodings_list[0]}
print("sono 7.1")
with open('train_encodings_PARALLELO_PTOKEN.pkl', 'wb') as f:
    pickle.dump(train_encodings, f)
with open('train_encodings_PARALLELO_PTOKEN.pkl', 'rb') as f:
    train_encodings = pickle.load(f)
    
print_memory_usage("After loading train encodings")
print("sono tokenizzazione completata e dati salvati pER IL TRAIN")
val_encodings_list = []
print("sono train1")
for i in tqdm(range(0, len(test), batch_size1), desc="Tokenizzazione Test"):
    batch_smiles = test['smiles'].iloc[i:i + batch_size1].tolist()
    batch_encodings = tokenizer(batch_smiles, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    val_encodings_list.append(batch_encodings)
print("sono train 2")
val_encodings = {key: torch.cat([batch[key] for batch in val_encodings_list], dim=0) for key in val_encodings_list[0]}
print("sono train3")
with open('test_encodings_PARALLELO_PTOKEN.pkl', 'wb') as f:
    pickle.dump(train_encodings, f)
with open('test_encodings_PARALLELO_PTOKEN.pkl', 'rb') as f:
    train_encodings = pickle.load(f)
print_memory_usage("After loading test encodings")
print("sono tokenizzazione completata e dati salvati pER IL TEST")
train_labels = train['value'].values
train_smiles = train['smiles'].values
test_labels = test['value'].values
test_smiles = test['smiles'].values
print(f"Length of train_encodings: {len(train_encodings['input_ids'])}")
print(f"Length of train_labels: {len(train_labels)}")
print(f"Length of train_smiles: {len(train_smiles)}")
print("sono qua")
train_graph_data = cache_graph_data('train_graph_PARALLELO_data_PTOKEN.pkl', create_graph_data_batch, train_smiles.tolist(), train_encodings['input_ids'].tolist(),500)
val_graph_data = cache_graph_data('val_graph_PARALLELO_data_PTOKEN.pkl', create_graph_data_batch, test_smiles.tolist(), val_encodings['input_ids'].tolist(),500)
print_memory_usage("After creating graph data")
print("sono qui")
print(f"Length of train_graph_data: {len(train_graph_data)}")
train_t = NewFinetuneDataset(train_encodings, train_labels,train_smiles,train_graph_data)
train_loader = DataLoader(train_t, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
print("sono finetunetrain")
test_encodings = tokenizer(test['smiles'].tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
print("sono finetunetest")
test_t = NewFinetuneDataset(test_encodings, test_labels, test_smiles,val_graph_data)
print("sono prima del dataloader")
test_loader = DataLoader(test_t, batch_size=16, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
print("DataLoader creati.")
print_memory_usage("After creating DataLoaders")
print("sono 5")
torch.save(train_graph_data, 'train_graph_PARALLELO_data_PTOKEN.pt')
torch.save(val_graph_data, 'val_graph_PARALLELO_data_PTOKEN.pt')
print("sono 6")
train_graph_data = torch.load('train_graph_PARALLELO_data_PTOKEN.pt')
val_graph_data = torch.load('val_graph_PARALLELO_data_PTOKEN.pt')
print("sono 8")
train_data_dict = train_loader
test_data_dict = test_loader
train_dict = {}
print("sono 12")
embedding_evaluator = NewPredModel(device)
optimizer = torch.optim.Adam(embedding_evaluator.parameters(), lr=1e-4)
print("sono 13")
scaler = GradScaler()
train_dict["model"] = embedding_evaluator
train_dict["loss_fn"] = MSELoss()
train_dict["optimizer"] = optimizer
train_dict["batch_size"] = 16




best_model_path=train_epochs(train_data_dict, test_data_dict,500, train_dict, device, patience_threshold=30)


# Carica i pesi migliori
model = load_best_model(embedding_evaluator, best_model_path)

# Valuta il modello sui dati di test
test_results = evaluate_model(device, model, MSELoss(), test_loader,best_model_path)

# Salva i risultati delle metriche di valutazione in un file CSV
metrics_df = pd.DataFrame([test_results])
metrics_df.to_csv('/home/fvilla/prova/Encoder/FILE_DOUBLEBERTa/best_value_chembPARALLELO_PTOKEN.csv', index=False)