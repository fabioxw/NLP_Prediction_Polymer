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
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
from torch_geometric.data import Batch
#from torch.utils.tensorboard import SummaryWriter
# Inizializza TensorBoard
#writer = SummaryWriter()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import psutil
import gc
import pickle
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, global_mean_pool
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import gc
import torch
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.nn import LayerNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score

import os
import torch
import gc
import psutil
import pickle
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from copy import deepcopy
from torch.nn import LayerNorm
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import RobertaModel
import time

# Funzione per liberare la memoria
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

free_memory()

def print_memory_usage(message):
    process = psutil.Process(os.getpid())
    print(f"{message} - Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")

print_memory_usage("Before data loading")

def process_smile(args):
    smile, input_ids = args
    mol = GraphMol(smile)
    graph = Data(x=mol.node_features, edge_index=mol.adjacency_info, edge_attr=mol.edge_features)
    graph.smile = smile
    graph.input_ids = input_ids
    return graph

print_memory_usage("After defining process_smile")

def create_graph_data(smiles_list, input_ids_list):
    graph_data_list = []
    valid_smiles = []
    valid_input_ids = []

    for smile, input_ids in zip(smiles_list, input_ids_list):
        try:
            graph = process_smile((smile, input_ids))
            graph_data_list.append(graph)
            valid_smiles.append(smile)
            valid_input_ids.append(input_ids)
        except Exception as e:
            print(f"Errore durante la creazione del grafo per {smile}: {e}")
            continue
    
    return graph_data_list, valid_smiles, valid_input_ids

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
    graph_data = Batch.from_data_list([item['graph'] for item in batch])
    token_data = {key: torch.stack([item[key] for item in batch]) for key in batch[0] if key not in ['smiles', 'labels', 'graph']}
    return {**token_data, 'smiles': smiles, 'labels': labels, 'graph': graph_data}

class NewFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, smiles, graph_data):
        self.encodings = encodings
        self.labels = labels
        self.smiles = smiles
        self.graph_data = graph_data

        assert len(self.encodings['input_ids']) == len(self.labels) == len(self.smiles) == len(self.graph_data), \
            f"Lengths of encodings ({len(self.encodings['input_ids'])}), labels ({len(self.labels)}), smiles ({len(self.smiles)}), and graph_data ({len(self.graph_data)}) must be the same"

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
            node_feats = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetHybridization(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs(),
                atom.GetNumRadicalElectrons(),
                atom.IsInRing(),
                atom.GetChiralTag()
            ]
            all_node_feats.append(node_feats)
    
        all_node_feats = np.asarray(all_node_feats)
        all_node_feats_star = np.array([[0, -1, -1, -1, -1, -1, -1, -1, -1] if node[0] == 0 else node for node in all_node_feats])
        return torch.tensor(all_node_feats_star, dtype=torch.float)

    def _get_edge_features(self):
        all_edge_feats = []
        for bond in self.mol.GetBonds():
            edge_feats = [
                bond.GetBondTypeAsDouble(),
                bond.IsInRing()
            ]
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
        return edge_indices.t().to(torch.long).view(2, -1)
class GCNEmbeddingModel(nn.Module):
    def __init__(self, feature_node_dim=9):
        super(GCNEmbeddingModel, self).__init__()
        self.conv1 = GraphConv(feature_node_dim, 32)
        self.conv2 = GraphConv(32, 64)
        self.ln1 = LayerNorm([32])
        self.ln2 = LayerNorm([64])

        
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
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
        if graph_vector.size(0) != input_ids.size(0):
            graph_vector = graph_vector.view(input_ids.size(0), -1)
        cat_tensor = torch.cat((input_ids, graph_vector), dim=1).long()
        attention_masks = torch.cat((attention_masks.to(self.device), torch.ones(attention_masks.size(0), graph_vector.size(1)).to(self.device)), dim=1)
        attention_masks = attention_masks.long()
        out = self.chemBERTA(input_ids=cat_tensor, attention_mask=attention_masks)
        logits = out.logits
        return logits



class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        squared_diff = diff ** 2
        loss = torch.mean(squared_diff)
        return loss

def train_one_epoch(device: torch.device, model: NewPredModel, loss_fn: MSELoss, optimizer: torch.optim.Optimizer, scaler, epoch_index: int, batch_size: int, token_loader: DataLoader) -> float:
    running_loss = 0
    last_loss = 0.0
    all_labels = []
    all_outputs = []
    model.train()
    start_time = time.time()
    for (i, token_data) in enumerate(tqdm(token_loader, desc=f"Training Epoch {epoch_index+1}")):
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

def train_epochs(train_data_dict: dict, val_data_dict: dict, num_epochs: int, train_dict: dict, device: torch.device, patience_threshold: int, save_dir: str, dataset_name: str) -> None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = float('inf')
    best_model_path = None
    stop_training = False
    metrics_list = []
    scaler = GradScaler()
    optimizer = train_dict["optimizer"]
    

    for epoch in range(num_epochs):
        if stop_training:
            print(f"Early stopping bloccato in epoca: {epoch + 1}")
            break
        
        start_time = time.time()
        model = train_dict["model"].train(True)
        avg_loss, train_rmse, train_mae, train_medae, train_r2 = train_one_epoch(device, model, train_dict["loss_fn"], optimizer, scaler, epoch, train_dict["batch_size"], train_data_dict)
        
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
                    voutputs = model(input_ids=tokens, attention_masks=attention_masks, graph_data=graph_data)
                    loss = train_dict["loss_fn"]
                    vloss = loss(voutputs, vlabels.unsqueeze(1))
                    curr_vloss += vloss.item()
                all_vlabels.extend(vlabels.detach().cpu().numpy())
                all_voutputs.extend(voutputs.detach().cpu().numpy())
        
        end_time = time.time()
        epoch_time = end_time - start_time
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
        
        # Salva le metriche in FILEDOUBLE_{dataset_name}.csv
        metrics_df = pd.DataFrame(metrics_list)
        metrics_file_path = os.path.join(save_dir, f'FILEDOUBLE_{dataset_name}.csv')
        metrics_df.to_csv(metrics_file_path, index=False)
        
        if avg_vloss < best_vloss:
            patience_counter = 0
            best_vloss = avg_vloss
            checkpoint_name = f'checkpoint_{dataset_name}_{timestamp}.pth'
            best_model_path = os.path.join(save_dir, checkpoint_name)
            torch.save(model.state_dict(), best_model_path)
            print(f'Checkpoint salvato per epoca {epoch+1} a {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience_threshold:
                stop_training = True
    
    print(f'Risultati salvati in {metrics_file_path}')

    if best_model_path is not None:
        train_dict["model"].load_state_dict(torch.load(best_model_path))
        print(f'Miglior modello caricato da {best_model_path}')
    return best_model_path

def load_best_model(model, best_model_path):
    model.load_state_dict(torch.load(best_model_path))
    return model

def evaluate_model(device: torch.device, model: NewPredModel, loss_fn: MSELoss, test_loader: DataLoader, dataset_name: str, save_dir: str, best_model_path: str) -> dict:
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

    # Preparare i risultati per essere salvati
    results_df = pd.DataFrame({
        'smiles': all_smiles,
        'predicted_value': all_outputs.flatten(),
        'actual_value': all_labels.flatten(),  # Aggiunta dei valori reali per confronto
        'dataset_name': [dataset_name] * len(all_smiles),  # Aggiunta del nome del dataset
        'model_weights': [os.path.basename(best_model_path)] * len(all_smiles)  # Nome del peso utilizzato per la valutazione
    })

    # Salvare il file risultato.csv nella cartella del dataset
    results_path = os.path.join(save_dir, f'{dataset_name}_risultato.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Risultati salvati in {results_path}')

    return {
        'Test Loss': avg_loss,
        'Test RMSE': rmse,
        'Test MAE': mae,
        'Test MedAE': medae,
        'Test R2': r2
    }
def train_on_all_datasets():
    # Definizione del percorso ai dataset
    directory_path = "/home/fvilla/prova/DATI/POLYONE_DA_USARE"
    dataset_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    # Definire il dispositivo all'interno della funzione
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Creazione del file .txt per salvare i risultati
    result_txt_path = "results_summary.txt"
    
    with open(result_txt_path, 'w') as result_txt_file:
        result_txt_file.write("Inizio addestramento\n")  # Aggiunta di un'intestazione
        result_txt_file.flush()  # Flush per forzare la scrittura immediata
        for dataset_file in dataset_files:
            dataset_name = os.path.splitext(dataset_file)[0]
            print(f"Inizio addestramento per il dataset: {dataset_file}")
            result_txt_file.write(f"Dataset: {dataset_name}\n")
            
            dataset_path = os.path.join(directory_path, dataset_file)
            dataset = pd.read_csv(dataset_path, delimiter=',', encoding="utf-8-sig")

            # Creare la directory per salvare i file nel percorso specificato
            save_dir = os.path.join("/home/fvilla/prova/Equilibrio_Chem_polytoken/chem_t64g64", f"Trans_{dataset_name}_UTILS")
            os.makedirs(save_dir, exist_ok=True)

            # Prepara il dataset per l'addestramento
            dataset = dataset[['smiles', 'value']]
            train, test = train_test_split(dataset, train_size=0.8)

            tokenizer = tokenizer_utils.PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=64)
            model = NewPredModel(device)
            max_length = 64
            batch_size1 = 350

            print_memory_usage("Before data loading")

            encodings_list = []
            for i in tqdm(range(0, len(train), batch_size1), desc="Tokenizzazione Train"):
                batch_smiles = train['smiles'].iloc[i:i + batch_size1].tolist()
                batch_encodings = tokenizer(batch_smiles, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
                encodings_list.append(batch_encodings)

            train_encodings = {key: torch.cat([batch[key] for batch in encodings_list], dim=0) for key in encodings_list[0]}
            
            train_encodings_path = os.path.join(save_dir, f'{dataset_name}_train_encodings_T_SEQUENZIALE.pkl')
            with open(train_encodings_path, 'wb') as f:
                pickle.dump(train_encodings, f)
            
            with open(train_encodings_path, 'rb') as f:
                train_encodings = pickle.load(f)

            train_graph_data, valid_train_smiles, valid_train_input_ids = create_graph_data(train['smiles'].tolist(), train_encodings['input_ids'].tolist())

            val_encodings_list = []
            for i in tqdm(range(0, len(test), batch_size1), desc="Tokenizzazione Test"):
                batch_smiles = test['smiles'].iloc[i:i + batch_size1].tolist()
                batch_encodings = tokenizer(batch_smiles, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
                val_encodings_list.append(batch_encodings)

            val_encodings = {key: torch.cat([batch[key] for batch in val_encodings_list], dim=0) for key in val_encodings_list[0]}
            
            val_encodings_path = os.path.join(save_dir, f'{dataset_name}_test_encodings_T_SEQUENZIALE.pkl')
            with open(val_encodings_path, 'wb') as f:
                pickle.dump(val_encodings, f)
            
            with open(val_encodings_path, 'rb') as f:
                val_encodings = pickle.load(f)

            val_graph_data, valid_val_smiles, valid_val_input_ids = create_graph_data(test['smiles'].tolist(), val_encodings['input_ids'].tolist())

            train_t = NewFinetuneDataset(
                {key: torch.tensor(valid_train_input_ids) if key == 'input_ids' else train_encodings[key] for key in train_encodings},
                train['value'][train['smiles'].isin(valid_train_smiles)].values,
                np.array(valid_train_smiles),
                train_graph_data
            )

            train_loader = DataLoader(train_t, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)

            test_t = NewFinetuneDataset(
                {key: torch.tensor(valid_val_input_ids) if key == 'input_ids' else val_encodings[key] for key in val_encodings},
                test['value'][test['smiles'].isin(valid_val_smiles)].values,
                np.array(valid_val_smiles),
                val_graph_data
            )
            test_loader = DataLoader(test_t, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True)

            print("DataLoader creati.")
            print_memory_usage("After creating DataLoaders")

            train_graph_path = os.path.join(save_dir, f'{dataset_name}_train_graph_T_SEQUENZIALE_data.pt')
            val_graph_path = os.path.join(save_dir, f'{dataset_name}_val_graph_T_SEQUENZIALE_data.pt')
            torch.save(train_graph_data, train_graph_path)
            torch.save(val_graph_data, val_graph_path)

            train_data_dict = train_loader
            test_data_dict = test_loader
            train_dict = {}
            embedding_evaluator = model
            optimizer = torch.optim.Adam(embedding_evaluator.parameters(), lr=1e-4)
            scaler = GradScaler()

            

            train_dict["model"] = embedding_evaluator
            train_dict["loss_fn"] = MSELoss()
            train_dict["optimizer"] = optimizer
            train_dict["batch_size"] = 32

            print_memory_usage("Before training")
            best_model_path = train_epochs(train_data_dict, test_data_dict, 1000, train_dict, device, patience_threshold=30, save_dir=save_dir, dataset_name=dataset_name)
            print_memory_usage("After training")

            best_model_path = os.path.join(save_dir, best_model_path)  # Spostare il checkpoint nella cartella corretta
            model = load_best_model(model, best_model_path)

            test_results = evaluate_model(device, model, MSELoss(), test_loader, dataset_name, save_dir, best_model_path)

            # Scrivi i risultati nel file .txt
            result_txt_file.write(f"Test Results for {dataset_name}:\n")
            for key, value in test_results.items():
                result_txt_file.write(f"{key}: {value}\n")
            result_txt_file.write("\n")
            result_txt_file.flush()

            print(f"Completato addestramento per il dataset: {dataset_file}")

            
# Avvia l'addestramento
train_on_all_datasets()