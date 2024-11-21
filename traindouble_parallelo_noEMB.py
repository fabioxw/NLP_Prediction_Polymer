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
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Questa funzione si occupare di ripulire la cache inutilizzata attraverso la libreria garbage
def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

free_memory()
#Qua mi stampo l'utilizzo della memoria fino a quell' istante.
def print_memory_usage(message):
    process = psutil.Process(os.getpid())
    print(f"{message} - Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")


print_memory_usage("Before data loading")

#INPUT : SMILES E TENSORE CHE CONTIENE GLI INPUT IDS (RAPPRESENTANO LA SMILES TOKENIZZATA)
#OUTPUT : IL GRAFO DELLA SMILES 

#Funzione che ha come obiettivo quello di prendere una stringa smiles e i corrispondenti input_ids come tensore di interi.
#Dalla smile, io genero un grafo dalla classe GraphMol.
#Dalla molecola io riesco ad ottenere un oggetto DATA che mi rappresenta il grafo
#Da questo grafo io gli includo la smiles e gli input_ids cosi da potere matchare dopo e correlare senza problemi con la tokenizzazione
def process_smile(smile, input_ids):
    mol = GraphMol(smile)
    graph = Data(x=mol.node_features, edge_index=mol.adjacency_info, edge_attr=mol.edge_features)
    graph.smile = smile
    graph.input_ids = input_ids
    return graph





print_memory_usage("After tokenizing train data")
#INPUT : UNA LISTA DI SMILES,LA SUA LISTA DI INPUT_IDS, GRANDEZZA DEL BATCH.
#OUTPUT: LA LISTA DEI GRAFI 
#L'obiettivo della funzione è quello di creare una lista di grafi a partire da una lista di smiles e liste di corrispondenti input_ids suddividendoli in batch.
def create_graph_data_batch(smiles_list, input_ids_list, batch_size):
    #Mi credo la lista che conterrà i grafi in output
    graph_data_list = []
    
    #Itero attraverso la lista delle smiles nei blocchi di batch
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Creating graph data in batches"):
        #Creo il batch si smiles 
        batch_smiles = smiles_list[i:i + batch_size]
        #Creo il batch di input_ids
        batch_input_ids = input_ids_list[i:i + batch_size]
        #Creo una lista di grafi del batch dandogli la funzione che me li crea in parallelo per il batch.
        batch_graph_data = create_graph_data_parallel(batch_smiles, batch_input_ids)
        #I grafi creati del batch vengono aggiunti in una lista che conterrà tutti  i grafi
        graph_data_list.extend(batch_graph_data)
    return graph_data_list

#INPUT: La lista di stringhe smiles , la lista degli input_ids
#OUTPUT :  Lista di oggetti grafo(uno per ogni smiles)
def create_graph_data_parallel(smiles_list, input_ids_list):
    #Mi definisco la dimensione di processi da utilizzare
    pool_size = min(mp.cpu_count(), 2) #Significa che utilizzo al massimo 2 processi in parallelo nonostante la disponibilità
    #Vengono creati un numero pool di processi, with mi garantisce la chiusura ogni volta che finiscono.
    with mp.Pool(pool_size) as pool:
        #Creazione della lista dei grafi in cui :
          #con pool.imap permette di eseguire più processi parallelamente.
          #impap applica praticamente la funzione process_smile che mi ritorna il grafo data la singola smiles.
          #total= ecc, si occupa di specificare il numero di iterazioni attese per il tqdm.
          #List mi converte l'iterabile e me lo converte in lista
        graph_data_list = list(tqdm(pool.imap(process_smile, zip(smiles_list, input_ids_list)), total=len(smiles_list)))
    print(f"Length of graph_data_list: {len(graph_data_list)}")
    #Ottengo quindi la lista che contiene tutti i grafi generati dalle coppie smiles,input_ids che gli fornisco.
    return graph_data_list

#Metodo per caricare i dati dei grafi, se esiste, altrimenti il salvataggio
def cache_graph_data(file_path, func, *args):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        data = func(*args)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return data

#INPUT  : batch è una lista di elementi, in cui ogni elemento sono un campione di dati preso dal dataset.


#Questa funzione ha lo scopo di combinare una lista di elementi chiamati batch provenienti dal dataloader.
#unendoli in un unico batch di dati da passare al modello
def collate_fn(batch):
    #Per ogni campione nel batch, estraggo la lista delle stringhe SMILES
    smiles = [item['smiles'] for item in batch]
    #Per ogni campione nel batch, estraggo il tensore con le etichette come valore
    labels = torch.stack([item['labels'] for item in batch])
    #Mi genero il grafo per tutte le molecole nel batch
    #from_data_list è di geometric e mi combina tutti i grafi nel gruppo in un BATCH UNICO DI GRAFI.
    graph_data = Batch.from_data_list([process_smile(smile, item['input_ids']) for smile, item in zip(smiles, batch)])
    #Qua vengono combinati i dati in tensori
    token_data = {key: torch.stack([item[key] for item in batch]) for key in batch[0] if key not in ['smiles', 'labels', 'graph']}
    #Ottengo in output il dizionario che poi dovrò passare al modello.
    #Le chiavi di questo dizionario sono : 
        #input_ids,attention_mask,smiles,labels,graph che a loro volta indirizzano a valori con i tensori 
        # o lista di  elementi in quel batch di dati.
    return {**token_data, 'smiles': smiles, 'labels': labels, 'graph': graph_data}



class NewFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, smiles, graph_data):
        self.encodings = encodings
        self.labels = labels
        self.smiles = smiles
        self.graph_data = graph_data
         # Debug 
        print(f"Length of encodings: {len(next(iter(self.encodings.values())))}")
        print(f"Length of labels: {len(self.labels)}")
        print(f"Length of smiles: {len(self.smiles)}")
        print(f"Length of graph_data: {len(self.graph_data)}")
        
    #Metodo per prendere il singolo elemento del dataset  
    #INPUT : index dell' elemento.
    #OUTPUT : resituisce un  dizionario che contiene i dati di tokenizzazione,labels,smiles,grafo di un singolo elemento del dataset.   
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        item['smiles'] = self.smiles[idx]
        item['graph'] = self.graph_data[idx]
        #Ritorno un dizionario che contiene tutti i dati necessari per il singolo campione specifico .
        return item
    
    #Funzione che mi stabilisce la lunghezza
    def __len__(self):
        return len(self.labels)

#INPUT : SMILES DI UNA MOLECOLA
#OUTPUT : GRAFO COMPOSTO TRA TRE ELEMENTI : 
                                        #X : Le features del nodo che rappresentano le caratteristiche dell' atomo.
                                        #Edge_index : Indice il collegamento degli atomi tra di loro.
                                        #Batch : Indice hce indica a quale molecola appartiene ciascun nodo batch.
#Questa classe si occupa della conversione di una smiles in un grafo molecolare rappresentato da tre elementi.
#L'obiettivo principale è estrarre le caratteristiche : 
                                        #Dei nodi
                                        #Dei legami
                                        #Adiacenze del grafo(come matrice di adiacenza)
class GraphMol():
    def __init__(self, smile):
        #Prima di tutto mi piglio la smiles e me la converto in un'oggetto molecola tramite funzione rdkit.
        self.mol = Chem.MolFromSmiles(smile)
        #Queste tre righe mi permettono di calcolare le caratteristiche che ho elencato sopra.
        #Estrapolandole dalla molecola ottenuta sopra.
        self.node_features = self._get_node_features()
        self.edge_features = self._get_edge_features()
        self.adjacency_info = self._get_adjacency_info()

#Questo metodo si occupa di estrarre le caratteristiche di ogni atomo della molecola smiles.
#Le caratteristiche sono : 
            #Numero atomico --> atomicNum
            #Grado dell' atomo --> Degree
            #Carica formale --> FormalCharge
            #Ibridazione --> Hybridazation
            #Aromaticità --> isAromatic
            #Numero di atomi di idrogeno --> TotalNumHs
            #Numero di elettroni radicali --> NumRadicalElectrons
            #Se l'atomo è anello --> isRing
            #Chilarità --> ChiralTag.
            
            
 #OUTPUT : Otterrò da questo metodo un tensore di dimensione (Numero di atomi nella molecola x Numero di caratteristiche (9))
            
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
            #Mi aggiungo la lista delle caratteristiche in questa lista che contiene la lista di tutte le liste delle caratteristiche.
            all_node_feats.append(node_feats)
            #Converto in array numpy
        all_node_feats = np.asarray(all_node_feats)
        
        #Blocco di correzione nel caso in cui nella caratteristica sia presente 0 allora inserisce quella stringa.
        all_node_feats_star = []
        for node in all_node_feats:
            if node[0] == 0:
                all_node_feats_star.append([0, -1, -1, -1, -1, -1, -1, -1, -1])
            else:
                all_node_feats_star.append(node)
         #Converto la lista delle liste nuova e corretta da eventuali errori in un array numpy 
         #Successivamente lo converto in un tensore float       
        all_node_feats_star = np.array(all_node_feats_star)
        return torch.tensor(all_node_feats_star, dtype=torch.float)

#Metodo che si occupa di estrarre le caratteristiche dei legami tra gli atomi nella molecola smiles.
#Estraggo : 
            #Tipo di legame --> BondTypeAdDouble (singolo,doppio,triplo)
            #Se il legame è un anello --> IsInRing 
#OUTPUT : Ottengo un tensore di dimensioni (Numero di archi della molecola x Numero di caratteristiche estratte (2)
#Ogni riga mi rappresenta un legame e ogni colonna mi rappresenta una caratteristica.

    def _get_edge_features(self):
        
    #Inizializzo una lista vuota per memorizzare tutte le caratteristiche.
        all_edge_feats = []
        #itero per tutti i legami della molecola ed estraggo le due caratteristiche di interesse
        for bond in self.mol.GetBonds():
            edge_feats = []
            edge_feats.append(bond.GetBondTypeAsDouble())
            edge_feats.append(bond.IsInRing())
            #Aggiungo due volte il legame per distinguere ciascuna dimensione
            all_edge_feats += [edge_feats, edge_feats]
         #Converto la lista di lista prima in array numpy e poi in tensore float   
        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

#Metodo per ottenere la matrice di adiacenza del grafo (permettendo di rappresentare le connessioni)
#OUTPUT : Ottngo una matrice di dimensioni 2 x numero di legami nella molecola
    def _get_adjacency_info(self):
        #inizializzo una lista vuosta per memorizzare le informazioni di adiacenza.
        edge_indices = []
        
        #itero su tutti i legami della molecole estraendo gli indici i e j per le coppie di legami.
        for bond in self.mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            #inserisco due volte i valori per rappresentare il legame che si svolge in modo biderzionale.
            edge_indices += [[i, j], [j, i]]
        #converto la lista in un tensor e lo traspongo 
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        #ottengo quindi un tensore che è la mia matrice di adiacenza
        return edge_indices

#LA RETE NEUTRALE CONVOLUZIONALE CHE PRENDE IN INPUT UN GRAFO DELLA MOLECOLA PER OTTENERE UNA RAPPRESENTAZIONE VETTORIALE
#OVVERO EMBEDDING DELLA MOLECOLA CHE è UN TENSORE DI DIEMSNIONE [BATCH,256]

class GCNEmbeddingModel(nn.Module):
    #prima di tutto specifico la dimensione delle caratteristiche dei nodi in input(9 nel nostro caso).
    #Significa che ogni nodo, atomo ha un vettore delle caratteristiche di 9 elementi  
    def __init__(self, feature_node_dim=9):
        super(GCNEmbeddingModel, self).__init__()
        
        #Mi effettuo una serie di convoluzione per grafi,modificandomi la grandezza dello spazio
        self.conv1 = GraphConv(feature_node_dim, 32)
        self.conv2 = GraphConv(32, 64)
        self.conv3 = GraphConv(64, 128)
        self.conv4 = GraphConv(128, 256)
        #Questo viene applicato per normalizzare il vettore delle caratteristiche ad ogni livello.
        #Evitando la divergenza dei valori.
        self.ln1 = LayerNorm([32])
        self.ln2 = LayerNorm([64])
        self.ln3 = LayerNorm([128])
        self.ln4 = LayerNorm([256])

#Qui definiscono come passano i dati attraverso la rete.
            #X mi rappresenta un tensore di dimensione [numero di nodi nel batch,numero di caratteristiche del nodo (9)].
            #edge_index è il tensore che rappresenta la matrice di adiacenza calcolata prima su adiacency info.[2 X numero di archi presenti]
            #Batch : vettore che indica a quale grafo appartiene ciascun nodo.
  
  #Approfondimento batch.          
 #Visto che ogni grafo di ogni smiles viene processato in parallelo e i grafi possono avere 
 # diversi nodi e collegamenti, bisogna gestire i nodi di grafi diversi e distinguerli.
 #Questo tensore Batch è un vettore che aiuta ad identificare a quale grafo appartiene ciascun nodo.
 #Se nel batch ho due grafi (uno di 3 nodi e uno di due nodi).
 #Avrei un vettore di Batch [0,0,0,1,1] signfiica che i primi 3 nodi appartengono al primo e i due al secondo.
 
 #Il vettore di batch viene utilizzato nel calcolo di global_poll per capire quali nodi appartengono allo stesso grafo e capire quali usare per calcolare il numero.
 
 
    def forward(self, x, edge_index, batch):
        #Applico la convoluzione che aggrega le informazioni dei nodi vicini 
        x = self.conv1(x, edge_index)
        #Applico normalizzazione
        x = self.ln1(x)
        #Applico la funzione d'attivazione relu.
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
        #Dopo l'ultimo livello di convoluzione, viene eseguito un pooling globale.
        #Questo prende la medie delle caratteristiche dei nodi per ogni grafo nel batch.
        #Producendo un embedding per ciascun grafo.
        x_embed = global_mean_pool(x, batch)
        return x_embed
    #X_EMBED : Un tensore che contiene un vettore di embedding per ciascun grafo.
    #Questa rappresentazione dopo il pooling mi diventa un tensore con una riga per ogni molecola.
    #per cui se il batch è di 8 allora avrò una dimensione di 8,256 come x_embed
'''
class NewPredModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        #Mi passo il device su cui eseguire il modello
        self.device = device
        #Modello chemberta
        self.chemBERTA = RobertaForSequenceClassification.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250", num_labels=1).to(device)
        #Mi passo la wdMPNN
        self.wdMPNN = GCNEmbeddingModel().to(device)
        
    #Metodo che definisci il come il modello elabora i dati in input per ottenere l'output.
    #INPUT :  i token,maschere d'attenzione,grafi sottoforma dei tre elementi.
              #i primi due hanno dimensione [batch_size,lunghezza della sequenza (256)]
    #OUTPUT : 
    def forward(self, input_ids, attention_masks, graph_data):
        #Il grafo me lo sposto sul device
        graph_data = graph_data.to(self.device)
        #Genero l'embedding passandogli i tre elementi che mi rappresentano il grafo. ottenendo cosi.
        #un tensore di dimensione [batch,256]
        graph_vector = self.wdMPNN(graph_data.x, graph_data.edge_index, graph_data.batch)
        print(f"input_ids shape: {input_ids.shape}")
        print(f"graph_vector shape: {graph_vector.shape}")
        #Mi controllo se il numero di embedding generati dal grafo corrisponda al numero di molecole.
        if graph_vector.size(0) != input_ids.size(0):
            graph_vector = graph_vector.view(input_ids.size(0), -1)
        print(f"graph_vector shape after adjustment: {graph_vector.shape}")
        
        #Concateno le due rappresentazioni lungo l'asse creando un nuovo tensore [BATCH,512]
        cat_tensor = torch.cat((input_ids, graph_vector), dim=1).long()
        print(f"cat_tensor shape: {cat_tensor.shape}")
        #Espendo la maschera d'attenzione alla dimensione pari al cat_tensor di prima
        attention_masks = torch.cat((attention_masks.to(self.device), torch.ones(attention_masks.size(0), graph_vector.size(1)).to(self.device)), dim=1)
        attention_masks = attention_masks.long()
        #Da qua mi calcolo l'output del modello
        out = self.chemBERTA(input_ids=cat_tensor, attention_mask=attention_masks)
        #Logit è la parte dell' output accessibile e rappresenta la previsione del modello
        logits = out.logits
        #Ritorno la previsione del modello
        return logits
'''
#Codice che definisce il modello composto dai due modelli in riferimento
class NewPredModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        #Viene spostato il modello sul device
        self.device = device
        
        # Modello RoBERTa pre-addestrato senza classificatore finale, che mi fornisce un embedding di 768
        self.chemBERTA = RobertaModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_396_250").to(device)
        
        # Modello GCN per ottenere embedding dai grafi
        self.wdMPNN = GCNEmbeddingModel().to(device)
        
        #Sarebbe la somma delle dimensioni dell' embedding di chemberta+quello della gcn
        hidden_size = self.chemBERTA.config.hidden_size + 256  # 768 (RoBERTa) + 256 (GCN)
        self.regressor = nn.Linear(hidden_size, 1).to(device)  # Regressione lineare con output continuo

    def forward(self, input_ids, attention_masks, graph_data):
        # Spostiamo i dati del grafo sul device corretto
        graph_data = graph_data.to(self.device)
        
        # Otteniamo gli embedding dal modello GCN per il grafo
        graph_vector = self.wdMPNN(graph_data.x, graph_data.edge_index, graph_data.batch)
        
        # Otteniamo gli embedding dei token dalla chemBERTA (senza il classificatore finale)
        chemberta_output = self.chemBERTA(input_ids=input_ids.to(self.device), attention_mask=attention_masks.to(self.device))
        
        # Prendiamo l'embedding del primo token [CLS] che rappresenta la sequenza
        token_embedding = chemberta_output.last_hidden_state[:, 0, :]
        
        # Concateniamo l'embedding del grafo e l'embedding di RoBERTa lungo l'asse delle caratteristiche (dim=1)
        combined_embedding = torch.cat((token_embedding, graph_vector), dim=1)
        
        # Passiamo l'embedding concatenato al regressore lineare per ottenere un valore continuo
        output = self.regressor(combined_embedding)
        
        return output





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
#Funzione che si occupa di addestrare il modello per una singola epoca.GLI PASSO : 
#IL DEVICE sui cui eseguire l'addestramento.
#Il modello che utilizzo per addestrare.
#La loss da utilizzare.
#L'ottimizzatore
#scaler : Oggetto per la scalatura automatica
#l'indice di epoca corrente.
#batch_size : la dimensione del batch.
#token_loader : Il DataLoader 
def train_one_epoch(device: torch.device, model: NewPredModel, loss_fn: MSELoss, optimizer: torch.optim.Optimizer, scaler, epoch_index: int, batch_size: int, token_loader: DataLoader) -> float:
    running_loss = 0 #Variabile che accumulerà la perdita totale dell'epoca.
    last_loss = 0.  #Utilizzata per memorizzare la loss media alla fine
    all_labels = []  #Lista che raccoglie le etichette
    all_outputs = [] #Lista che raccoglie le predizioni per il calcolo delle metriche
    
    #Metto il modello in modalità TRAIN E INIZIA
    model.train()
    #Inizio a contare per capire quanto poi sta l'epoca.
    start_time = time.time()
    
    #Itero su tutti i batch  del dataloader
    for (i, token_data) in enumerate(tqdm(token_loader, desc=f"Training Epoch {epoch_index+1}")):
        #debug
        print(token_data.keys())
        
        #Mi prendo tutti i dati e me li sposto sul device.
        tokens = token_data['input_ids'].to(device)
        attention_masks = token_data['attention_mask'].to(device)
        labels = token_data['labels'].to(device)
        graph_data = token_data['graph'].to(device)
        #Mi aggiungo una dimensione alla labels perchè deve avere la stessa dimensione con l'outputs
        #perchè devo poi andare a confrontare loro due per la loss_fn dopo.
        labels = labels.unsqueeze(1)
        
        #Azzero i gradienti
        optimizer.zero_grad()
        
        #Abilito la precisione mista riducendo consumo di memorie e aumentando la velocità
        with autocast():
            #Mi calcolo la predizione
            outputs = model(input_ids=tokens, attention_masks=attention_masks, graph_data=graph_data)
            #Mi calcolo la loss tra la predizione e l'etichetta
            loss = loss_fn(outputs, labels)
            
        #Calcola il gradiente della loss
        scaler.scale(loss).backward()
        #Aggiorna i pesi del modello
        scaler.step(optimizer)
        #Aggiorno 
        scaler.update()
        #Aggiungo la loss che ho calcolato prima in runnig_loss che mi accumula tutte le loss 
        running_loss += loss.item()
        #Aggiungo le etichette in questa lista
        all_labels.extend(labels.detach().cpu().numpy())
        #Aggiungo le predizioni in questa lista
        all_outputs.extend(outputs.detach().cpu().numpy())
        
        #finisco il tempo di addestramento e me lo calcolo
        end_time = time.time()
        epoch_time = end_time - start_time
        #Stampo la loss corrente per il batch di riferimento
        print(f"Epoch {epoch_index}, Batch {i}, Loss: {loss.item()}")
    #Finita l'epoca, mi calcolo la perdita media  dividento la somma totale delle loss diviso la grandezza del dataloader.     
    last_loss = running_loss / len(token_loader)
    #Stampo l'epoca e la loss media dell' epoca.
    print(f'Epoch {epoch_index} completed. Average Loss: {last_loss:.4f}')
    #Converto la lista  di etichetta in array numpy
    all_labels = np.array(all_labels)
    #Converto la lista di predizioni in array numpy
    all_outputs = np.array(all_outputs)
    #Mi calcolo tutte le metriche  passandogli le liste
    train_rmse = np.sqrt(mean_squared_error(all_labels, all_outputs))
    train_mae = mean_absolute_error(all_labels, all_outputs)
    train_medae = median_absolute_error(all_labels, all_outputs)
    train_r2 = r2_score(all_labels, all_outputs)
    #Mi stampo le metriche per l'epoca 
    print(f'Training Metrics - Epoch: {epoch_index} RMSE: {train_rmse:.5f} MAE: {train_mae:.5f} MedAE: {train_medae:.5f} R2: {train_r2:.5f}')
    return last_loss, train_rmse, train_mae, train_medae, train_r2

#Processo di addestramento per più epoche
def train_epochs(train_data_dict: dict, val_data_dict: dict, num_epochs: int, train_dict: dict, device: torch.device, patience_threshold: int) -> None:
    #Mi segno il timestamp per poi usarlo nel checkpoint
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #Inizializzo una variabile per rappresentare la migliore loss 
    best_vloss = float('inf')
    best_model_path = None  # Variabile per memorizzare il percorso del miglior modello
    stop_training = False   #Flag per determinare se stoppare o meno l'addestramento
    metrics_list = []       #Mi scrivo una lista che raccoglie le metriche per ogni epoca 
    scaler = GradScaler()   #Variabile per istanziare la precisione mista.

   #Eseguo l'addestramento per il numero di epoche
    for epoch in range(num_epochs):
        #Se il flag è True allora blocco
        if stop_training:
            print(f"Early stopping bloccato in epoca: {epoch + 1}")
            break
        
        #Registro l'orario di inizio d'epoca
        start_time = time.time()
        #Imposto il modello im modalità addestramento.
        model = train_dict["model"].train(True)
        #Mi calcolo gli elementi sulla singola epoca
        avg_loss, train_rmse, train_mae, train_medae, train_r2 = train_one_epoch(device, model, train_dict["loss_fn"], train_dict["optimizer"], scaler, epoch, train_dict["batch_size"], train_data_dict)
        
        #INIZIO PARTE DI VALIDAZIONE
        
        #Inizializzo la loss di valizazione a 0
        curr_vloss = 0.0
        all_vlabels = [] #Mi segno una lista di etichette
        all_voutputs = []  #Mi segno la lista
        
        #SETTO IN MODALITà VALUTAZIONE
        model.eval()
        
        #Disabilito il calcolo dei gradienti
        with torch.no_grad():
            #Per tutto il dataloader
            for (_, vtoken_data) in enumerate(val_data_dict):
                #Mi prendo tutte le informazioni che mi interessano dal dizionario
                tokens = vtoken_data['input_ids'].to(device)
                attention_masks = vtoken_data['attention_mask'].to(device)
                vlabels = vtoken_data['labels'].to(device)
                graph_data = vtoken_data['graph'].to(device)
                
                #Attivazione della precisione mista
                with autocast():
                    #Calcolo la previsione dando le informazioni al modello
                    voutputs = model(input_ids=tokens, graph_data=graph_data, attention_masks=attention_masks)
                    loss = train_dict["loss_fn"]  #prendo la funzione di loss
                    vloss = loss(voutputs, vlabels.unsqueeze(1)) #mi calcolo la loss
                    curr_vloss += vloss.item() #mi sommo la loss in una variabile che me le accumula per l'epoca
                all_vlabels.extend(vlabels.detach().cpu().numpy()) #mi salvo l'etichetta nella lista
                all_voutputs.extend(voutputs.detach().cpu().numpy()) #mi salvo la previsione nella lista
        
        #Fine dell' epoca 
        end_time = time.time()
        epoch_time = end_time - start_time
        #avg_vloss = curr_vloss / (epoch + 1)
        avg_vloss = curr_vloss / len(val_data_dict) #Calcolo la media della loss
        #scheduler.step(avg_vloss)

        
        #Mi converto le liste in array numpy
        all_vlabels = np.array(all_vlabels)
        all_voutputs = np.array(all_voutputs)
        #Calcolo le varie metriche
        val_rmse = np.sqrt(mean_squared_error(all_vlabels, all_voutputs))
        val_mae = mean_absolute_error(all_vlabels, all_voutputs)
        val_medae = median_absolute_error(all_vlabels, all_voutputs)
        val_r2 = r2_score(all_vlabels, all_voutputs)
        #Stampo i risultati
        print(f'Epoch: {epoch} Train Loss: {avg_loss:.5f} Validation Loss: {avg_vloss:.5f}')
        print(f'Validation Metrics - Epoch: {epoch} RMSE: {val_rmse:.5f} MAE: {val_mae:.5f} MedAE: {val_medae:.5f} R2: {val_r2:.5f}')
        
        #Mi salvo in questa lista tutte le metriche di addestramento e validazione
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
        #Prendo la lista e la srutto per creare un dataframe per salvarmi le metriche
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv('/home/fvilla/prova/Encoder/FILE_DOUBLEBERTa/FILEDOUBLE.csv', index=False)
        
        #Gestisco la pazienza
        if avg_vloss < best_vloss:
            patience_counter = 0
            best_vloss = avg_vloss
            best_model_path = '/home/fvilla/prova/Encoder/checkpoint_PARALLELODOUBLE/checkpoint_{}.pth'.format(timestamp)
            torch.save(model.state_dict(), best_model_path)
            print(f'Checkpoint salvato per epoca {epoch+1} a {best_model_path}')
        else:
            patience_counter += 1
            if patience_counter >= patience_threshold:
                stop_training = True
    
    print('Risultati salvati in FILEDOUBLE.csv')

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
    model.load_state_dict(torch.load(best_model_path))
    print(f"Pesi caricati da: {best_model_path}")

    return model

def evaluate_model(device: torch.device, model: NewPredModel, loss_fn: MSELoss, test_loader: DataLoader,best_model_path : str) -> dict:
    model.eval()
    all_labels = []
    all_outputs = []
    total_loss = 0.0

    with torch.no_grad():
        for token_data in tqdm(test_loader, desc="Evaluating"):
            tokens = token_data['input_ids'].to(device)
            attention_masks = token_data['attention_mask'].to(device)
            labels = token_data['labels'].to(device)
            graph_data = token_data['graph'].to(device)
            
            labels = labels.unsqueeze(1)
            with autocast():
                outputs = model(input_ids=tokens, attention_masks=attention_masks, graph_data=graph_data)
                loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            all_labels.extend(labels.detach().cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    rmse = np.sqrt(mean_squared_error(all_labels, all_outputs))
    mae = mean_absolute_error(all_labels, all_outputs)
    medae = median_absolute_error(all_labels, all_outputs)
    r2 = r2_score(all_labels, all_outputs)
    print(f"Valutazione del modello con i pesi caricati da: {best_model_path}")
    print(f'Test Loss: {avg_loss:.5f}')
    print(f'Test RMSE: {rmse:.5f}')
    print(f'Test MAE: {mae:.5f}')
    print(f'Test MedAE: {medae:.5f}')
    print(f'Test R2: {r2:.5f}')

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
#Istruzioni altrimenti chemberta fa capricci
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Definisco il device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = pd.read_csv("/home/fvilla/prova/DATI/smiles_with_values.csv", delimiter=',', encoding="utf-8-sig")
#dataset = pd.read_csv("/home/fvilla/prova/MTL_Khazana/Khazana_ei.csv", delimiter=',', encoding="utf-8-sig")
#dataset = pd.read_csv("/home/fvilla/prova/polyset/SMILES_Ei.csv", delimiter=',', encoding="utf-8-sig")
#dataset = pd.read_csv("/home/fvilla/prova/polyset/smiles100k_less_than_61.csv", delimiter=',', encoding="utf-8-sig")
dataset = dataset[['smiles','value']]
print("sono 1")
#Mi divideo il dataset in due 
train, test = train_test_split(dataset, train_size=0.8)
print("sono 2")
print("sono 3")
#Mi prendo il tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
max_length = 256
print("sono 4")
batch_size1 = 350
encodings_list = [] #Lista che mi conterrà le tokenizzazioni
print_memory_usage("Before data loading")
print("sono 5")
#Mi itero le tokenizzazioni a batch a batch. per la tokenizzazione del dataset di train
for i in tqdm(range(0, len(train), batch_size1), desc="Tokenizzazione Train"):
    #Mi suddiviso la lunghezza dei dati in gruppi di batch. e mi muovo all' interno del batch ed estraggo
    #il sottoinsieme dei dati che sono di quel batch vengono estratti e diventano una lista
    batch_smiles = train['smiles'].iloc[i:i + batch_size1].tolist()
    #Mi converto sto batch di smiles e me le aggiungo nella lista successivamente in un unica lista
    batch_encodings = tokenizer(batch_smiles, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    encodings_list.append(batch_encodings)
    
    
print_memory_usage("After tokenizing train data")
print("sono 7")
train_encodings = {key: torch.cat([batch[key] for batch in encodings_list], dim=0) for key in encodings_list[0]}
print("sono 7.1")
with open('train_encodings_PARALLELO.pkl', 'wb') as f:
    pickle.dump(train_encodings, f)
with open('train_encodings_PARALLELO.pkl', 'rb') as f:
    train_encodings = pickle.load(f)
    
print_memory_usage("After loading train encodings")
print("sono tokenizzazione completata e dati salvati pER IL TRAIN")
val_encodings_list = []
print("sono train1")
#Viene fatta la tokenizzazione in batch come prima ma per i dataset di validation
for i in tqdm(range(0, len(test), batch_size1), desc="Tokenizzazione Test"):
    batch_smiles = test['smiles'].iloc[i:i + batch_size1].tolist()
    batch_encodings = tokenizer(batch_smiles, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    val_encodings_list.append(batch_encodings)
print("sono train 2")
val_encodings = {key: torch.cat([batch[key] for batch in val_encodings_list], dim=0) for key in val_encodings_list[0]}
print("sono train3")
with open('test_encodings_PARALLELO.pkl', 'wb') as f:
    pickle.dump(train_encodings, f)
with open('test_encodings_PARALLELO.pkl', 'rb') as f:
    train_encodings = pickle.load(f)
    
print_memory_usage("After loading test encodings")
print("sono tokenizzazione completata e dati salvati pER IL TEST")
#Estraggo le labels e le smiles dai dataset di train e di test
train_labels = train['value'].values
train_smiles = train['smiles'].values
test_labels = test['value'].values
test_smiles = test['smiles'].values
print(f"Length of train_encodings: {len(train_encodings['input_ids'])}")
print(f"Length of train_labels: {len(train_labels)}")
print(f"Length of train_smiles: {len(train_smiles)}")
print("sono qua")
#Mi creo i grafi sia per il train che per il val, usando : 
#-Cache_graph_data batch per la creazione stessa e poi il risultato viene memorizzato 
# a poco a poco attraverso la funzine cache_data_graph
train_graph_data = cache_graph_data('train_graph_PARALLELO_data.pkl', create_graph_data_batch, train_smiles.tolist(), train_encodings['input_ids'].tolist(),500)
val_graph_data = cache_graph_data('val_graph_PARALLELO_data.pkl', create_graph_data_batch, test_smiles.tolist(), val_encodings['input_ids'].tolist(),500)
print_memory_usage("After creating graph data")
print("sono qui")
print(f"Length of train_graph_data: {len(train_graph_data)}")
#Da qui mi credo un oggetto di tipo newfinetunedataset che combina le tokenizzazioni,label e smiles 
#Per poi creare il dataloader per gestire il caricamento durante il train
train_t = NewFinetuneDataset(train_encodings, train_labels,train_smiles,train_graph_data)
train_loader = DataLoader(train_t, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
print("sono finetunetrain")
#Faccio la tokenizzazione delle smiles di test perchè gli servono come parametro.
test_encodings = tokenizer(test['smiles'].tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
print("sono finetunetest")
#Mi creo un oggetto di tipo newfinetunedataset per raccogliere le informazioni per poi passarle al dataloader.
test_t = NewFinetuneDataset(test_encodings, test_labels, test_smiles,val_graph_data)
print("sono prima del dataloader")
test_loader = DataLoader(test_t, batch_size=16, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
print("DataLoader creati.")
print_memory_usage("After creating DataLoaders")
print("sono 5")
torch.save(train_graph_data, 'train_graph_PARALLELO_data.pt')
torch.save(val_graph_data, 'val_graph_PARALLELO_data.pt')
print("sono 6")
train_graph_data = torch.load('train_graph_PARALLELO_data.pt')
val_graph_data = torch.load('val_graph_PARALLELO_data.pt')
print("sono 8")
train_data_dict = train_loader
test_data_dict = test_loader
train_dict = {}
print("sono 12")
embedding_evaluator = NewPredModel(device)
optimizer = torch.optim.Adam(embedding_evaluator.parameters(), lr=1e-4)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

print("sono 13")
scaler = GradScaler()
train_dict["model"] = embedding_evaluator
train_dict["loss_fn"] = MSELoss()
train_dict["optimizer"] = optimizer
train_dict["batch_size"] = 16




best_model_path=train_epochs(train_data_dict, test_data_dict,300, train_dict, device, patience_threshold=30)


# Carica i pesi migliori
model = load_best_model(embedding_evaluator, best_model_path)

# Valuta il modello sui dati di test
test_results = evaluate_model(device, model, MSELoss(), test_loader,best_model_path)

# Salva i risultati delle metriche di valutazione in un file CSV
metrics_df = pd.DataFrame([test_results])
metrics_df.to_csv('/home/fvilla/prova/Encoder/FILE_DOUBLEBERTa/best_value_chembPARALLELO_double_mix.csv', index=False)