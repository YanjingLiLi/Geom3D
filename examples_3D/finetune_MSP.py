import os
import time
from sklearn.metrics import roc_auc_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool
from tqdm import tqdm
from torch_geometric.data import Data

from config import args
from Geom3D.datasets import DatasetGVP, DatasetMSP
from Geom3D.models import ProNet, MQAModel, GearNetIEConv, CD_Convolution
from dataset_loader import *




def model_setup():
    num_class = 1
    graph_pred_linear = None

    if args.model_3d == "GVP":
        node_in_dim = (6, 3)
        node_h_dim = (100, 16)
        edge_in_dim = (32, 1)
        edge_h_dim = (32, 1)
        model = MQAModel(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim)
        
        ns, _ = node_h_dim
        ns *= 2
        drop_rate = 0.1
        graph_pred_linear = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate),
            nn.Linear(2*ns, num_class),
        )

    elif args.model_3d == "GearNet":
        input_dim = 21
        model = GearNetIEConv(
            input_dim=input_dim, embedding_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=7,
            batch_norm=True, concat_hidden=True, short_cut=True, readout="sum", layer_norm=True, dropout=0.2)

        num_mlp_layer = 3
        hidden_dims = [model.output_dim] * (num_mlp_layer - 1)
        graph_pred_linear = GearNet_layer.MultiLayerPerceptron(
            model.output_dim, hidden_dims + [num_class], batch_norm=True, dropout=0.5)
        
    elif args.model_3d == "GearNet_IEConv":
        input_dim = 21
        model = GearNetIEConv(
            input_dim=input_dim, embedding_dim=512, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=7,
            batch_norm=True, concat_hidden=True, short_cut=True, readout="sum", layer_norm=True, dropout=0.2, use_ieconv=True)

        num_mlp_layer = 3
        hidden_dims = [model.output_dim] * (num_mlp_layer - 1)
        graph_pred_linear = GearNet_layer.MultiLayerPerceptron(
            model.output_dim, hidden_dims + [num_class], batch_norm=True, dropout=0.5)

    elif args.model_3d == "ProNet":
        model = ProNet(
            level=args.ProNet_level,
            dropout=args.ProNet_dropout,
            out_channels=num_class,
            euler_noise=args.euler_noise,
        )

        graph_pred_linear = torch.nn.Sequential()
        out_layers = 2
        hidden_channels=128 * 2
        out_channels=1
        dropout_rate = 0

        for _ in range(out_layers-1):
            graph_pred_linear.add_module("linear", nn.Linear(hidden_channels, hidden_channels))
            graph_pred_linear.add_module("relu", nn.ReLU())
            graph_pred_linear.add_module("dropout", nn.Dropout(dropout_rate))
        graph_pred_linear.add_module("output", nn.Linear(hidden_channels, out_channels))

    elif args.model_3d == "CDConv":
        geometric_radii = [x * args.CDConv_radius for x in args.CDConv_geometric_raddi_coeff]
        model = CD_Convolution(
            geometric_radii=geometric_radii,
            sequential_kernel_size=args.CDConv_kernel_size,
            kernel_channels=args.CDConv_kernel_channels, channels=args.CDConv_channels, base_width=args.CDConv_base_width,
            num_classes=num_class)
        
        graph_pred_linear = MLP(in_channels=args.CDConv_channels[-1] * 2,
                              mid_channels=max(args.CDConv_channels[-1], num_class),
                              out_channels=num_class,
                              batch_norm=True,
                              dropout=0.2)

    else:
        raise Exception("3D model {} not included.".format(args.model_3d))
    return model, graph_pred_linear


def load_model(model, graph_pred_linear, model_weight_file):
    geometric_radii = [x * args.CDConv_radius for x in args.CDConv_geometric_raddi_coeff]
    original_model = CD_Convolution(
        geometric_radii=geometric_radii,
        sequential_kernel_size=args.CDConv_kernel_size,
        kernel_channels=args.CDConv_kernel_channels, channels=args.CDConv_channels, base_width=args.CDConv_base_width,
        num_classes=1195)

    print("Loading from {}".format(model_weight_file))
    if "MoleculeSDE" in model_weight_file:
        model_weight = torch.load(model_weight_file)
        model.load_state_dict(model_weight["model_3D"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])
    else:
        model_weight = torch.load(model_weight_file)
        original_model.load_state_dict(model_weight["model"])
        if (graph_pred_linear is not None) and ("graph_pred_linear" in model_weight):
            graph_pred_linear.load_state_dict(model_weight["graph_pred_linear"])
        for name, param in original_model.named_parameters():
            if "classifier" not in name:  
                model.state_dict()[name].copy_(param)
    return



def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["model"] = model.state_dict()
            if graph_pred_linear is not None:
                saved_model_dict["graph_pred_linear"] = graph_pred_linear.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(epoch, device, loader, optimizer):
    model.train()
    if graph_pred_linear is not None:
        graph_pred_linear.train()

    loss_acc = 0
    num_iters = len(loader)

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)
        # sub_batch_1, sub_batch_2 = Data().to(device), Data().to(device)
        # sub_batch_1.seq, sub_batch_1.side_chain_angle_encoding, sub_batch_1.backbone_angle_encoding, sub_batch_1.coords_ca, sub_batch_1.coords_n, sub_batch_1.coords_c, sub_batch_1.x, sub_batch_1.pos, sub_batch_1.num_nodes, sub_batch_1.edge_index, sub_batch_1.node_s, sub_batch_1.node_v, sub_batch_1.edge_s, sub_batch_1.edge_v = batch.seq_1, batch.side_chain_angle_encoding_1, batch.backbone_angle_encoding_1, batch.coords_ca_1, batch.coords_n_1, batch.coords_c_1, batch.x_1, batch.pos_1, batch.num_nodes_1, batch.edge_index_1, batch.node_s_1, batch.node_v_1, batch.edge_s_1, batch.edge_v_1
        # sub_batch_2.seq, sub_batch_2.side_chain_angle_encoding, sub_batch_2.backbone_angle_encoding, sub_batch_2.coords_ca, sub_batch_2.coords_n, sub_batch_2.coords_c, sub_batch_2.x, sub_batch_2.pos, sub_batch_2.num_nodes, sub_batch_2.edge_index, sub_batch_2.node_s, sub_batch_2.node_v, sub_batch_2.edge_s, sub_batch_2.edge_v = batch.seq_2, batch.side_chain_angle_encoding_2, batch.backbone_angle_encoding_2, batch.coords_ca_2, batch.coords_n_2, batch.coords_c_2, batch.x_2, batch.pos_2, batch.num_nodes_2, batch.edge_index_2, batch.node_s_2, batch.node_v_2, batch.edge_s_2, batch.edge_v_2
        
        if args.model_3d == "ProNet":
            if args.mask:
                # random mask node aatype
                mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
                batch.x[:, 0][mask_indice] = 25
            if args.noise:
                # add gaussian noise to atom coords
                gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
                batch.coords_ca += gaussian_noise
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n += gaussian_noise
                    batch.coords_c += gaussian_noise
            if args.deform:
                # Anisotropic scale
                deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
                batch.coords_ca *= deform
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n *= deform
                    batch.coords_c *= deform

        if args.model_3d == "GVP":
            molecule_3D_repr_1 = model(batch.node_s_1, batch.node_v_1, batch.edge_s_1, batch.edge_v_1, batch.edge_index_1, batch.batch_protein_1, get_repr=True)
            molecule_3D_repr_2 = model(batch.node_s_2, batch.node_v_2, batch.edge_s_2, batch.edge_v_2, batch.edge_index_2, batch.batch_protein_2, get_repr=True)
            molecule_3D_repr = torch.cat((molecule_3D_repr_1, molecule_3D_repr_2), dim=1)
        elif args.model_3d in ["GearNet", "GearNet_IEConv"]:
            molecule_3D_repr = model(batch, batch.node_feature.float())["graph_feature"]
        elif args.model_3d == "ProNet":
            molecule_3D_repr_1 = model(batch.seq_1, batch.coords_n_1, batch.coords_ca_1, batch.coords_c_1, batch.side_chain_angle_encoding_1, batch.backbone_angle_encoding_1, batch.batch_protein_1, get_repr=True)
            molecule_3D_repr_2 = model(batch.seq_2, batch.coords_n_2, batch.coords_ca_2, batch.coords_c_2, batch.side_chain_angle_encoding_2, batch.backbone_angle_encoding_2, batch.batch_protein_2, get_repr=True)
            molecule_3D_repr = torch.cat((molecule_3D_repr_1, molecule_3D_repr_2), dim=1)
        elif args.model_3d == "CDConv":
            molecule_3D_repr_1 = model(batch.seq_1, batch.coords_ca_1, batch.batch_protein_1, split="training", get_repr=True)
            molecule_3D_repr_2 = model(batch.seq_2, batch.coords_ca_2, batch.batch_protein_2, split="training", get_repr=True)
            molecule_3D_repr = torch.cat((molecule_3D_repr_1, molecule_3D_repr_2), dim=1)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze().sigmoid()
        else:
            pred = molecule_3D_repr.squeeze().sigmoid()

        y = batch.y

        loss = criterion(pred.float(), y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_acc += loss.cpu().detach().item()

        if args.lr_scheduler in ["CosineAnnealingWarmRestarts"]:
            lr_scheduler.step(epoch - 1 + step / num_iters)

    loss_acc /= len(loader)
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in [ "ReduceLROnPlateau"]:
        lr_scheduler.step(loss_acc)

    return loss_acc


@torch.no_grad()
def eval(device, loader):
    model.eval()
    if graph_pred_linear is not None:
        graph_pred_linear.eval()
    y_true = []
    y_scores = []
    y_scores_raw = []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for batch in L:
        batch = batch.to(device)
        # sub_batch_1, sub_batch_2 = Data().to(device), Data().to(device)
        # sub_batch_1.seq, sub_batch_1.side_chain_angle_encoding, sub_batch_1.backbone_angle_encoding, sub_batch_1.coords_ca, sub_batch_1.coords_n, sub_batch_1.coords_c, sub_batch_1.x, sub_batch_1.pos, sub_batch_1.num_nodes, sub_batch_1.edge_index, sub_batch_1.node_s, sub_batch_1.node_v, sub_batch_1.edge_s, sub_batch_1.edge_v = batch.seq_1, batch.side_chain_angle_encoding_1, batch.backbone_angle_encoding_1, batch.coords_ca_1, batch.coords_n_1, batch.coords_c_1, batch.x_1, batch.pos_1, batch.num_nodes_1, batch.edge_index_1, batch.node_s_1, batch.node_v_1, batch.edge_s_1, batch.edge_v_1
        # sub_batch_2.seq, sub_batch_2.side_chain_angle_encoding, sub_batch_2.backbone_angle_encoding, sub_batch_2.coords_ca, sub_batch_2.coords_n, sub_batch_2.coords_c, sub_batch_2.x, sub_batch_2.pos, sub_batch_2.num_nodes, sub_batch_2.edge_index, sub_batch_2.node_s, sub_batch_2.node_v, sub_batch_2.edge_s, sub_batch_2.edge_v = batch.seq_2, batch.side_chain_angle_encoding_2, batch.backbone_angle_encoding_2, batch.coords_ca_2, batch.coords_n_2, batch.coords_c_2, batch.x_2, batch.pos_2, batch.num_nodes_2, batch.edge_index_2, batch.node_s_2, batch.node_v_2, batch.edge_s_2, batch.edge_v_2

        if args.model_3d == "ProNet":
            if args.mask:
                # random mask node aatype
                mask_indice = torch.tensor(np.random.choice(batch.num_nodes, int(batch.num_nodes * args.mask_aatype), replace=False))
                batch.x[:, 0][mask_indice] = 25
            if args.noise:
                # add gaussian noise to atom coords
                gaussian_noise = torch.clip(torch.normal(mean=0.0, std=0.1, size=batch.coords_ca.shape), min=-0.3, max=0.3)
                batch.coords_ca += gaussian_noise
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n += gaussian_noise
                    batch.coords_c += gaussian_noise
            if args.deform:
                # Anisotropic scale
                deform = torch.clip(torch.normal(mean=1.0, std=0.1, size=(1, 3)), min=0.9, max=1.1)
                batch.coords_ca *= deform
                if args.ProNet_level != 'aminoacid':
                    batch.coords_n *= deform
                    batch.coords_c *= deform
        
        if args.model_3d == "GVP":
            molecule_3D_repr_1 = model(batch.node_s_1, batch.node_v_1, batch.edge_s_1, batch.edge_v_1, batch.edge_index_1, batch.batch_protein_1, get_repr=True)
            molecule_3D_repr_2 = model(batch.node_s_2, batch.node_v_2, batch.edge_s_2, batch.edge_v_2, batch.edge_index_2, batch.batch_protein_2, get_repr=True)
            molecule_3D_repr = torch.cat((molecule_3D_repr_1, molecule_3D_repr_2), dim=1)
        elif args.model_3d in ["GearNet", "GearNet_IEConv"]:
            molecule_3D_repr = model(batch, batch.node_feature.float())["graph_feature"]
        elif args.model_3d == "ProNet":
            molecule_3D_repr_1 = model(batch.seq_1, batch.coords_n_1, batch.coords_ca_1, batch.coords_c_1, batch.side_chain_angle_encoding_1, batch.backbone_angle_encoding_1, batch.batch_protein_1, get_repr=True)
            molecule_3D_repr_2 = model(batch.seq_2, batch.coords_n_2, batch.coords_ca_2, batch.coords_c_2, batch.side_chain_angle_encoding_2, batch.backbone_angle_encoding_2, batch.batch_protein_2, get_repr=True)
            molecule_3D_repr = torch.cat((molecule_3D_repr_1, molecule_3D_repr_2), dim=1)
        elif args.model_3d == "CDConv":
            molecule_3D_repr_1 = model(batch.seq_1, batch.coords_ca_1, batch.batch_protein_1, get_repr=True)
            molecule_3D_repr_2 = model(batch.seq_2, batch.coords_ca_2, batch.batch_protein_2, get_repr=True)
            molecule_3D_repr = torch.cat((molecule_3D_repr_1, molecule_3D_repr_2), dim=1)

        if graph_pred_linear is not None:
            pred = graph_pred_linear(molecule_3D_repr).squeeze().sigmoid()
        else:
            pred = molecule_3D_repr.squeeze().sigmoid()

        y = batch.y

        y_scores_raw.append(pred)
        pred = (pred >= 0.5).long()
        y_true.append(y)
        y_scores.append(pred)

    for i in range(len(y_scores)):
        if y_scores[i].dim() == 0:
            y_scores[i] = y_scores[i].unsqueeze(0)
            y_scores_raw[i] = y_scores_raw[i].unsqueeze(0)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    y_scores_raw = torch.cat(y_scores_raw, dim=0).cpu().numpy()

    L = len(y_true)
    acc =  sum(y_true == y_scores) * 1. / L

    auroc = roc_auc_score(y_true, y_scores_raw)

    return acc, auroc

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data_root = args.data_root
    
    dataset_class = DatasetMSP

    train_dataset = dataset_class(root=data_root, split='train')
    valid_dataset = dataset_class(root=data_root, split='val')
    test_dataset = dataset_class(root=data_root, split='test')

    if args.model_3d == "GVP":
        data_root = "../data/FOLD_GVP"
        train_dataset = DatasetGVP(
            root=data_root, dataset=train_dataset, split='train', num_positional_embeddings=args.num_positional_embeddings, top_k=args.top_k, num_rbf=args.num_rbf, multi_protein=True)
        valid_dataset = DatasetGVP(
            root=data_root, dataset=valid_dataset, split='val', num_positional_embeddings=args.num_positional_embeddings, top_k=args.top_k, num_rbf=args.num_rbf, multi_protein=True)
        test_dataset = DatasetGVP(
            root=data_root, dataset=test_dataset, split='test', num_positional_embeddings=args.num_positional_embeddings, top_k=args.top_k, num_rbf=args.num_rbf, multi_protein=True)
        
    criterion = nn.BCELoss()

    DataLoaderClass = DataLoaderMultiPro
    dataloader_kwargs = {}
    if args.model_3d in ["GearNet", "GearNet_IEConv"]:
        dataloader_kwargs["collate_fn"] = DatasetFOLD.collate_fn
        DataLoaderClass = TorchDataLoader

    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )
    val_loader = DataLoaderClass(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )
    test_loader = DataLoaderClass(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        **dataloader_kwargs
    )

    model, graph_pred_linear = model_setup()

    if args.input_model_file is not "":
        load_model(model, graph_pred_linear, args.input_model_file)
    model.to(device)
    print(model)
    if graph_pred_linear is not None:
        graph_pred_linear.to(device)
    print(graph_pred_linear)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = [{"params": model.parameters(), "lr": args.lr}]
    if graph_pred_linear is not None:
        model_param_group.append(
            {"params": graph_pred_linear.parameters(), "lr": args.lr}
        )

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model_param_group, lr=args.lr, weight_decay=5e-4, momentum=0.9)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "CosineAnnealingWarmRestarts":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.epochs, eta_min=1e-4
        )
        print("Apply lr scheduler CosineAnnealingWarmRestarts")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        print("Apply lr scheduler ReduceLROnPlateau")
    elif args.lr_scheduler == "StepLRCustomized":
        print("Will decay with {}, at epochs {}".format(args.lr_decay_factor, args.StepLRCustomized_scheduler))
        print("Apply lr scheduler StepLR (customized)")
    else:
        print("lr scheduler {} is not included.".format(args.lr_scheduler))
    global_learning_rate = args.lr

    train_acc_list, val_acc_list, test_acc_list = [], [], []
    train_auroc_list, val_auroc_list, test_auroc_list = [], [], []
    best_val_auroc, best_val_idx = -1e10, 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss_acc = train(epoch, device, train_loader, optimizer)
        print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

        if epoch % args.print_every_epoch == 0:
            if args.eval_train:
                train_acc, train_auroc = eval(device, train_loader)
            else:
                train_acc, train_auroc = 0, 0
            val_acc, val_auroc = eval(device, val_loader)
            test_acc, test_auroc = eval(device, test_loader)

            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            train_auroc_list.append(train_auroc)
            val_auroc_list.append(val_auroc)
            test_auroc_list.append(test_auroc)
            print(
                "train_acc: {:.6f}\ttrain_auroc: {:.6f}\tval_acc: {:.6f}\tval_auroc: {:.6f}\ttest_acc: {:.6f}\ttest_auroc: {:.6f}".format(
                    train_acc, train_auroc, val_acc, val_auroc, test_acc, test_auroc
                )
            )

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_val_idx = len(train_auroc_list) - 1
                if not args.output_model_dir == "":
                    save_model(save_best=True)

        if args.lr_scheduler == "StepLRCustomized" and epoch in args.StepLRCustomized_scheduler:
            print('ChanGINg learning rate, from {} to {}'.format(global_learning_rate, global_learning_rate * args.lr_decay_factor)),
            global_learning_rate *= args.lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = global_learning_rate
        print("Took\t{}\n".format(time.time() - start_time))

    print(
        "best train_acc: {:.6f}\ttrain_auroc: {:.6f}\tval_acc: {:.6f}\tval_auroc: {:.6f}\ttest_acc: {:.6f}\ttest_auroc: {:.6f}".format(
            train_acc_list[best_val_idx],
            train_auroc_list[best_val_idx],
            val_acc_list[best_val_idx],
            val_auroc_list[best_val_idx],
            test_acc_list[best_val_idx],
            test_auroc_list[best_val_idx],
        )
    )

    save_model(save_best=False)