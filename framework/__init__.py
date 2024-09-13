# from .models import GCN, GAT, GIN, RGCN, RGAT, GCNDelete, GATDelete, GINDelete, RGCNDelete, RGATDelete
from .trainer.base import Trainer
from .models import LightGCN, LightGCNDelete, GCN, UltraGCN, SimGCL
from .trainer.gnndelete import GNNDeleteTrainer
from .trainer.retrain import RetrainTrainer
from .trainer.utu import UtUTrainer
from .trainer.RecUn import RecUnTrainer
from .trainer.User_trainer import UserTrainer
from .trainer.descent_to_delete import DtdTrainer
from.trainer.RecEraser_Ori import RecEraser_Ori
from.trainer.RecEraser import RecEraserTrainer
from.trainer.RecUn_ultra import RecUnUltraTrainer
# from .trainer.retrain_sim import RetrainTrainer_sim
from .trainer.CEU import CEUTrainer
# from .trainer.cgu import CGUTrainer

trainer_mapping = {
    'original': Trainer,
    'gnndelete': GNNDeleteTrainer,
    'retrain': RetrainTrainer,
    # 'retrain_sim': RetrainTrainer_sim,
    
    # 'LightGCN': LightGCN
    
    'RecUn': RecUnTrainer,
    'RecUn_ultra': RecUnUltraTrainer,
    
    'user': UserTrainer,
    'descent_to_delete': DtdTrainer,
    'RecEraser_Ori':RecEraser_Ori,
    'RecEraser':RecEraserTrainer,
    'utu': UtUTrainer,
    'ceu': CEUTrainer,
    # 'cgu': CGUTrainer,


}



def get_model(args, mask_1hop=None, mask_2hop=None, num_nodes=None, num_edge_type=None, constraint_mat=None, ii_constraint_mat=None, ii_neighbor_mat=None):

    if 'gnndelete' in args.unlearning_model:
        # model_mapping = {'gcn': GCNDelete, 'gat': GATDelete, 'gin': GINDelete, 'rgcn': RGCNDelete, 'rgat': RGATDelete}
        model_mapping = {'LightGCN':LightGCNDelete}
        pass

    else:
        model_mapping = {'LightGCN': LightGCN, 'gcn': GCN, 'UltraGCN': UltraGCN, 'simGCL': SimGCL}

    if  args.gnn == 'UltraGCN':
        return model_mapping[args.gnn](args, constraint_mat, ii_constraint_mat, ii_neighbor_mat)
    else:      
        return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes, num_edge_type=num_edge_type)


def get_trainer(args):
    # if args.gnn in ['rgcn', 'rgat']:
    #     return kg_trainer_mapping[args.unlearning_model](args)

    # else:
    #     return trainer_mapping[args.unlearning_model](args)
    return trainer_mapping[args.unlearning_model](args)
