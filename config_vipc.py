from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.VIPC_PATH        = '/home/ycx/dataset/ShapeNetViPC/ShapeNetViPC-Dataset'#path to dataset
#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.DATA_perfetch                          = 8
#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = '/home/ycx/code/VIPC-My/project_logs'#path to save checkpoints and logs
__C.CONST.DEVICE                                 = '0,1'
__C.DIR.CHECKPOINTS                              = '../checkpoints/VIPC'
#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.EGIInet                              = edict()
__C.NETWORK.EGIInet.embed_dim                    = 192
__C.NETWORK.EGIInet.depth                        = 6
__C.NETWORK.EGIInet.img_patch_size               = 14
__C.NETWORK.EGIInet.pc_sample_rate               = 0.125
__C.NETWORK.EGIInet.pc_sample_scale              = 2
__C.NETWORK.EGIInet.fuse_layer_num               = 2
__C.NETWORK.shared_encoder                       = edict()
__C.NETWORK.shared_encoder.block_head            = 12
__C.NETWORK.shared_encoder.pc_h_hidden_dim       = 192
#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 16 #TODO: 训练批次大小
__C.TRAIN.N_EPOCHS                               = 300
__C.TRAIN.SAVE_FREQ                              = 40
__C.TRAIN.LEARNING_RATE                          = 0.0001
__C.TRAIN.LR_MILESTONES                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.LR_DECAY_STEP                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.WARMUP_STEPS                           = 1
__C.TRAIN.GAMMA                                  = 0.7
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.CATE                                   = 'plane' #TODO:训练使用的类别，不选all的话，写上想要训练的单独类别
__C.TRAIN.d_size                                 = 1
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.CATE                                    = 'plane'
__C.TEST.BATCH_SIZE                              = 16 #TODO: 测试批次大小
__C.CONST.WEIGHTS = r"/home/ycx/code/checkpoints/EGIInet/plane-ckpt-best.pth" #TODO:path to pre-trained checkpoints


#
# PreTrain 对比学习预训练中使用的参数
#
__C.PRETRAIN                                        = edict()
__C.PRETRAIN.BATCH_SIZE                             = 8 #TODO: 对比学习训练批次大小
__C.PRETRAIN.N_EPOCHS                               = 120 #对比学习训练轮数
__C.PRETRAIN.LEARNING_RATE                          = 0.001
__C.PRETRAIN.CATE                                   = 'plane' #TODO:训练使用的类别，不选all的话，写上想要训练的单独类别




#P2C自监督
__C.P2C                                          = edict()
__C.P2C.LEARNING_RATE                            = 0.0001
__C.P2C.WEIGHT_DECAY                             = 0.0001
__C.P2C.BATCH_SIZE                               = 32
__C.P2C.CATE                                     = 'plane'  

__C.P2C.decay_steps                              = 60
__C.P2C.lr_decay                                 = 0.9
__C.P2C.lowest_decay                             = 0.02
__C.P2C.max_epoch                                = 200
__C.P2C.SAVE_FREQ                                = 40

#P2C MODEL
__C.P2Cnet                                      = edict()
__C.P2Cnet.num_group                            = 64
__C.P2Cnet.group_size                           = 32                               
__C.P2Cnet.mask_ratio                           = [10, 50, 4]
__C.P2Cnet.feat_dim                             = 256  #潜在空间维度           
__C.P2Cnet.n_points                             = 2048                   
__C.P2Cnet.shape_recon_weight                   = 1000                  
__C.P2Cnet.shape_matching_weight                = 1000              
__C.P2Cnet.latent_weight                        = 100           
__C.P2Cnet.nbr_ratio                            = 2.0               
__C.P2Cnet.support                              = 24                   
__C.P2Cnet.neighborhood_size                    = 32            
__C.P2Cnet.manifold_weight                      = 0    