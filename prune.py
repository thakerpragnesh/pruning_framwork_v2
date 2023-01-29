# In[1]: Import all the require files
import torch
import load_dataset as dl
import load_model as lm
import train_model as tm
import initialize_pruning as ip
import facilitate_pruning as fp
import torch.nn.utils.prune as prune
import os  # use to access the files
from datetime import date

# In[2]: Set date to store information in the logs 
today = date.today()
d1 = today.strftime("%d_%m") #ex "27_11"

# In[3] Set pathe of all the directories

# Specific Dir for selected program and dataset 
program_name = 'pruning'
selectedModel = 'vgg16_IntelIc_Prune'
selected_dataset_dir = 'IntelIC'

# Common Dir use at many loc
dir_home_path = '/home/pragnesh/'
dir_specific_path =f'{program_name}/{selected_dataset_dir}'

# Model Paths
model_dir   = f"{dir_home_path}Model/{dir_specific_path}"
loadModel = False
load_path = f'{model_dir}/{selectedModel}'
is_transfer_learning = False

# Dataset Paths
dataset_dir = f"{dir_home_path}Dataset/{selected_dataset_dir}" 
train_folder = 'train'
test_folder = 'test'

# Logs Path
log_dir       = f"{dir_home_path}Logs/{dir_specific_path}" 
logResultFile = f'{log_dir}/result.log'
outFile       = f'{log_dir}/lastResult.log'
outLogFile    = f'{log_dir}/outLogFile.log'


# In[4]: Check Cuda Devices
if torch.cuda.is_available():
    device1 = torch.device('cuda')
else:
    device1 = torch.device('cpu')


# In[5]: Function to create folder if not exist
def ensure_dir(dir_path):
    directory = os.path.dirname(dir_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir(f'{dir_home_path}Model/{dir_specific_path}/')
ensure_dir(model_dir)
ensure_dir(f'{dir_home_path}Logs/{dir_specific_path}/')
ensure_dir(log_dir)

# In[6]: Set Image Properties
dl.set_image_size(224)
dl.set_batch_size = 16
dataLoaders = dl.data_loader(set_datasets_arg=dataset_dir,
                             selected_dataset_arg='',
                             train_arg=train_folder, test_arg=test_folder)

# In[7]: Load appropriate model
if loadModel:  # Load the saved trained model
    new_model = torch.load(load_path, map_location=torch.device(device1))
else:  # Load the standard model from library
    new_model = lm.load_model(model_name='vgg16', number_of_class=6,
                              pre_train_flag=is_transfer_learning,
                              freeze_feature_arg=False, device_l=device1)

opt_func = torch.optim.Adam

# In[8]: Create require lists for pruning
block_list = []; feature_list = []; prune_count = []
conv_layer_index = []; module = []
new_list = []; candidate_conv_layer = []
layer_number = 0; st = 0; en = 0


def initialize_lists_for_pruning():
    global module, block_list, feature_list, prune_count, conv_layer_index
    
    module = ip.make_list_conv_param(new_model)
    block_list = ip.create_block_list(new_model)  # ip.getBlockList('vgg16')
    feature_list = ip.create_feature_list(new_model)
    prune_count = ip.get_prune_count(module=module, blocks=block_list, max_pr=.1)
    conv_layer_index = ip.find_conv_index(new_model)
    



# In[11]:
def compute_conv_layer_saliency_channel_pruning(module_cand_conv, block_list_l, block_id, k=1):
    global layer_number
    candidate_convolution_layer = []
    end_index = 0
    for bl in range(len(block_list_l)):
        start_index = end_index
        end_index = end_index + block_list_l[bl]
        if bl != block_id:
            continue

        for lno in range(start_index, end_index):
            print(lno)
            candidate_convolution_layer.append(
                
                fp.compute_saliency_score_channel(module_cand_conv[lno]._parameters['weight'],
                                                  n=1, 
                                                  dim_to_keep=[0], 
                                                  prune_amount=prune_count[lno]))
        break
    return candidate_convolution_layer


# In[12]: Compute mask matrix using saliency score
prune_index = []
layer_base=0
class ChannelPruningMethodSaliency(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    

    def compute_mask(self, t, default_mask):
        layer_prune_index = []
        global prune_index
        global layer_number
        global layer_base
        mask = default_mask.clone()
        i=layer_number-layer_base
        for j in range(len(new_list[i])):
            k = new_list[i][j][0]
            layer_prune_index.append(k)
            #print(k)
            mask[k] =0
            
        prune_index.append(layer_prune_index)
        return mask
    
def channel_unstructured_saliency(module, name):
    ChannelPruningMethodSaliency.apply(module, name)
    return module


# In[]:
layer_base=0
def set_grad_true(param):
    param.requires_grad=True
def iterative_channel_pruning_saliency_block_wise(new_model_arg, prune_module, 
                                             block_list_l, prune_epochs):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nPruning Process Start")
    out_file.close()
    print("\nPruning Process Start")
    # pc = [1, 3, 9, 26, 51]
    
    global new_list
    global layer_base
    global d1
    
    for e in range(prune_epochs):
        start = 0
        end = len(block_list_l)
        for blkId in range(start, end):
            # 2 Compute distance between kernel for candidate conv layer
            '''
            new_list = compute_conv_layer_saliency_channel_pruning(module_candidate_convolution=prune_module,
                                                                   block_list_l=block_list_l, block_id=blkId)
            '''
            new_list = compute_conv_layer_saliency_channel_pruning(module_cand_conv=prune_module, 
                                                                   block_list_l=block_list_l, block_id=blkId)
            # 5 perform Custom pruning where we mask the prune weight
            for j in range(block_list_l[blkId]):
                if blkId < 2:
                    layer_number_to_prune = (blkId * 2) + j
                else:  # blkId >= 2:
                    layer_number_to_prune = 4 + (blkId - 2) * 3 + j
                channel_unstructured_saliency(
                    module=prune_module[layer_number_to_prune], 
                    name='weight')
            new_list = None
        
       
        # 10.  Train pruned model
        with open(outLogFile, 'a') as out_file:
            out_file.write(f'\n ...Deep Copy Completed...on {d1}')
            out_file.write('\n Fine tuning started....on {d1}')
        out_file.close()
        print(('\n Fine tuning started....on {d1}'))
        tm.fit_one_cycle( dataloaders=dataLoaders,
                          train_dir=dl.train_directory, test_dir=dl.test_directory,
                          # Select a variant of VGGNet
                          model_name='vgg16', model=new_model, device_l=device1,
                          # Set all the Hyper-Parameter for training
                          epochs=8, max_lr=0.001, weight_decay=0.01, L1=0.01, grad_clip=0.1,
                          opt_func=opt_func, log_file=logResultFile)



# In[]:

def compute_conv_layer_dist_channel_pruning(module_cand_conv, block_list_l, block_id):
    global layer_number
    candidate_convolution_layer = []
    end_index = 0
    for bl in range(len(block_list_l)):
        start_index = end_index
        end_index = end_index + block_list_l[bl]
        if bl != block_id:
            continue

        with open(outLogFile, "a") as out_file:
            out_file.write(f'\nblock ={bl} blockSize={block_list_l[bl]}, start={start_index}, End={end_index}')
        out_file.close()
        # newList = []
        # candidList = []
        for lno in range(start_index, end_index):
            # layer_number =st+i
            with open(outLogFile, 'a') as out_file:
                out_file.write(f"\nlno in compute candidate {lno}")
            out_file.close()
            candidate_convolution_layer.append(fp.compute_distance_score_channel(
                module_cand_conv[lno]._parameters['weight'],
                n=1,
                dim_to_keep=[0],
                prune_amount=prune_count[lno]))
        break
    return candidate_convolution_layer


# In[ ]:
class ChannelPruningMethodSimilarities(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        with open(outLogFile, "a") as log_file:
            log_file.write("\n Executing Compute Mask")
        log_file.close()
        mask = default_mask.clone()
        # mask.view(-1)[::2] = 0
        size = t.shape
        print(f"\n{size}")
        with open(outLogFile, "a") as log_file:
            log_file.write(f'\nLayer Number:{layer_number} \nstart={st} \nlength of new list={len(new_list)}')
        log_file.close()
        for k1 in range(len(new_list)):
            for k2 in range(len(new_list[layer_number - st][k1])):
                i = new_list[layer_number - st][k1][k2][1]
                j = new_list[layer_number - st][k1][k2][0]
                if k1 == j:
                    print(":", end='')
                # print(f"i= {i} , j= {j}")

                mask[i][j] = 0
        return mask


# In[ ]:


def channel_unstructured_similarities(kernel_module, name):
    ChannelPruningMethodSimilarities.apply(kernel_module, name)
    return kernel_module


initialize_lists_for_pruning()

# In[ ]:
layer_base=0
def iterative_channel_pruning_similarities_block_wise(new_model_arg, prune_module, 
                                             block_list_l, prune_epochs):
    with open(outLogFile, "a") as out_file:
        out_file.write("\nPruning Process Start")
    out_file.close()
    # pc = [1, 3, 9, 26, 51]
    
    global new_list
    global layer_base
    
    for e in range(prune_epochs):
        start = 0
        end = len(block_list_l)
        for blkId in range(start, end):
            # 2 Compute distance between kernel for candidate conv layer
            new_list = compute_conv_layer_dist_channel_pruning(module_cand_conv=prune_module,
                                                                   block_list_l=block_list_l, block_id=blkId)
            # 5 perform Custom pruning where we mask the prune weight
            for j in range(block_list_l[blkId]):
                if blkId < 2:
                    layer_number_to_prune = (blkId * 2) + j
                else:  # blkId >= 2:
                    layer_number_to_prune = 4 + (blkId - 2) * 3 + j
                channel_unstructured_similarities(
                    module=prune_module[layer_number_to_prune], 
                    name='weight')
            new_list = None
        
        # 10.  Train pruned model
        with open(outLogFile, 'a') as out_file:
            out_file.write('\n ...Deep Copy Completed...')
            out_file.write('\n Fine tuning started....')
        out_file.close()

        tm.fit_one_cycle( dataloaders=dataLoaders,
                          train_dir=dl.train_directory, test_dir=dl.test_directory,
                          # Select a variant of VGGNet
                          model_name='vgg16', model=new_model, device_l=device1,
                          # Set all the Hyper-Parameter for training
                          epochs=8, max_lr=0.001, weight_decay=0.01, L1=0.01, grad_clip=0.1,
                          opt_func=opt_func, log_file=logResultFile)
        
       

# In[ ]:
def compute_conv_layer_saliency_kernel_pruning(module_candidate_convolution, block_list_l, block_id, k=1):
    return module_candidate_convolution + block_list_l + block_id + k
    # replace the demo code above


# In[ ]:
class KernelPruningSaliency(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        return 0


# In[ ]:
def kernel_unstructured_saliency(kernel_module, name):
    KernelPruningSaliency.apply(kernel_module, name)
    return kernel_module




# In[ ]:
initialize_lists_for_pruning()
iterative_channel_pruning_saliency_block_wise(new_model_arg=new_model,
                                               prune_module=module, 
                                               block_list_l=block_list, 
                                               prune_epochs=6)


# In[ ]:
initialize_lists_for_pruning()
iterative_channel_pruning_similarities_block_wise(new_model_arg=new_model, 
    prune_module=module, block_list_l=block_list, prune_epochs=6)


