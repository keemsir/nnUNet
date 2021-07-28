## Command

'''
source /home/ncc/.bashrc

conda info --envs
conda activate envs_name
conda activate

## final model name

If you interrupted the training,

then then rename model_best.model(사본).pkl
to
model_final_checkpoint.model.pkl

and

model_best.model
to
model_final_checkpoint(사본).model

for the given fold.
'''


## path setting

'''
export nnUNet_raw_data_base="media/ncc/nnUNet_raw_data_base"
export nnUNet_preprocessed="media/ncc/nnUNet_preprocessed"
export RESULTS_FOLDER="media/ncc/nnunet_trained_models"

# Train order

# Convert
nnUNet_convert_decathlon_task -i media/ncc/Tasks/Task05_Prostate -output_task_id OUTPUT_TASK_ID # Task04_Hippocampus
# Pipeline configuration
nnUNet_plan_and_preprocess -t OUTPUT_TASK_ID

# Start train

# Recent
nnUNet_train 2d nnUNetTrainerV2 510 all --npz
nnUNet_train 3d_fullres nnUNetTrainerV2 510 all --npz -c --cuda_device 1
nnUNet_train 3d_lowres nnUNetTrainerV2 510 all --npz --cuda_device 0

ing))) nnUNet_train 3d_fullres nnUNetTrainerV2_Loss_MSE 510 all # shape 안맞

nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes 510 all --npz --cuda_device 0
nnUNet train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes_focalLoss 510 0 --npz
# predict
(If RESULTS_FOLDER don't contain cv_niftis)
nnUNet_determine_postprocessing -t 511 -m 2d # creating cv

nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_NAME_OR_ID -m CONFIGURATION --save_npz
nnUNet_predict -i media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task509_Spleen_reloc/imagesTs -o OUTPUT_DIRECTORY/2d_predict -t 511 -tr nnUNetTrainerV2 -m 2d --num_threads_preprocessing 1 --save_npz
nnUNet_predict -i media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task509_Spleen_reloc/imagesTs -o OUTPUT_DIRECTORY/3d_fullres_predict -t 511 -tr nnUNetTrainerV2 -m 3d_fullres --num_threads_preprocessing 1 --save_npz
nnUNet_predict -i media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task509_Spleen_reloc/imagesTs -o OUTPUT_DIRECTORY/3d_cascade_predict -t 511 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_cascade_fullres --num_threads_preprocessing 1 --save_npz

nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t
nnUNet_predict -i media/ncc/Tasks/onlytest_spleen/ -o OUTPUT_DIRECTORY/onlytest/3d_fullres -t 511 -tr nnUNetTrainerV2 -m 3d_fullres --save_npz

nnUNet_print_pretrained_model_info Task04_Hippocampus

nnUNet_ensemble -f FOLDER1 FOLDER2 ... -o OUTPUT_FOLDER -pp POSTPROCESSING_FILE


# oncologist
nnUNet_convert_decathlon_task -i media/ncc/Tasks/Task06_Lung_staple -output_task_id OUTPUT_TASK_ID
nnUNet_plan_and_preprocess -t 10
nnUNet_train 3d_fullres nnUNetTrainerV2 106 0
# model list = ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres']

(before) nnUNet_ensemble -f 0 1 2 3 4 -o OUTPUT_DIRECTORY/staple_ensemble -pp POSTPROCESSING_FILE
nnUNet_ensemble -f OUTPUT_DIRECTORY/2d_predict OUTPUT_DIRECTORY/3d_cascade_predict OUTPUT_DIRECTORY/3d_fullres_predict -o OUTPUT_DIRECTORY/509_ensemble -pp POSTPROCESSING_FILE

nnUNet_train 2d nnUNetTrainerV2 106 all -c
nnUNet_train 3d_fullres nnUNetTrainerV2 106 0 -c

nnUNet_train 3d_fullres nnUNetTrainerV2 509 4 --cuda_device 0 --npz

nnUNet_train 3d_lowres nnUNetTrainerV2 510 all --npz --cuda_device 0 -c

# Cascade train method
nnUNet_train 3d_lowres nnUNetTrainerV2 106 FOLD
nnUNet_train 3d_cascade_fullres nnUNetTrainerV2CascadeFullRes TaskXXX_MYTASK FOLD
nnUNet_train CONFIGURATION TRAINER_CLASS_NAME TASK_NAME_OR_ID FOLD


# Error massage
FutureWarning: Non-finite norm encountered in torch.nn.utils.clip_grad_norm_; continuing anyway.
Note that the default behavior will change in a future release to error out if a non-finite total norm is encountered.
At that point, setting error_if_nonfinite=false will be required to retain the old behavior.
torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)

# ============= training cmd ==================
nnUNet_train 3d_fullres nnUNetTrainerV2_Loss_MSE 510 0 --cuda_device 1

## For GPU memory kill (GPU 1 memory initialize) ##
sudo fuser -v /dev/nvidia*
for i in $(sudo lsof /dev/nvidia1 | grep python | awk '{print $2}' | sort -u); do kill -9 $i; done

'''

## Train Data

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
cur_dir = os.getcwd()
# task_dir = os.path.join(cur_dir, 'media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task509_Spleen_reloc/')
task_dir = os.path.join(cur_dir, 'media/ncc/Tasks/Task09_Spleen_reloc/')
train_image_dir = os.path.join(task_dir, 'imagesTr/')
train_label_dir = os.path.join(task_dir, 'labelsTr/')
test_dir = os.path.join(task_dir,'imagesTs/')
test_label_dir = os.path.join(task_dir,'labelsTs/')


train_img_list = os.listdir(train_image_dir)
train_img_name = train_img_list[np.random.randint(0,len(train_img_list))]
train_img = np.array(nib.load(os.path.join(train_image_dir,train_img_name)).dataobj)[:,:,:5]
# train_label_name = train_img_name[:train_img_name.find('_000')]+'.nii.gz' # ('_0000.nii.gz')
train_label = np.array(nib.load(os.path.join(train_label_dir,train_img_name)).dataobj)[:,:,:5]

train_img_num = train_img_name[train_img_name.find('_')+1:train_img_name.find('.nii')]

print(train_img.shape,train_label.shape)


## Prediction TestTest Visualization (2d_predict, 3d_cascade_predict, 3d_fullres_predict, 509_ensemble)

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
TASK_NAME = 'spleen_2.nii.gz'
TASK_NAME_0 = 'spleen_2_0000.nii.gz'
Task_name_npz = 'spleen_10.npz'

cur_dir = os.getcwd()
task_dir = os.path.join(cur_dir, 'media/ncc/Tasks/onlytest_spleen')
# output_dir = os.path.join(cur_dir, 'OUTPUT_DIRECTORY/onlytest')
output_dir = os.path.join(cur_dir, 'OUTPUT_DIRECTORY/510')

PRED1_NAME = '3dfulres_mcc'
PRED2_NAME = 'fulres_topk'

pred1 = os.path.join(output_dir, '{}/{}'.format(PRED1_NAME, TASK_NAME))
pred2 = os.path.join(output_dir, '{}/{}'.format(PRED2_NAME, TASK_NAME))
# pred1 = os.path.join(output_dir, '{}'.format(TASK_NAME))
# pred2 = os.path.join(output_dir, '{}'.format(TASK_NAME))

ts_image = os.path.join(task_dir, 'imagesTs/{}'.format(TASK_NAME_0))
ts_label = os.path.join(task_dir, 'labelsTs/{}'.format(TASK_NAME))
pred_label = os.path.join(task_dir, TASK_NAME_0)



result_img = np.array(nib.load(ts_image).dataobj)
_, _, RESULT_LEN = result_img.shape
RESULT_LEN_RAN = np.random.randint(0, RESULT_LEN)
RESULT_LEN_RAN = 75 # custom number

test_img = np.array(nib.load(ts_image).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
test_label = np.array(nib.load(ts_label).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
pr_label1 = np.array(nib.load(pred1).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
pr_label2 = np.array(nib.load(pred2).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]



max_rows = 4
max_cols = test_img.shape[2]

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test Image' + str(idx + 1) + '_{}'.format(RESULT_LEN_RAN + idx))
    axes[0, idx].imshow(test_img[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('Ground Truth Label' + str(idx + 1))
    axes[1, idx].imshow(test_label[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('do_mirroring_cc_{}_'.format(PRED1_NAME) + str(idx + 1))
    axes[2, idx].imshow(pr_label1[:, :, idx])
for idx in range(max_cols):
    axes[3, idx].axis("off")
    axes[3, idx].set_title('dont_mirroring_cc_{}_'.format(PRED2_NAME) + str(idx + 1))
    axes[3, idx].imshow(pr_label2[:, :, idx])

plt.suptitle('Path : {}'.format(output_dir))
plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()

## cv Visualization 위에 먼저 실행


IMG_NUM = 3
RESULT_LEN_RAN = 45

cv_dir = 'media/ncc/nnunet_trained_models/nnUNet/3d_fullres/Task509_Spleen_reloc/nnUNetTrainerV2__nnUNetPlansv2.1/'

cv_raw_dir = os.path.join(cv_dir, 'all/validation_raw') # validation_raw, cv_niftis_raw'
cv_raw_list = os.listdir(cv_raw_dir)
cv_raw_list.sort()
cv_raw_name = cv_raw_list[IMG_NUM]
cv_raw_img = np.array(nib.load(os.path.join(cv_raw_dir, cv_raw_name)).dataobj)
cv_raw_img_range = np.array(nib.load(os.path.join(cv_raw_dir, cv_raw_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

cv_pp_dir = os.path.join(cv_dir, 'all/validation_raw_postprocessed') # validation_raw_postprocessed, cv_niftis_postprocessed
cv_pp_list = os.listdir(cv_pp_dir)
cv_pp_list.sort()
cv_pp_name = cv_pp_list[IMG_NUM]
cv_pp_img = np.array(nib.load(os.path.join(cv_pp_dir, cv_pp_name)).dataobj)
cv_pp_img_range = np.array(nib.load(os.path.join(cv_pp_dir, cv_pp_name)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]


print('CV Row Image Shape: {}'.format(cv_raw_name), cv_raw_img.shape)
print('CV PP Image Shape: {}'.format(cv_pp_name), cv_pp_img.shape)

max_rows = 3
max_cols = test_img.shape[2]

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test Image' + str(idx + 1))
    axes[0, idx].imshow(test_img[:, :, idx]) # , cmap="gray"
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('cv_raw' + str(idx + 1))
    axes[1, idx].imshow(cv_raw_img_range[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('cv_pp_img' + str(idx + 1))
    axes[2, idx].imshow(cv_pp_img_range[:, :, idx])

plt.subplots_adjust(wspace=.1, hspace=.1)
plt.show()

## Prediction Visualization (2d_predict, 3d_cascade_predict, 3d_fullres_predict, 509_ensemble)

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# path
cur_dir = os.getcwd()
# task_dir = os.path.join(cur_dir, 'media/ncc/nnUNet_raw_data_base/nnUNet_raw_data/Task509_Spleen_reloc/')
task_dir = os.path.join(cur_dir, 'media/ncc/Tasks/Task509_Spleen_reloc/')
train_image_dir = os.path.join(task_dir, 'imagesTr/')
train_label_dir = os.path.join(task_dir, 'labelsTr/')
test_dir = os.path.join(task_dir,'imagesTs/')
test_label_dir = os.path.join(task_dir,'labelsTs/')

IMG_NUM = 0
PRED_LIST = ['2d_predict', '3d_fullres_predict', '3d_cascade_predict', '509_ensemble']

test_img_list = os.listdir(test_dir)
test_img_list.sort()
IMG_NAME = test_img_list[IMG_NUM]

# os.path.join(cur_dir, 'OUTPUT_DIRECTORY/')

result_img = np.array(nib.load(os.path.join(test_dir, IMG_NAME)).dataobj)
_, _, RESULT_LEN = result_img.shape
RESULT_LEN_RAN = np.random.randint(0, RESULT_LEN)
# RESULT_LEN_RAN = 70

test_img = np.array(nib.load(os.path.join(test_dir, IMG_NAME)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
test_label = np.array(nib.load(os.path.join(test_label_dir, IMG_NAME)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]

for i in PRED_LIST:
    result_dir = os.path.join(cur_dir, 'OUTPUT_DIRECTORY/{}/'.format(i))
    predict_label = np.array(nib.load(os.path.join(result_dir, IMG_NAME)).dataobj)[:, :, RESULT_LEN_RAN:RESULT_LEN_RAN + 5]
    if i == '2d_predict':
         nnunet_2d_label = predict_label
    if i == '3d_fullres_predict':
         fullres_label = predict_label
    if i == '3d_cascade_predict':
         cascade_label = predict_label
    if i == '509_ensemble':
         ensemble_label = predict_label
    print(predict_label.shape)


test_img_num = IMG_NAME[IMG_NAME.find('_')+1:IMG_NAME.find('.nii')]


max_rows = 6
max_cols = test_img.shape[2]

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, figsize=(20, 8))
for idx in range(max_cols):
    axes[0, idx].axis("off")
    axes[0, idx].set_title('Test Image' + str(idx + 1) + '({}_{})'.format(test_img_num,RESULT_LEN_RAN+idx))
    axes[0, idx].imshow(test_img[:, :, idx])
for idx in range(max_cols):
    axes[1, idx].axis("off")
    axes[1, idx].set_title('Ground Truth Label' + str(idx + 1))
    axes[1, idx].imshow(test_label[:, :, idx])
for idx in range(max_cols):
    axes[2, idx].axis("off")
    axes[2, idx].set_title('2d_Predicted' + str(idx + 1))
    axes[2, idx].imshow(nnunet_2d_label[:, :, idx])
for idx in range(max_cols):
    axes[3, idx].axis("off")
    axes[3, idx].set_title('3d_fullres_Predicted' + str(idx + 1))
    axes[3, idx].imshow(fullres_label[:, :, idx])
for idx in range(max_cols):
    axes[4, idx].axis("off")
    axes[4, idx].set_title('3d_cascade_Predicted' + str(idx + 1))
    axes[4, idx].imshow(cascade_label[:, :, idx])
for idx in range(max_cols):
    axes[5, idx].axis("off")
    axes[5, idx].set_title('Ensemble_Predicted' + str(idx + 1))
    axes[5, idx].imshow(ensemble_label[:, :, idx])

plt.subplots_adjust(wspace=.1, hspace=.2)
plt.show()

## Task dataset.json
import os

cur_dir = os.getcwd()


task_dir = os.path.join(cur_dir, 'media/ncc/Tasks/Task09_Spleen/')

imagesTr_dir = os.path.join(task_dir, 'imagesTr/')
imagesTr_list = os.listdir(imagesTr_dir)
imagesTr_list.sort()
imagesTs_dir = os.path.join(task_dir, 'imagesTs/')
imagesTs_list = os.listdir(imagesTs_dir)
imagesTs_list.sort()
labelsTr_dir = os.path.join(task_dir, 'labelsTr/')
labelsTr_list = os.listdir(labelsTr_dir)
labelsTr_list.sort()

print(len(imagesTr_list), len(imagesTs_list), len(labelsTr_list))

for i in range(len(imagesTr_list)):

    print('{"image":"./imagesTr/','{}'.format(imagesTr_list[i]), '","label":"./labelsTr/', '{}'.format(labelsTr_list[i]), '"},' ,sep='', end='')


for i in range(len(imagesTs_list)):

    print('"./imagesTs/','{}'.format(imagesTs_list[i]), '",' ,sep='')


## excise

import torch
from torch import nn, Tensor

class Custom_MSELoss(nn.MSELoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # if len(target.shape) == len(input.shape):
        #     assert target.shape[1] == 1
        #     target = target[:, 0]
        return super().forward(input, target)

loss = Custom_MSELoss()


# input = test_label[:,:,0]
input_tensor = torch.from_numpy(test_label)
input_flat = torch.flatten(input_tensor)
input_reshape = input_flat.reshape(5,512,512)

# target = pr_label1[:,:,0]
target_tensor = Tensor(pr_label1)
target_flat = torch.flatten(target_tensor)
target_reshape = target_flat.reshape(5,512,512)

output = loss(input_reshape, target_reshape)
output.backward()

