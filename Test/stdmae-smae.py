import torch

loaded_tensor=torch.load("C:\Code\BasicTS-master\checkpoints\Mask_SMAE\PEMS08_100_2016_12\c420b6232d390580a5af0d22e463c036\Mask_SMAE_100.pt")
#dict_keys(['epoch', 'model_state_dict', 'optim_state_dict', 'best_metrics'])
keys = loaded_tensor.keys()


print("------------------**keys**------------------")
print(keys)

epoch_value = loaded_tensor['epoch']
model_state_dict_value = loaded_tensor['model_state_dict'] #模型的参数（如权重和偏置）
optim_state_dict_value = loaded_tensor['optim_state_dict'] #包含了优化器的状态，如当前的学习率、动量缓冲区和参数的历史梯度等
best_metrics_value = loaded_tensor['best_metrics']         #验证值最好指标

# print("------------------**epoch_value**------------------")
# print(epoch_value)


print("------------------**model_state_dict_value**------------------")
# print(model_state_dict_value)
#print(model_state_dict_value.keys())
print(model_state_dict_value['encoder.transformer_encoder.layers.0.self_attn.in_proj_weight'].shape)
print(model_state_dict_value['encoder.transformer_encoder.layers.0.self_attn.in_proj_bias'].shape)


# print("------------------**optim_state_dict_value**------------------")
# print(optim_state_dict_value)


# print("------------------**best_metrics_value**------------------")
# print(best_metrics_value)