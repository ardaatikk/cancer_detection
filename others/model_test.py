from model import MyModel
import sys
import torch

# Create an instance of MyModel
model = MyModel()

# # Access the weights of each layer
# conv1_weights = model.conv1.weight
# conv2_weights = model.conv2.weight
# fc1_weights = model.fc1.weight
# fc2_weights = model.fc2.weight

# # Print the shape of the weights
# print("conv1 weights:", conv1_weights.shape)
# print("conv2 weights:", conv2_weights.shape)
# print("fc1 weights:", fc1_weights.shape)
# print("fc2 weights:", fc2_weights.shape)

# for name, param in model.named_parameters():  # named_parameters()  function gives us parameters as tuples names and values.
#     print(param)

# sys.exit()

# for name, param in model.named_parameters():  # named_parameters()  function gives us parameters as tuples names and values.
#     print("Parameter name:", name)
#     print("Parameter shape:", param.shape)
    
# for name, param in model.named_parameters():   
#     print("name:", name, "trainable:", param.requires_grad)
#     # print("Parameter name:", name)
#     # print("Trainable:", param.requires_grad)
# sys.exit()
 
print("Before freezing:")
# for name, param in model.named_parameters():
#     if name.startswith('conv1'):
#         print(name, param.requires_grad)

for name, param in model.named_parameters():   
    print("name:", name, "trainable:", param.requires_grad)

# Freeze the conv1 layer by setting requires_grad to False for its parameters
for name, param in model.named_parameters():
    if name.startswith('conv1'):
        param.requires_grad = False

# Print requires_grad attribute of conv1 parameters after freezing
print("\nAfter freezing:")
for name, param in model.named_parameters():
    if name.startswith('conv1'):
        print(name, param.requires_grad)

for name, param in model.named_parameters():   
    print("name:", name, "trainable:", param.requires_grad)
# print("before")
# conv1_bias = model.conv1.bias
# print("conv1 bias:", conv1_bias.shape)
# print("after")
# model.conv1.bias.data = torch.zeros(32, dtype=torch.float32)
# print("conv1 bias:", conv1_bias)

# print("before")
# conv1_weight = model.conv1.weight
# print("conv1 weight:", conv1_weight.shape)
# print("after")
# model.conv1.weight.data = torch.zeros(32,3,3,3, dtype=torch.float32)
# print("conv1 weight:", conv1_weight)