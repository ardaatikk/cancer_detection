from torch.utils.data import DataLoader
from dataset import Dataset
import matplotlib.pyplot as plt
import sys 
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', default='../images/images', help='Image directory')
parser.add_argument('--data_file', default='../myTrain.csv', help='Data file')

FLAGS = parser.parse_args()

dataset = Dataset(img_dir= FLAGS.img_dir, annotations_file= FLAGS.data_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0) 


#batch_size is the number of the samples the dataloader will load at a time
#num_workers is the number of worker processes to use for dataloading. If it is 0, then data will load in the main process.
#if num_worker > 0 data will loaded parallel using more workers and it will be faster but it can cause memory and CPU utilization errors.


for i, (image, label) in enumerate(dataloader):  # Loop through each batch of data in the dataloader
    fig, ax = plt.subplots(1,4)
    for j in range(len(image)): # Loop through each image in the batch
        index = i * len(image) + j # Calculate the index of the current image in the dataset
        ax[j].imshow(image[j].permute(1, 2, 0)) # Display the image using matplotlib, converting the tensor to a numpy array and rearranging the dimensions
        ax[j].set_title(f"Index: {index}, Label: {label[j]}") # Add a title to the image, showing its index in the dataset and its label
    plt.show() # Display the image on the screen
    if i == 3:
        break
# images.shape part gives (batch_size, number of channels, height, width)
# labels.shape part gives how many images are there in the batch

sys.exit()

#######################################################
# access the first item in the dataset if you want only one item same code without "for" and change "i" with the index number
for i in range(10):
    index = i
    item = dataset[index] # using [] to get the idx number
    image, label = item
    print(image.shape)
# display the image using matplotlib
    plt.imshow(image.permute(1, 2, 0))  # permute to change the order of the dimensions from (Channel, Height, Width) to (Height, Width, Channel)
    # using this permute method is important because 'matplotlib' expects images in this order. 
    plt.title(f'Label: {label}')
    # setting the title to the image's label
    plt.show()
    
########################################################   

