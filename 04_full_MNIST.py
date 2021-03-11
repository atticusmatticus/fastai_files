####
from fastai.vision.all import *
plt.style.use('dark_background')
####

# Download MNIST_SAMPLE dataset which is just 3's and 7's.
####
path = untar_data(URLs.MNIST, dest='/media/xar/barracuda1/fast.ai/data/')
path.ls()
print((path/'training').ls())
print((path/'testing').ls())
####

# number of numerals/catagories.
####
nNum = len((path/'training').ls())
nNum
####

# create list of training files as tensors.
####
train = []
for i in range(nNum):
    train.append( torch.stack( [tensor(Image.open(o)) for o in (path/'training'/str(i)).ls()] ).float()/255 )
    print(i, train[i].shape)
show_images([train[0][0], train[1][0], train[2][0]])
show_images([train[3][0], train[4][0], train[5][0]])
show_images([train[6][0], train[7][0], train[8][0]])
show_images([train[9][0]])
####

# create list of validation images as tensors.
####
valid = []
for i in range(nNum):
    valid.append( torch.stack( [tensor(Image.open(o)) for o in (path/'testing'/str(i)).ls()] ).float()/255 )
    print(i, valid[i].shape)
show_images([valid[0][0], valid[1][0], valid[2][0]])
show_images([valid[3][0], valid[4][0], valid[5][0]])
show_images([valid[6][0], valid[7][0], valid[8][0]])
show_images([valid[9][0]])
####

# concatenate the training images into one tensor of flattened vectors.
####
trainX = torch.cat(train).view(-1, 28*28)
print(trainX.shape)
####
trainY = tensor([ item for sublist in [[i]*len(train[i]) for i in range(nNum)] for item in sublist ]).unsqueeze(1)
print(trainY.shape)
####
trainDS = list(zip(trainX, trainY))
x,y = trainDS[0]
x.shape, y
####
# concatenate the validation images into one tensor of flattened vectors.
####
validX = torch.cat(valid).view(-1, 28*28)
print(validX.shape)
####
validY = tensor([ item for sublist in [[i]*len(valid[i]) for i in range(nNum)] for item in sublist ]).unsqueeze(1)
print(validY.shape)
####
validDS = list(zip(validX, validY))
x,y = validDS[0]
x.shape, y
####

# create DataLoaders.
####
trainDL = DataLoader(trainDS, batch_size=256, shuffle=True)
print('Number of training batches: {}'.format(len(list(trainDL))))
xb,yb = first(trainDL)
print(xb.shape, yb.shape)
####
validDL = DataLoader(validDS, batch_size=256, shuffle=True)
####
dls = DataLoaders(trainDL, validDL)
####

# create a Learner and fit it.
####
dls = ImageDataLoaders.from_folder(path, train='training', valid='testing')
learn = cnn_learner(dls, arch=resnet18, pretrained=False, n_out=10, loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1)
learn.recorder.values[-1][2]
####

# test valid accuracy. plot confussion matrix.
####
####
