####
from fastai.vision.all import *
plt.style.use('dark_background')
####

# Download MNIST_SAMPLE dataset which is just 3's and 7's.
####
path = untar_data(URLs.MNIST_SAMPLE, dest='/media/xar/barracuda1/fast.ai/data/')
path.ls()
print((path/'train').ls())
print((path/'valid').ls())
####

# Use list comprehension to create a list of image tensors for training 3's and 7's. Then turn the list dimension into a tensor dimension with torch.stack(). Then convert each element into a float between 0 and 1.
####
train_threes = torch.stack([tensor(Image.open(o)) for o in (path/'train'/'3').ls()]).float()/255
train_sevens = torch.stack([tensor(Image.open(o)) for o in (path/'train'/'7').ls()]).float()/255
print(train_threes.shape, train_sevens.shape)
show_images([train_threes[0], train_sevens[0]])
####

# Do the same for the validation set of 3's and 7's. 
####
valid_threes = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()]).float()/255
valid_sevens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()]).float()/255
print(valid_threes.shape, valid_sevens.shape)
show_images([valid_threes[0], valid_sevens[0]])
####

# Concatenate the training 3's and 7's into one training tensor. Also, flatten each training image into a rank-1 tensor with .view(). -1 in .view() means to infer this dimensions size from other dimensions.
####
train_x = torch.cat([train_threes, train_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(train_threes) + [0]*len(train_sevens)).unsqueeze(1)
train_x.shape, train_y.shape
####

# Create a training dataset, which means a 'list' that returns tuples as its elements.
####
train_dset = list(zip(train_x, train_y))
x,y = train_dset[0]
print(type(train_dset))
x.shape, y
####

# Concatenate the validation set and create a validation dataset.
####
valid_x = torch.cat([valid_threes, valid_sevens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_threes) + [0]*len(valid_sevens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))
####

# Initialize parameters (weights and bias).
####
def init_params(size, var=1.0):
    return (torch.randn(size)*var).requires_grad_()

weights = init_params((28*28, 1))
bias = init_params(1)
weights.shape, bias.shape
####

# Plot histrogram of the initialized weights.
####
plt.hist(weights.detach().numpy(), bins=50)
plt.xlabel('Weight Values')
plt.ylabel('Counts')
plt.show()
####

# Calculate prediction for first training image. And define a subroutine that does this operation on our x batch, xb.
####
(train_x[0]*weights.T).sum() + bias
def linear1(xb):
    return xb@weights + bias
preds = linear1(train_x)
preds
####

# Convert the predictions into booleans from floats and print out just the value .item() of the factor of correct predictions.
####
corrects = (preds > 0.5).float() == train_y
corrects, corrects.float().mean().item()
####

# Create a loss function for our dataset. Also, plot a Sigmoid function.
####
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1-predictions, predictions).mean()

t1 = torch.arange(start=-4, end=4, step=0.01)
plt.plot(t1, torch.sigmoid(t1))
####

# Experiment with DataLoaders. They take in list or tensor etc. and can output shuffled batches of that data.
####
coll = range(15)
dl = DataLoader(coll, batch_size=5, shuffle=True)
list(dl)
####

# DataLoaders also work with datasets, (lists of tuples). So create a list of tuples.
####
ds = L(enumerate(string.ascii_lowercase))
ds
####

# Try the DataLoader on the dataset (list of tuples).
####
dl = DataLoader(ds, batch_size=6, shuffle=True)
list(dl)
####

# Putting it all together. Reinitialize parameters.
####
weights = init_params((28*28, 1))
bias = init_params(1)
####

# Create a DataLoader from the training and validation datasets.
####
dl = DataLoader(train_dset, batch_size=256, shuffle=True)
xb,yb = first(dl)
xb.shape, yb.shape

valid_dl = DataLoader(valid_dset, batch_size=256, shuffle=True)
####

# Test on small mini-batch.
####
batch = train_x[:4]
print(batch.shape)

# Calculate predicitions.
preds = linear1(batch)
print(preds)

# Calculate loss.
loss = mnist_loss(preds, train_y[:4])
print(loss)

# Calculate the gradients, .backward() calculates the grads and adds the grads to the parameters.
loss.backward()
print(weights.grad.shape, weights.grad.mean(), bias.grad)
####

# Put that all into a function.
####
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()
####

# And test it. Thre gradients change if we run more than once. This is because .backward() adds the grads to the params.
####
calc_grad(xb, yb, linear1)
print(weights.grad.mean(), bias.grad)
####

# So zero the grads (in place, .zero_()) first.
####
weights.grad.zero_()
bias.grad.zero_()
calc_grad(xb, yb, linear1)
print(weights.grad.mean(), bias.grad)
####

# Make a training epoch subroutine.
####
def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr   # .data means that PyTorch wont take the gradient of this step.
            p.grad.zero_()  # zero the gradient in place. The trailing '_' means *in place* operation.
####

# Test an accuracy calculation.
####
(preds>0.0).float() == train_y[:4]
####

# Define an accuracy subroutine based off of the previous cell.
####
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()
####

# Check that it works.
####
batch_accuracy(xb, yb)
####

# Calculate accuracy on validation set.
####
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
####

# Validate after one epoch.
####
validate_epoch(linear1)
####

# Train for 1 epoch to see if it imporves.
####
lr = 1.
params = weights, bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)
####

# And do a few more.
####
for i in range(20):
    train_epoch(linear1, lr, params)
    print(i+1, validate_epoch(linear1))
####

# Create optimizer to handle the SGD step for us.
####
linear_model = nn.Linear(28*28, 1)
w,b = linear_model.parameters()
print(w.shape, b.shape)
####

# Create an optimizer class.
####
class BasicOptim:
    def __init__(self, params, lr):
        self.params, self.lr = list(params), lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None
####

# Create an optimizer instance.
####
opt = BasicOptim(linear_model.parameters(), lr)
####

# We can simplify the training loop now.
####
def train_epoch(model):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()
####

# Make a training loop function.
####
def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(linear_model)
        print(validate_epoch(model), end=' ')
####

# Test it.
####
train_model(linear_model, 20)
####

# The fastai SGD class does the same thing as our BasicOptim.
####
linear_model = nn.Linear(28*28, 1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
####

# Learner.fit replaces our train_model(). We need DataLoaders to create a learner.
####
dls = DataLoaders(dl, valid_dl)
####

# Create a Learner and fit it.
####
learn = Learner(dls, nn.Linear(28*28, 1), loss_func=mnist_loss, opt_func=SGD, metrics=batch_accuracy)
learn.fit(10, lr=0.001)
####


# Adding a nonlinearity.
####
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0)) # return max of each element of res or 0, ie ReLU.
    res = res@w2 + b2
    return res
####

# Initialize the parameters for this NN. Notice the columns/outputs have to match the rows/inputs of the next layer.
####
w1 = init_params(size=(28*28, 30))
b1 = init_params(size=30)
w2 = init_params(size=(30, 1))
b2 = init_params(size=1)
print(w1.shape, b1.shape, w2.shape, b2.shape)
####

# Plot a ReLU function from F.
####
t1 = torch.arange(start=-4, end=4, step=0.01)
plt.plot(t1, F.relu(t1))
####

# Replace simple_net() with PyTorch code. nn.Sequential executes each listed layer/function in order.
####
simple_net = nn.Sequential(
        nn.Linear(28*28, 30),
        nn.ReLU(),              # same thing as F.relu except that this is an instance of a class. 
        nn.Linear(30, 1)
        )
####

# Since this is a deeper model we'll use a lower learning rate and more epochs.
####
validate_epoch(simple_net)
learn = Learner(dls, simple_net, loss_func=mnist_loss, opt_func=SGD, metrics=batch_accuracy)
learn.fit(40, lr=0.1)
####

# Plot accuracy results and view final accuracy.
####
plt.plot(L(learn.recorder.values).itemgot(2))
learn.recorder.values[-1][2]
####

# Going Deeper. Train a deeper NN, (18 layers).
####
dls = ImageDataLoaders.from_folder(path)
learn = cnn_learner(dls, resnet18, pretrained=False, loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(1, lr=0.1)
learn.recorder.values[-1][2]
####

