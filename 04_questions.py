####
from fastai.vision.all import *
plt.style.use('dark_background')
####

####
def collate_list_string(l, s):
    c = []
    for i in range(len(l)):
        c.append( (l[i], s[i]) )
    return c

a = [1,2,3,4]
b = 'abcd'
collate_list_string(a, b)
####

####
a = torch.arange(1, 10, 1)
print(a)
b = a.view(-1,3)
print(b)
####

####
plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{bm}',
    "font.family": "serif",
    "font.serif": "Palatino",
})
def calc_z(w, x, b): return x*w + b
x = torch.arange(start=-5, end=5, step=0.01)
w=1; b=0

z = calc_z(w, x, b)
plt.plot(x, torch.sigmoid(z), linewidth=2, label="b = {}".format(b))
b=-2
z = calc_z(w, x, b)
plt.plot(x, torch.sigmoid(z), linewidth=2, label="b = {}".format(b))
b=2
z = calc_z(w, x, b)
plt.plot(x, torch.sigmoid(z), linewidth=2, label="b = {}".format(b))

plt.grid(color='#333', linestyle='-.')
plt.legend(loc=0)
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$\sigma(\bar{\bm{z}})=\sigma(\bar{\bar{\bm{w}}}\cdot \bar{\bm{x}} + \bar{\bm{b}})$', fontsize=16)
plt.savefig('b.pdf')
####

####
plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{bm}',
    "font.family": "serif",
    "font.serif": "Palatino",
})
def calc_z(w, x, b): return x*w + b
x = torch.arange(start=-5, end=5, step=0.01)
w=1; b=0

z = calc_z(w, x, b)
plt.plot(x, torch.sigmoid(z), linewidth=2, label="w = {}".format(w))
w=2
z = calc_z(w, x, b)
plt.plot(x, torch.sigmoid(z), linewidth=2, label="w = {}".format(w))
w=4
z = calc_z(w, x, b)
plt.plot(x, torch.sigmoid(z), linewidth=2, label="w = {}".format(w))

plt.grid(color='#333', linestyle='-.')
plt.legend(loc=0)
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$\sigma(\bar{\bm{z}})=\sigma(\bar{\bar{\bm{w}}}\cdot \bar{\bm{x}} + \bar{\bm{b}})$', fontsize=16)
plt.savefig('w.pdf')
####

####
plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{bm}',
    "font.family": "serif",
    "font.serif": "Palatino",
})
x = torch.arange(start=-2, end=2, step=0.01)
y = x.max(tensor(0.0))

plt.plot(x, y, linewidth=2)

plt.grid(color='#333', linestyle='-.')
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'ReLU$(x)$', fontsize=16)
plt.savefig('relu.pdf')
####

# My implementation of Learner based on training loop.
####
path = untar_data(URLs.MNIST_SAMPLE, dest='/media/xar/barracuda1/fast.ai/data/')

train_threes = torch.stack([tensor(Image.open(o)) for o in (path/'train'/'3').ls()]).float()/255
train_sevens = torch.stack([tensor(Image.open(o)) for o in (path/'train'/'7').ls()]).float()/255

valid_threes = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'3').ls()]).float()/255
valid_sevens = torch.stack([tensor(Image.open(o)) for o in (path/'valid'/'7').ls()]).float()/255

train_x = torch.cat([train_threes, train_sevens]).view(-1, 28*28)
train_y = tensor([1]*len(train_threes) + [0]*len(train_sevens)).unsqueeze(1)
train_dset = list(zip(train_x, train_y))

valid_x = torch.cat([valid_threes, valid_sevens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_threes) + [0]*len(valid_sevens)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))

dl = DataLoader(train_dset, batch_size=256, shuffle=True)
valid_dl = DataLoader(valid_dset, batch_size=256, shuffle=True)

dls = DataLoaders(dl, valid_dl)

simple_net = nn.Sequential(
        nn.Linear(28*28, 30),
        nn.ReLU(),              # same thing as F.relu except that this is an instance of a class. 
        nn.Linear(30, 1)
        )

def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1-predictions, predictions).mean()

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean()

learn = Learner(dls, simple_net, loss_func=mnist_loss, opt_func=SGD, metrics=batch_accuracy)

def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()

def train_epoch(model):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        opt.step()
        opt.zero_grad()

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(i, validate_epoch(model))

def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb,yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
####

class MyLearner:
    """ My Implementation of Learner
    dls = DataLoaders
    """
    def __init__(self, dls, model, loss_func=None, opt_func=SGD, metrics=None):
        self.dls = dls
        self.model = model
        self.loss_func = loss_func
        self.opt_func = opt_func
        self.metrics = metrics

    def fit(self, n_epochs, lr):
        self.n_epochs = n_epochs
        self.lr = lr
        for i in self.n_epochs:
            for xb, yb in self.dls:
                preds = self.model(xb)
                loss = self.loss_func(preds, yb)
                loss.backward()
            print(i, validate_epoch(self.model))

