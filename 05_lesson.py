####
from fastai.vision.all import *
from fastbook import *
# %matplotlib widget
plt.style.use('dark_background')
# from fastprogress.fastprogress import master_bar, progress_bar
####

####
path = untar_data(URLs.PETS, dest='/media/xar/barracuda1/fast.ai/data')
Path.BASE_PATH = path
path.ls()
####

####
(path/'images').ls()
####

# Experimenting with regular expressions.
####
fname = (path/'images').ls()[3]
print('fname:\t\t', fname)
print('fname.name:\t', fname.name)
fnameRegex = re.findall(r'(.+)_\d+\.jpg$', fname.name)
print('regex:\t\t', fnameRegex)
####

####
pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+\.jpg$'), 'name'), # call RegexLabeller using the attribute .name as we did above
                 # the next two lines do Presizing.
                 item_tfms=Resize(460), # resize each item to 460 x 460 by grabbing a random 460x460 square of the source image
                 batch_tfms=aug_transforms(size=224, min_scale=0.75)) # grab a random (possibly warped and rotated) 224x224 crop of the 460x460 image.
# Presizing is done like this because it is a slow process. Because the first step (item_tfms) always creates something that is the same size, the second step can be done on the GPU which is preferrable because the augmentations are slow.
pets.summary(path/'images')
dls = pets.dataloaders(path/'images')
####

####
dls.show_batch(nrows=1, ncols=3)
dls.show_batch(nrows=1, ncols=3, unique=True)
####

# We dont pass in a loss function here, for an image classification task it knows the normal one to pick is Cross Entropy Loss.
####
learn = cnn_learner(dls=dls, arch=resnet34, metrics=error_rate)
learn.fine_tune(2)
learn.loss_func # print loss function name
####

# Now that we have trained a model we can look at ways of diagnosing problems and cleaning up our data by plotting confusion matrices, top losses, and using the widget to clean up the data from top loss candidates.
####
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(16,16), dpi=80)
interp.most_confused(min_val=5)

interp.plot_top_losses(6, nrows=2)

cleaner = ImageClassifierCleaner(learn)
cleaner
####

# Understanding cross entropy loss.
####
x,y = dls.one_batch()
print(dls.vocab)
print(y)
####

# Learning rate finder
####
learn = cnn_learner(dls, resnet34, metrics=error_rate)
lr_min, lr_steep = learn.lr_find()
print('Minimum / 10: {:.2e}, steepest point: {:.2e}'.format(lr_min, lr_steep))
####
learn.fine_tune(1, base_lr=0.1)
learn.recorder.values[-1][2]
####
lr = lr_min
learn.fit_one_cycle(5, lr)
learn.recorder.values[-1][2]
####
learn.lr_find()
####

# Discriminative learning rate
####
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6, 1e-4)) # slice is an object that gives first layer the first number for lr, and the last layer the second number lr, with each layer inbetween as evenly spaced multiples between these two values.
####
learn.recorder.plot_loss()
####
####
####
####
