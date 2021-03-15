####
from fastai.vision.all import *
from fastbook import *
# %matplotlib widget
plt.style.use('dark_background')
# from fastprogress.fastprogress import master_bar, progress_bar
####

####
path = untar_data(URLs.PETS, dest='/media/xar/barracuda1/fast.ai/data')
path.ls()
####

