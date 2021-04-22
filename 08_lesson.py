###############################################################################
from fastai.collab import *
from fastai.tabular.all import *
###############################################################################
path = untar_data(URLs.ML_100k, dest='/media/xar/barracuda1/fast.ai/data')
###############################################################################
ratings = pd.read_csv(path/'u.data', delimiter='\t', header=None,

names=['user','movie','rating','timestamp'])

ratings.head()
###############################################################################
movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1',

usecols=(0,1), names=('movie','title'), header=None)

movies.head()
###############################################################################
ratings = ratings.merge(movies)
ratings.head()
###############################################################################
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
dls.show_batch()
###############################################################################
n_users = len(dls.classes['user'])

n_movies = len(dls.classes['title'])

n_factors = 5
###############################################################################

# Create our own Embedding module
###############################################################################
class T(Module): # module subclass
    def __init__(self):
        self.a = torch.ones(3)

L(T().parameters()) # nothing is returned which means nothing will be learned. But we want self.a to be learned.
###############################################################################
class T(Module): # module subclass
    def __init__(self):
        self.a = nn.Parameter(torch.ones(3)) # Wrap all parameters that should be optimized in learning in nn.Parameter.

L(T().parameters()) # now we get our vector returned.
###############################################################################

# Create our own Embedding subroutine
###############################################################################
def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01)) # `*size` turns the variable size into a tuple
###############################################################################
class DotProductBias(Module):
    def __init__(self, nUsers, nMovies, nFactors, yRange=(0,5.5)):
        self.userFactors = create_params([nUsers, nFactors])
        self.userBias = create_params([nUsers])
        self.movieFactors = create_params([nMovies, nFactors])
        self.movieBias = create_params([nMovies])
        self.yRange = yRange

    def forward(self, x): # x is the movie ids and the user ids as two columns.
        users = self.userFactors[x[:,0]]
        movies = self.movieFactors[x[:,1]]
        res = (users*movies).sum(dim=1)
        res += self.userBias[x[:,0]] + self.movieBias[x[:,1]]
        return sigmoid_range(res, *self.yRange)
###############################################################################
model = DotProductBias(n_users, n_movies, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
