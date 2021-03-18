############
from fastai.vision.all import *
from fastbook import *
# %matplotlib widget
plt.style.use('dark_background')
# from fastprogress.fastprogress import master_bar, progress_bar
############

# L tutorial
############
doc(L)
print('Create short L list:', L(1,2,3))
L(1,2,3)
####
p = L.range(20).shuffle()
p
####
p[0,2,4] # supports indexing with a list or boolean mask
####
p.argwhere(ge(15)) # supports other methods used in numpy.array like argwhere
####
L.range(10) == L(range(10)) # True
####
random.sample(p,3) # L's are sequences so they can be used with methods like random.sample
#### # optimized indexers for arrays, tensors, and DataFrames
arr = np.arange(9).reshape(3,3)
t = L(arr, use_list=None)
print(t[1,2] == arr[[1,2]])
####
df = pd.DataFrame({'a':[1,2,3]})
t = L(df, use_list=None)
print(t[1,2] == L(pd.DataFrame({'a':[2,3]}, index=[1,2]), use_list=None))
#### # can modify with .append(), +, and *
t = L()
print(t == []) # True
t.append(1)
print(t == [1]) # True
t += [3,2]
print(t == [1,3,2]) # True
t = t + [4]
print(t == [1,3,2,4]) # True
t = 5 + t
print(t == [5,1,3,2,4]) # True
t = L(1)*5
print(t == [1,1,1,1,1])
t = t.map(operator.neg)
print(t == [-1]*5)
print(~L([True,False,False]) == L([False,True,True]))
t = L(range(4))
print(zip(t, L(1).cycle()) == zip(range(4),(1,1,1,1)))
############

# Path tutorial 
############
doc(Path)
####
# iterate through all files in directory `p` and pick out directories.
p = Path('.')
dirs = [x for x in p.iterdir() if x.is_dir()]
print(dirs)
####
# glob iterates over subtree `p` to yield all files matching the pattern in glob() ** is recursive.
list(p.glob('**/*.py'))
####
p = Path('/etc')
q = p / 'os-release' # add directories/files to the path with /
print(q)
print(q.resolve()) # make path absolute (resolve symlinks)
####
print(q.exists()) # querty path properties
print(q.is_dir())
####
with q.open() as f:
    print(f.readline())
####
q.readlines()
####
p = Path('.')
p.ls()
############

# Figure 5-3
############
out = tensor([0.02, -2.49, 1.25])
print('Output:', out)
expOut = torch.exp(out)
print('Exp:', expOut, 'Sum:', expOut.sum())
softOut = expOut / expOut.sum()
print('Softmax:', softOut, 'Sum:', softOut.sum())
############

############
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{bm}',
    'font.family': "Palatino"
})
x = torch.arange(start=-2, end=2, step=0.01)
y = torch.log(x)

plt.plot(x, y, linewidth=2)

plt.grid(color='#333', linestyle='-.')
plt.xlabel(r'$x$', fontsize=16)
plt.ylabel(r'$\ln(x)$', fontsize=16)
plt.savefig('log.pdf')
############
############
############
############
############
############
