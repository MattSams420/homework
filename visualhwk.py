from sklearn.datasets import load_iris
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import preprocessing

data = sb.load_dataset("iris")

def printproblem(i):
	print('problem %d' %i )
	i+=1
	return i

i = 1
i = printproblem(i)

'''data = sb.load_dataset("iris") #load iris data
data.describe().plot(kind = "area",fontsize=16, figsize = (15,8), table = True, colormap="Accent")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("General Statistics of Iris Dataset")
plt.show()'''

i = printproblem(i)
'''ax=plt.subplots(1,1,figsize=(10,8))
sb.countplot('species',data=data)
plt.title("Iris Species Count")
plt.show()'''

i = printproblem(i)
'''ax=plt.subplots(1,1,figsize=(10,8))
data['species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
plt.title("Iris Species %")
plt.show()'''

i = printproblem(i)
#print(data.head())
'''fig = data[data.species=='setosa'].plot(kind='scatter',x='sepal_length',y='sepal_width',color='orange', label='setosa')
data[data.species=='versicolor'].plot(kind='scatter',x='sepal_length',y='sepal_width',color='blue', label='versicolor',ax=fig)
data[data.species=='virginica'].plot(kind='scatter',x='sepal_length',y='sepal_width',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()'''

i = printproblem(i)

'''fig = data[data.species=='setosa'].plot.scatter(x='petal_length',y='petal_width',color='orange', label='setosa')
data[data.species=='versicolor'].plot.scatter(x='petal_length',y='petal_width',color='blue', label='versicolor',ax=fig)
data[data.species=='virginica'].plot.scatter(x='petal_length',y='petal_width',color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()'''

i = printproblem(i)

'''data.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,12)
plt.show()'''

i = printproblem(i)

'''fig=sb.jointplot(x='sepal_length', y='sepal_width', data=data, color='blue') 
plt.show()'''

i = printproblem(i)

'''fig=sb.jointplot(x='sepal_length', y='sepal_width', kind="hex", color="red", data=data)
plt.show()'''

i = printproblem(i)

'''fig=sb.jointplot(x='sepal_length', y='sepal_width', kind="kde", color='cyan', data=data)  
plt.show()'''

i = printproblem(i)

'''fig=sb.jointplot(x='sepal_length', y='sepal_width', kind="reg", color='red', data=data) 
plt.show()'''

i = printproblem(i)

'''sb.jointplot("sepal_length", "sepal_width", data=data, color="b").plot_joint(sb.kdeplot, zorder=0, n_levels=6) 
plt.show()'''

i = printproblem(i)

'''g = sb.jointplot(x="sepal_length", y="sepal_width", data=data, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Sepal Length(cm)$", "$Sepal Width(cm)$") 
plt.show()'''

i = printproblem(i)

'''g = sb.jointplot(x="sepal_length", y="sepal_width", data=data, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=40, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$Sepal Length(cm)$", "$Sepal Width(cm)$") 
plt.show()'''

i = printproblem(i)

'''sub=data[data['species']=='setosa']
sb.kdeplot(data=sub[['sepal_length','sepal_width']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('Iris-setosa')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()'''

i = printproblem(i)

'''sub=data[data['species']=='setosa']
sb.kdeplot(data=sub[['petal_length','petal_width']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('setosa')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()'''

i = printproblem(i)

'''sub=data[data['species']=='setosa']
sb.kdeplot(data=sub[['petal_length','petal_width']],cmap="plasma", shade=True, shade_lowest=False)
plt.title('setosa')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()'''

i = printproblem(i)

'''x = data.iloc[:, 0:4]
f, ax = plt.subplots(figsize=(10, 8))
corr = x.corr()
print(corr)
sb.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
          cmap=sb.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, linewidths=.5)
plt.show() '''

i = printproblem(i)

'''box_data = data #variable representing the data array
box_target = data.species #variable representing the labels array
sb.boxplot(data = box_data,width=0.5,fliersize=5)
sb.set(rc={'figure.figsize':(2,15)})
plt.show()'''

i = printproblem(i)

'''le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
data.species = le.fit_transform(data.species)
#Drop id column
x = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

fig = plt.figure(1, figsize=(7, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(x)
X = pca.transform(x)
for name, label in [('setosa', 0), ('versicolour', 1), ('virginica', 2)]:
    ax.text3D(x[y == label, 0].mean(),
              x[y == label, 1].mean() + 1.5,
              x[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()'''



