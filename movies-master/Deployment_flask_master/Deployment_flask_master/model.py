# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import pickle
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_excel(r'C:\Users\Admin\Desktop\Deployment_flask_master\movie_metadata1.xlsx')

dataset=dataset.drop(['num_critic_for_reviews','actor_3_name','country','actor_3_facebook_likes','num_user_for_reviews','budget','duration','content_rating','color','aspect_ratio','plot_keywords','facenumber_in_poster','cast_total_facebook_likes','movie_facebook_likes','movie_imdb_link'],axis=1)

dataset=dataset.dropna(subset=['director_name','director_facebook_likes','actor_1_facebook_likes', 'actor_1_name','actor_2_facebook_likes', 'actor_2_name','gross','language','title_year','imdb_score'])

dataset=dataset[dataset['language']=='English']

q=dataset['num_voted_users'].quantile(0.1)
dataset=dataset[dataset['num_voted_users']>q]

dataset=pd.get_dummies(dataset,columns=['genre1','genre2','genre3','genre4'])
dataset=dataset.reset_index()
dataset['Adventure']=dataset['genre1_Adventure']+dataset['genre2_Adventure']
dataset['Action']=dataset['genre1_Action']
dataset['Animation']=dataset['genre1_Animation']+dataset['genre2_Animation']+dataset['genre3_Animation']
dataset['Comedy']=dataset['genre1_Comedy']+dataset['genre2_Comedy']+dataset['genre3_Comedy']+dataset['genre4_Comedy']
dataset['Crime']=dataset['genre1_Crime']+dataset['genre2_Crime']+dataset['genre3_Crime']+dataset['genre4_Crime']
dataset['Drama']=dataset['genre1_Drama']+dataset['genre2_Drama']+dataset['genre3_Drama']+dataset['genre4_Drama']
dataset['Biography']=dataset['genre1_Biography']+dataset['genre2_Biography']+dataset['genre3_Biography']
dataset['Documentary']=dataset['genre1_Documentary']+dataset['genre2_Documentary']+dataset['genre3_Documentary']
dataset['Fantasy']=dataset['genre1_Fantasy']+dataset['genre2_Fantasy']+dataset['genre3_Fantasy']+dataset['genre4_Fantasy']
dataset['Family']=dataset['genre1_Family']+dataset['genre2_Family']+dataset['genre3_Family']+dataset['genre4_Family']
dataset['Horror']=dataset['genre1_Horror']+dataset['genre2_Horror']+dataset['genre3_Horror']+dataset['genre4_Horror']
dataset['Mystery']=dataset['genre1_Mystery']+dataset['genre2_Mystery']+dataset['genre3_Mystery']+dataset['genre4_Mystery']
dataset['Musical']=dataset['genre1_Musical']+dataset['genre2_Musical']+dataset['genre3_Musical']+dataset['genre4_Musical']
dataset['Music']=dataset['genre1_Music']+dataset['genre2_Music']+dataset['genre3_Music']+dataset['genre4_Music']
dataset['Romance']=dataset['genre1_Romance']+dataset['genre2_Romance']+dataset['genre3_Romance']+dataset['genre4_Romance']
dataset['Sci-Fi']=dataset['genre1_Sci-Fi']+dataset['genre2_Sci-Fi']+dataset['genre3_Sci-Fi']+dataset['genre4_Sci-Fi']
dataset['Thriller']=dataset['genre2_Thriller']+dataset['genre3_Thriller']+dataset['genre4_Thriller']
dataset['History']=dataset['genre2_History']+dataset['genre3_History']+dataset['genre4_History']
dataset['War']=dataset['genre2_War']+dataset['genre3_War']+dataset['genre4_War']
dataset['Western']=dataset['genre1_Western']+dataset['genre2_Western']+dataset['genre3_Western']+dataset['genre4_Western']

dataset=dataset.drop(['genre1_Adventure','genre2_Adventure','genre1_Action','genre1_Animation','genre2_Animation','genre3_Animation','genre1_Comedy','genre2_Comedy','genre3_Comedy','genre4_Comedy','genre1_Crime','genre2_Crime','genre3_Crime','genre4_Crime','genre1_Drama','genre2_Drama','genre3_Drama','genre4_Drama','genre1_Biography','genre2_Biography','genre3_Biography','genre1_Documentary','genre2_Documentary','genre3_Documentary','genre1_Fantasy','genre2_Fantasy','genre3_Fantasy','genre4_Fantasy','genre1_Family','genre2_Family','genre3_Family','genre4_Family','genre1_Horror','genre2_Horror','genre3_Horror','genre4_Horror','genre1_Mystery','genre2_Mystery','genre3_Mystery','genre4_Mystery','genre1_Musical','genre2_Musical','genre3_Musical','genre4_Musical','genre1_Music','genre2_Music','genre3_Music','genre4_Music','genre1_Romance','genre2_Romance','genre3_Romance','genre4_Romance','genre1_Sci-Fi','genre2_Sci-Fi','genre3_Sci-Fi','genre4_Sci-Fi','genre2_Thriller','genre3_Thriller','genre4_Thriller','genre2_War','genre3_War','genre4_War','genre4_History','genre3_History','genre2_History','genre2_Sport','genre1_Western','genre3_News','genre3_Sport','genre4_Sport','genre3_Film-Noir','genre2_Western','genre3_Western','genre4_Western'],axis=1)

data_action=dataset.loc[dataset['Action']==1]
data_action=data_action.drop(['Adventure', 'Drama', 'Animation', 'Comedy', 'Mystery',
       'Crime', 'Biography', 'Fantasy', 'Sci-Fi', 'Horror', 'Documentary',
       'Romance', 'Thriller', 'Family', 'Music', 'Western', 'Musical','History','War'],axis=1)
data_action=data_action.sort_values('director_name')

df = pd.DataFrame(data_action, columns=['director_name', 'imdb_score'])
df=df.groupby('director_name').mean().reset_index()

df=df.rename(columns={'imdb_score':'dir_average'})
data_action=pd.merge(data_action,df,on='director_name')


df = pd.DataFrame(data_action, columns=['actor_1_name', 'imdb_score'])
df=df.groupby('actor_1_name').mean().reset_index()

df=df.rename(columns={'imdb_score':'act1_average'})
data_action=pd.merge(data_action,df,on='actor_1_name')


df = pd.DataFrame(data_action, columns=['actor_2_name', 'imdb_score'])
df=df.groupby('actor_2_name').mean().reset_index()

df=df.rename(columns={'imdb_score':'act2_average'})
data_action=pd.merge(data_action,df,on='actor_2_name')


corr = data_action.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
        

dir_rating=data_action['dir_average']
actor1_rating=data_action['act1_average']
actor2_rating=data_action['act2_average']
imdb_score=data_action['imdb_score']


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dir_rating, actor1_rating,imdb_score, color='#ef1234')
# plt.legend()
plt.show()


x=data_action.iloc[:, -3:]
y=data_action.iloc[:, -5]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.linear_model import LinearRegression
mlr_model= LinearRegression(fit_intercept=True)

mlr_model.fit(x_train,y_train)
        

pickle.dump(mlr_model, open('model1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model1.pkl','rb'))
print(model.predict([[2, 9, 6]]))

print(mlr_model.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(mlr_model.coef_)#y=c+mx

print(mlr_model.score(x_train,y_train))

y_hat_test=mlr_model.predict(x_test)
df_pf=pd.DataFrame(y_hat_test,columns=['Predictions'])
df_pf['Target']=y_test.values
df_pf.head()



dir_dict=dict(zip(data_action.director_name,data_action.dir_average))

actor1_dict=dict(zip(data_action.actor_1_name,data_action.act1_average))

actor2_dict=dict(zip(data_action.actor_2_name,data_action.act2_average))








