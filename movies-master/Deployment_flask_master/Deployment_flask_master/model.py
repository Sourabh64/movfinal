# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()
import pickle
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_excel(r'C:\Users\Mr.Kulakarni\Desktop\Final_Year_Project\Deployment_flask_master\Deployment_flask_master\movie_metadata1.xlsx')

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








data_adventure=dataset.loc[dataset['Adventure']==1]
data_adventure=data_adventure.drop(['Action', 'Drama', 'Animation', 'Comedy', 'Mystery',
       'Crime', 'Biography', 'Fantasy', 'Sci-Fi', 'Horror', 'Documentary',
       'Romance', 'Thriller', 'Family', 'Music', 'Western', 'Musical','History','War'],axis=1)
data_adventure=data_adventure.sort_values('director_name')

df_adv = pd.DataFrame(data_adventure, columns=['director_name', 'imdb_score'])
df_adv=df_adv.groupby('director_name').mean().reset_index()

df_adv=df_adv.rename(columns={'imdb_score':'dir_average'})
data_adventure=pd.merge(data_adventure,df_adv,on='director_name')


df_adv = pd.DataFrame(data_adventure, columns=['actor_1_name', 'imdb_score'])
df_adv=df_adv.groupby('actor_1_name').mean().reset_index()

df_adv=df_adv.rename(columns={'imdb_score':'act1_average'})
data_adventure=pd.merge(data_adventure,df_adv,on='actor_1_name')


df_adv = pd.DataFrame(data_adventure, columns=['actor_2_name', 'imdb_score'])
df_adv=df_adv.groupby('actor_2_name').mean().reset_index()

df_adv=df_adv.rename(columns={'imdb_score':'act2_average'})
data_adventure=pd.merge(data_adventure,df_adv,on='actor_2_name')
        

dir_rating=data_adventure['dir_average']
actor1_rating=data_adventure['act1_average']
actor2_rating=data_adventure['act2_average']
imdb_score=data_adventure['imdb_score']


x_adv=data_adventure.iloc[:, -3:]
y_adv=data_adventure.iloc[:, -5]

from sklearn.model_selection import train_test_split
x_adv_train,x_adv_test,y_adv_train,y_adv_test=train_test_split(x_adv,y_adv,test_size=0.2,random_state=100)

from sklearn.linear_model import LinearRegression
mlr_model_adv= LinearRegression(fit_intercept=True)

mlr_model_adv.fit(x_adv_train,y_adv_train)
        

pickle.dump(mlr_model_adv, open('model2.pkl','wb'))

# Loading model to compare the results
model_adv = pickle.load(open('model2.pkl','rb'))
print(model_adv.predict([[2, 9, 6]]))

print(mlr_model_adv.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(mlr_model_adv.coef_)#y=c+mx

print(mlr_model_adv.score(x_adv_train,y_adv_train))

y_hat_test_adv=mlr_model_adv.predict(x_adv_test)
df_pf_adv=pd.DataFrame(y_hat_test_adv,columns=['Predictions'])
df_pf_adv['Target']=y_adv_test.values
df_pf_adv.head()



dir_adv_dict=dict(zip(data_adventure.director_name,data_adventure.dir_average))

actor1_adv_dict=dict(zip(data_adventure.actor_1_name,data_adventure.act1_average))

actor2_adv_dict=dict(zip(data_adventure.actor_2_name,data_adventure.act2_average))





data_comedy=dataset.loc[dataset['Comedy']==1]
data_comedy=data_comedy.drop(['Action', 'Drama', 'Animation', 'Adventure', 'Mystery',
       'Crime', 'Biography', 'Fantasy', 'Sci-Fi', 'Horror', 'Documentary',
       'Romance', 'Thriller', 'Family', 'Music', 'Western', 'Musical','History','War'],axis=1)
data_comedy=data_comedy.sort_values('director_name')

df_com = pd.DataFrame(data_comedy, columns=['director_name', 'imdb_score'])
df_com=df_com.groupby('director_name').mean().reset_index()

df_com=df_com.rename(columns={'imdb_score':'dir_average'})
data_comedy=pd.merge(data_comedy,df_com,on='director_name')


df_com = pd.DataFrame(data_comedy, columns=['actor_1_name', 'imdb_score'])
df_com=df_com.groupby('actor_1_name').mean().reset_index()

df_com=df_com.rename(columns={'imdb_score':'act1_average'})
data_comedy=pd.merge(data_comedy,df_com,on='actor_1_name')


df_com = pd.DataFrame(data_comedy, columns=['actor_2_name', 'imdb_score'])
df_com=df_com.groupby('actor_2_name').mean().reset_index()

df_com=df_com.rename(columns={'imdb_score':'act2_average'})
data_comedy=pd.merge(data_comedy,df_com,on='actor_2_name')


dir_rating=data_comedy['dir_average']
actor1_rating=data_comedy['act1_average']
actor2_rating=data_comedy['act2_average']
imdb_score=data_comedy['imdb_score']


x_com=data_comedy.iloc[:, -3:]
y_com=data_comedy.iloc[:, -5]

from sklearn.model_selection import train_test_split
x_com_train,x_com_test,y_com_train,y_com_test=train_test_split(x_com,y_com,test_size=0.2,random_state=100)

from sklearn.linear_model import LinearRegression
mlr_model_com= LinearRegression(fit_intercept=True)

mlr_model_com.fit(x_com_train,y_com_train)
        

pickle.dump(mlr_model_com, open('model3.pkl','wb'))

# Loading model to compare the results
model_com = pickle.load(open('model3.pkl','rb'))
print(model_com.predict([[2, 9, 6]]))

print(mlr_model_com.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(mlr_model_com.coef_)#y=c+mx

print(mlr_model_com.score(x_adv_train,y_adv_train))

y_hat_test_com=mlr_model_com.predict(x_com_test)
df_pf_com=pd.DataFrame(y_hat_test_com,columns=['Predictions'])
df_pf_com['Target']=y_com_test.values
df_pf_com.head()



dir_com_dict=dict(zip(data_comedy.director_name,data_comedy.dir_average))

actor1_com_dict=dict(zip(data_comedy.actor_1_name,data_comedy.act1_average))

actor2_com_dict=dict(zip(data_comedy.actor_2_name,data_comedy.act2_average))





data_romance=dataset.loc[dataset['Romance']==1]
data_romance=data_romance.drop(['Action', 'Drama', 'Animation', 'Adventure', 'Mystery',
       'Crime', 'Biography', 'Fantasy', 'Sci-Fi', 'Horror', 'Documentary',
       'Comedy', 'Thriller', 'Family', 'Music', 'Western', 'Musical','History','War'],axis=1)
data_romance=data_romance.sort_values('director_name')

df_rom = pd.DataFrame(data_romance, columns=['director_name', 'imdb_score'])
df_rom=df_rom.groupby('director_name').mean().reset_index()

df_rom=df_rom.rename(columns={'imdb_score':'dir_average'})
data_romance=pd.merge(data_romance,df_rom,on='director_name')


df_rom = pd.DataFrame(data_romance, columns=['actor_1_name', 'imdb_score'])
df_rom=df_rom.groupby('actor_1_name').mean().reset_index()

df_rom=df_rom.rename(columns={'imdb_score':'act1_average'})
data_romance=pd.merge(data_romance,df_rom,on='actor_1_name')


df_rom = pd.DataFrame(data_romance, columns=['actor_2_name', 'imdb_score'])
df_rom=df_rom.groupby('actor_2_name').mean().reset_index()

df_rom=df_rom.rename(columns={'imdb_score':'act2_average'})
data_romance=pd.merge(data_romance,df_rom,on='actor_2_name')


dir_rating=data_romance['dir_average']
actor1_rating=data_romance['act1_average']
actor2_rating=data_romance['act2_average']
imdb_score=data_romance['imdb_score']


x_rom=data_romance.iloc[:, -3:]
y_rom=data_romance.iloc[:, -5]

from sklearn.model_selection import train_test_split
x_rom_train,x_rom_test,y_rom_train,y_rom_test=train_test_split(x_rom,y_rom,test_size=0.2,random_state=100)

from sklearn.linear_model import LinearRegression
mlr_model_rom= LinearRegression(fit_intercept=True)

mlr_model_rom.fit(x_rom_train,y_rom_train)
        

pickle.dump(mlr_model_com, open('model4.pkl','wb'))

# Loading model to compare the results
model_rom = pickle.load(open('model4.pkl','rb'))
print(model_rom.predict([[2, 9, 6]]))

print(mlr_model_rom.intercept_) # (PRICE=(-4481.80028058845)+8.65903854)*AREA
print(mlr_model_rom.coef_)#y=c+mx

print(mlr_model_rom.score(x_adv_train,y_adv_train))

y_hat_test_rom=mlr_model_rom.predict(x_rom_test)
df_pf_rom=pd.DataFrame(y_hat_test_rom,columns=['Predictions'])
df_pf_rom['Target']=y_rom_test.values
df_pf_rom.head()



dir_rom_dict=dict(zip(data_romance.director_name,data_romance.dir_average))

actor1_rom_dict=dict(zip(data_romance.actor_1_name,data_romance.act1_average))

actor2_rom_dict=dict(zip(data_romance.actor_2_name,data_romance.act2_average))