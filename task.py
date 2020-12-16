#!/usr/bin/env python
# coding: utf-8

# ###  AUTHOR : TWINKLE RAWAT
# 
#  
# 
# 

# ###  TASK Exploratory Data Analysis - Sports
# 

# ####  IMPORTING LIBRARIES

# In[39]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[40]:


pip install GitPython


# #### IMPORTING DATASET

# In[3]:


df = pd.read_csv("C:\\Users\\user\\Downloads\\Indian Premier League\\matches.csv")
df1 = pd.read_csv("C:\\Users\\user\\Downloads\\Indian Premier League\\deliveries.csv")


# In[ ]:


df.head()


# In[ ]:


df1.head()


# #### Seeking information of Dataset

# In[4]:


df.info()
#attributes that will be used for further analysis


# id - id of the match
# 
# season - season of ipl
# 
# city - the city at which the match is held
# 
# date - date
# 
# team1 & team2 - names of the teams
# 
# toss_winner - winner of the toss
# 
# toss_decision - bat/bowl
# 
# result - normal/tie/no result
# 
# dl_applied - due to any reasons dl method is applied?(binary value 0 or 1)
# 
# win_by_runs - win by runs
# 
# win_by_wickets - win by wickets
# 
# player_of_match - man of the match
# 
# venue - stadium
# 
# umpire1,2,3 - names of umpires

# In[5]:


df1.info()


# 
# match_id - id of match
# 
# inning - 1st or 2nd
# 
# batting_team - batting team
# 
# bowling_team - bowling team
# 
# over - over in the innings
# 
# ball - ball number in the over
# 
# batsman - name of batsman on strike
# 
# non_striker - name of batsman at non-striker end
# 
# bowler - name of bowler
# 
# vis_super_over - binary value 0 or 1(yes/no)
# 
# wide_runs - how many runs consided by wide
# 
# bye_runs - bye runs
# 
# legbye_runs - leg bye runs
# 
# noball_runs - no ball runs
# 
# penalty_runs - penalty runs
# 
# batsman_runs - batsman runs
# 
# total_runs - total runs for the ball
# 
# player_dismissed - name of the batsman dismissed
# 
# dismissal_kind - how the bastman is dismissed(LBW/Hitwicket/runout ..etc)
# 
# fielder - named of the fielder who catched the ball
# 
# 

# #### Details on Toss won by each team, Total Matches played so far, total matches being won list.

# In[6]:



team_stats = pd.DataFrame({'Total Matches played': df.team1.value_counts() + df.team2.value_counts(), 'Total won': df.winner.value_counts(), 'Toss won': df.toss_winner.value_counts(), 
                          'Total lost': ((df.team1.value_counts() + df.team2.value_counts()) - df.winner.value_counts())})
team_stats = team_stats.reset_index()
team_stats.rename(columns = {'index':'Teams'}, inplace = True)
winloss = team_stats['Total won'] / team_stats['Total Matches played']
winloss = pd.DataFrame({'Winloss Ratio': team_stats['Total won'] / team_stats['Total Matches played']})
winloss= winloss.round(2)
team_stats = team_stats.join(winloss)
team_stats


# #### Maximum Toss Won:

# In[7]:




plt.subplots(figsize=(10,7))
ax=df['toss_winner'].value_counts().plot.barh(width=0.8,color=['red', 'green','blue','pink','orange'])
plt.title("Maximum Toss Won")


# 
# As you know in cricket toss plays a mojor role , the team which wins the toss has a heigher advantage.
# mumbai indians has won maximum no.of toss in IPL

# In[8]:


Tosswin_matchwin=df[df['toss_winner']==df['winner']]
slices=[len(Tosswin_matchwin),(len(df)-len(Tosswin_matchwin))]
labels=['Yes','No']
plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%',colors=['y','g'])
plt.title("Teams who had won Toss and Won the match")
fig = plt.gcf()
fig.set_size_inches(5,5)
plt.show()
#The Chances of the team winning, if it has won the toss are reasonably high.
#Toss favours to the victory of team


# #### Plotting the above data Analysis - Total Won

# In[9]:


Total_win = df.winner.value_counts()
ax = Total_win.plot(kind='bar',title ="Overall wins by each team ",figsize=(10,5), fontsize=12,color='green')
ax.set_xlabel("Teams in IPL",fontsize=12)
ax.set_ylabel("Wins",fontsize=12)


# #### Seeking shape as in rows and columns in our dataset

# In[10]:


df.shape


# In[11]:


df1.describe()


# #### Seeking Description of our dataset

# In[12]:


df.describe()


# #### the no. of matches in the dataset

# In[13]:


df['id'].max()


# In[14]:


df1['match_id'].max()


# #### Seasons represented in the dataset

# In[15]:


df['season'].unique()


# In[16]:


len(df['season'].unique())


# In[17]:


winner_each_season = df.drop_duplicates(subset=['season'],keep='last')[['season', 'winner']].reset_index(drop=True)
winner_each_season


# #### Most wins 

# In[18]:



winner_each_season.winner.value_counts().plot(kind='bar',orientation='vertical',title='Most wins',color='cyan')


# In[19]:


data = winner_each_season.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h')


# chennai won 3 times and mumbai indian 4 times

# #### Team detail that won by maxium runs

# In[20]:


df.iloc[df['win_by_runs'].idxmax()]


# #### Each team comparison

# In[21]:


for i in df["team1"].unique():
    #print(i)
    new_df=df.copy()
    for j in range(len(df)):
        if (new_df["team1"][j] != i and new_df["team2"][j] == i):
            a=new_df["team1"][j]
            new_df["team1"][j]=new_df["team2"][j]
            new_df["team2"][j]=a
        elif(new_df["team1"][j] != i and new_df["team2"][j] != i):
            new_df=new_df.drop([j],axis=0)
    new_df["won"]=(new_df["winner"]==i).astype(int)
    #new_df.to_csv(i+'.csv')
    won=pd.DataFrame({'Total':new_df.team2.value_counts(),'won': new_df.winner.value_counts()})
    won=won.drop(index=i)
    fig,ax=plt.subplots(figsize=(10,10))
    bar_width = 0.3
    opacity = 0.9
    index=np.arange(len(won))
    ax.bar(index,won['Total'],bar_width, alpha=opacity,color='g',label='total matches')
    ax.bar(index+bar_width,won['won'],bar_width, alpha=opacity,color='c',label='lost')
    ax.bar(index+2*bar_width,won['Total']-won['won'],bar_width, alpha=opacity,color='y',label='won')
    ax.legend()
    ax.set_xlabel('Teams')
    ax.set_ylabel('No.of matches')
    ax.set_title(i+' v/s Other teams')
    plt.xticks(ticks=index,labels=won.index,rotation=90)
    plt.show()


# ####  From the Above Mumbai Indians and Chennai Super Kings are showing complete domination in this tournament.

# #### Team winning by maxium runs name

# In[22]:


df.iloc[df['win_by_runs'].idxmax()]['winner']


# #### Team winning by maxium wickets name

# In[23]:


df.iloc[df['win_by_wickets'].idxmax()]['winner']


# #### Team winning by minimum runs name

# In[24]:


df.iloc[df[df['win_by_runs'].ge(1)].win_by_runs.idxmin()]['winner']


# #### Team winning by maxium wickets name

# In[25]:


df.iloc[df[df['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]


# #### Team winning by minimum wickets name

# In[26]:



df.iloc[df[df['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]['winner']


# #### Observation :
# Mumbai Indians is the team which won by maximum and minimum runs
# Kolkata Knight Riders is the team which won by maximum and minimum wickets

# #### Season having most number of matches

# In[27]:


sns.countplot(x='season', data=df)
plt.show()


# #### TOP player

# In[28]:


top_players = df.player_of_match.value_counts()[:10]

fig, ax = plt.subplots()
ax.set_ylim([0,30])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
top_players.plot.bar()
sns.barplot(x = top_players.index, y = top_players, orient='v', palette="Greens");
plt.show()


# #### Batsmen overview
# 

# In[29]:


batsmen = df1.groupby("batsman").agg({'ball': 'count','batsman_runs': 'sum'})
batsmen.rename(columns={'ball':'balls', 'batsman_runs': 'runs'}, inplace=True)
batsmen = batsmen_summary.sort_values(['balls','runs'], ascending=False)
batsmen['batting_strike_rate'] = batsmen['runs']/batsmen['balls'] * 100
batsmen['batting_strike_rate'] = batsmen['batting_strike_rate'].round(2)
batsmen.head(10)


# In[30]:


#utility function used later
def trybuild(lookuplist, buildlist):
    alist = []
    for i in buildlist.index:
        try:
            #print(i)
            alist.append(lookuplist[i])
            #print(alist)
        except KeyError:
            #print('except')
            alist.append(0)
    return alist


# In[31]:


TopBatsman = batsmen.sort_values(['balls','runs'], ascending=False)[:20]
TopBatsman


# In[32]:


alist = []
for r in df1.batsman_runs.unique():
    lookuplist = df1[df1.batsman_runs == r].groupby('batsman')['batsman'].count()
    batsmen[str(r) + 's'] = trybuild(lookuplist, batsmen)
    try:
        alist.append(lookuplist[r])
    except KeyError:
        alist.append(0)
TopBatsman = batsmen.sort_values(['balls','runs'], ascending=False)[:20]
TopBatsman.head(10)


# In[ ]:


#Build a dictionary of Matches player by each batsman
played = {}
def BuildPlayedDict(x):
    #print(x.shape, x.shape[0], x.shape[1])
    for p in x.batsman.unique():
        if p in played:
            played[p] += 1
        else:
            played[p] = 1

df1.groupby('match_id').apply(BuildPlayedDict)
import operator


# In[33]:


TopBatsman['matches_played'] = [played[p] for p in TopBatsman.index]
TopBatsman['average']= TopBatsman['runs']/TopBatsman['matches_played']

TopBatsman['6s/match'] = TopBatsman['6s']/TopBatsman['matches_played']  
TopBatsman['6s/match'].median()

TopBatsman['4s/match'] = TopBatsman['4s']/TopBatsman['matches_played']  
TopBatsman['4s/match']
TopBatsman


# #### Total runs by each batsmen

# In[34]:


plt.figure(figsize=(10,8))
plt.bar(np.arange(len(TopBatsman)),TopBatsman['runs'],color='g')
plt.xticks(ticks=np.arange(len(TopBatsman)),labels=TopBatsman.index,rotation=90)
plt.xlabel('Batsmen')
plt.ylabel('Runs')
plt.title('Total Runs')
plt.show()


# #### each batsmen strike rate

# In[35]:


plt.figure(figsize=(10,8))
plt.bar(np.arange(len(TopBatsman)),TopBatsman['batting_strike_rate'])
plt.xticks(ticks=np.arange(len(TopBatsman)),labels=TopBatsman.index,rotation=90)
plt.xlabel('Batsmen')
plt.ylabel('Strike Rate')
plt.title('Batsmen Strike Rate')
plt.show()


# It is an  important factor for a batsman in an T20 league to maintain a good strike rate.
# AB de Villiers and CH Gayle have almost equal strike rates.

# ##### bowler information
# 

# In[36]:


bowler_wickets = df1.groupby('bowler').aggregate({'ball': 'count', 'total_runs': 'sum', 'player_dismissed' : 'count'})
bowler_wickets.columns = ['runs','balls','wickets']
TopBowlers = bowler_wickets.sort_values(['wickets'], ascending=False)[:20]
TopBowlers


# In[37]:


TopBowlers['economy'] = TopBowlers['runs']/(TopBowlers['balls']/6)
TopBowlers = TopBowlers.sort_values(['economy'], ascending=True)[:20]
TopBowlers


# In[38]:


plt.figure(figsize=(10,8))
plt.bar(np.arange(len(TopBowlers)),TopBowlers['economy'],color='y')
plt.xticks(ticks=np.arange(len(TopBowlers)),labels=TopBowlers.index,rotation=90)
plt.xlabel('Bowler')
plt.ylabel('economy')
plt.title('Bowlers Economy')
plt.show()


# In[39]:


# economy has to be low


# #### Wickets taken by a bowler

# In[40]:



plt.figure(figsize=(10,8))
plt.bar(np.arange(len(TopBowlers)),TopBowlers['wickets'],color='pink')
plt.xticks(ticks=np.arange(len(TopBowlers)),labels=TopBowlers.index,rotation=90)
plt.xlabel('Bowler')
plt.ylabel('wickets')
plt.title('Bowlers Wickets')
plt.show()


# ### Observation and conclusion
# 
# #### SL Malinga and DJ Bravo are the leading wicket takers in this tournament.
# 
# ##### Best Pick for endorsement :
# #### Batsmen:
# *Virat kohli,SK Raina,CH Gayle,AB de Villiers,DA warner*
# 
# #### Bowlers:
# *UT Yadhav,DJ Bravo,SL Malinga*
# 
# #### Teams:
# MUMBAI INDIANS,CHENNAI SUPER KINGS,KOLKATA KNIGHT RIDERS

# ##### thank you

# In[41]:


git remote add origin https://github.com/TWINKLE-RAWAT/GRIP-THE-SPARKS-FOUNDATION-TASK-5-SPORTS


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




