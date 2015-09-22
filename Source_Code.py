#Sai Krishna Prasad Jayakumar
#Converted Excel sheet into .Csv file and renamed to DataScience.csv

#Import packages
from __future__ import division


import numpy as np
import pandas as pd



dataset = pd.read_csv("DataScience.csv") #Upload CSV file as Data Frame
#print len(dataset.columns)
Mod_Conc = [0 for x in range(1000)] #Moderator Concurrence 
GOC = ["NA" for x in range(1000)] #GOC column
df = dataset


'''Question 1'''
print "Question1 \n"
df["Mod_Conc"] = Mod_Conc #Added Column to dataset
df["GOC"] = GOC  #Add GOC Column
#For all rows where at any of {S0,S1, S2, S3, S4, S5} has at least 2 votes, the moderators Concur. So, Mod_Concur is 1.
df["Mod_Conc"][(df["S0"]>=2) | (df["S1"]>=2) | (df["S2"]>=2) | (df["S3"]>=2) | (df["S4"]>=2) | (df["S5"]>=2)] = 1
#For all rows where Mod_Conc is 1, set GOC to the manually classified job
df["GOC"][df["S0"]>=2] = "Other"
df["GOC"][df["S1"]>=2] = df["GOC1"]
df["GOC"][df["S2"]>=2] = df["GOC2"]
df["GOC"][df["S3"]>=2] = df["GOC3"]
df["GOC"][df["S4"]>=2] = df["GOC4"]
df["GOC"][df["S5"]>=2] = df["GOC5"]

'''Grouping the data by GOC using Pandas groupby function
Finding the Sum of Impressions and Jobs for each GOC group
Finding the ratio of the two, defined as Competitiveness'''
df_goc = df.groupby("GOC")
Competitiveness =  (df_goc["# Impressions"].aggregate(np.sum))/(df_goc["# Jobs"].aggregate(np.sum)) 
Newarray = np.array(Competitiveness) #Convert to array
Indices = sorted(range(len(Newarray)), key=lambda k: Newarray[k]) #Returns indices sorted in increasing order of Competitiveness
Jobs_byComp = Competitiveness[Indices] 
print "5 Most Competitive jobs are: ", "\n" , Jobs_byComp[-5:] #Returns top 5 Most Competitive Jobs
print "\n"
print "Least Competitive Jobs are: ", "\n", Competitiveness[Competitiveness == min(Competitiveness)] # Returns Least Competitive Jobs
print "\n"




'''Question 2'''
print "\n Question 2 "
#What is the Accuracy of the classifier?
#Metric 3 chosen: (Number of JobTitles where max(Conf) = Job Chosen by both Mods) / (Total Number of Job Titles)
#Define a new column Algo_Choice which indicates the job pointed to by the Maximum Confidence score
Algo_Choice = ["NA" for x in range(1000)]
df["Algo_Choice"] = Algo_Choice #GOC with Top Confidence Score
df["Total_Score"] = [0 for x in range(1000)] #Row wise Sum of confidence scores
df["Total_Score"] = df["Conf1"] + df["Conf2"] + df["Conf3"] + df["Conf4"] + df["Conf5"]



for i in range(1000):
    if (df.loc[i,"Conf1"] == max(df.loc[i,"Conf1":"Conf5"])):
        df.loc[i,"Algo_Choice"] = df.loc[i,"GOC1"]        
    else:
        if (df.loc[i,"Conf2"] == max(df.loc[i,"Conf1":"Conf5"])):
            df.loc[i,"Algo_Choice"] = df.loc[i,"GOC2"]
        else:
            if (df.loc[i,"Conf3"] == max(df.loc[i,"Conf1":"Conf5"])):
                df.loc[i,"Algo_Choice"] = df.loc[i,"GOC3"]
            else:
                if (df.loc[i,"Conf4"] == max(df.loc[i,"Conf1":"Conf5"])):
                    df.loc[i,"Algo_Choice"] = df.loc[i,"GOC4"]
                else:
                    if (df.loc[i,"Conf5"] == max(df.loc[i,"Conf1":"Conf5"])):
                        df.loc[i,"Algo_Choice"] = df.loc[i,"GOC5"]



#Now calculating the Metric for Accuracy
Accuracy = len(df[df["Algo_Choice"]==df["GOC"]])/len(df)
print "\n", "Accuracy"
print "The Accuracy of the Classifier is =", Accuracy*100 , "%"

#Accuracy of Classifier from User Experience Perspective
#Ratio of Impressions (Analogous to Search Results) that could not be classified to a GOC
print "Classifier Accuracy by User Experience=  ", (sum(df["# Impressions"][df["Algo_Choice"]==df["GOC"]]))/(sum(df["# Impressions"]))*100, "%", "\n"



'''Question 3'''
print "\n Question 3"
df["Top_Score"] = [0 for x in range(1000)] #Row wise top Score Percentage
for i in range(1000):
    if (df.loc[i,"Conf1"] == max(df.loc[i,"Conf1":"Conf5"])):
        df.loc[i,"Top_Score"] = df.loc[i,"Conf1"]  / df.loc[i,"Total_Score"]      
    else:
        if (df.loc[i,"Conf2"] == max(df.loc[i,"Conf1":"Conf5"])):
            df.loc[i,"Top_Score"] = df.loc[i,"Conf2"] / df.loc[i,"Total_Score"]
        else:
            if (df.loc[i,"Conf3"] == max(df.loc[i,"Conf1":"Conf5"])):
                df.loc[i,"Top_Score"] = df.loc[i,"Conf3"] / df.loc[i,"Total_Score"]
            else:
                if (df.loc[i,"Conf4"] == max(df.loc[i,"Conf1":"Conf5"])):
                    df.loc[i,"Top_Score"] = df.loc[i,"Conf4"] / df.loc[i,"Total_Score"]
                else:
                    if (df.loc[i,"Conf5"] == max(df.loc[i,"Conf1":"Conf5"])):
                        df.loc[i,"Top_Score"] = df.loc[i,"Conf5"] / df.loc[i,"Total_Score"]
                        
#Goodness of Confidence Estimates
#Measured as  1 / Abs((Average percentage of times Top Score was chosen) / (Average percentage of Top Confidence Score)-1)
Goodness_of_ConfScores = 1/ np.abs(Accuracy /  np.mean(df["Top_Score"]) - 1)
print "\nThe Goodness of the Confidence Estimates is measured as ", Goodness_of_ConfScores


#Confidence Threshold for Rejection
#Choose Average  of Row-wise percentage of Maximum of Confidence Scores with 0 votes from moderator
df["Highest_Reject"] = [0 for x in range(1000)] #Row wise Highest Reject Score Percentage

df1 = df[["S1","S2","S3","S4", "S5"]]
df2 = df[["Conf1", "Conf2", "Conf3", "Conf4", "Conf5"]]
zero = [0,0,0,0,0]
for i in range(1000):
    row_vector = np.array(df1.ix[i,:])
    df["Highest_Reject"].ix[i] = (max(df2.ix[i,:][(row_vector == zero)])/ df["Total_Score"].ix[i])*100
    
print "Mean Reject Percentage i.e. If Confidence Score is less than " , np.mean(df["Highest_Reject"]), "% of Total Confidence Score of Top 5, it can be rejected"




