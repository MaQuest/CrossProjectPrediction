'''
Read the files and do the following scenarios
Part 1
1) train using the same product and compute the F1 using same project data
2) test for the unseen data from same project and for different project
3) measure accuracy

Part 2:
1) train using the same product + partial other project and compute the F1
2) test for the unseen data from same project and for different project
3) measure accuracy


NOTE: the that the negative samples data has the positive samples pair but with the "norequires" label
So it is important to subtract or eliminate these records first from it before processing
'''
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
#import utilities as util
import importlib
#import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score,train_test_split, StratifiedKFold
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,classification_report
from textblob import TextBlob
import matplotlib.pyplot as plt
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
#winsound.Beep(frequency, duration)

DepFile = "Processed_Requires_enhancement_2_enhancementData.csv"
#AllIndeFiles = os.listdir('Data/')
Projects = ["Firefox","Core","Toolkit"]
# IndePendentFiles = ["ProcessedData/ProcessedFirefoxVsFirefox50000.csv",
#                     "ProcessedData/ProcessedFirefoxVsCore.csv",                 
#                     "ProcessedData/ProcessedFirefoxVsToolkit.csv"] #["ProcessedData/ProcessedToolkitVsToolkit.csv"]#["ProcessedData/ProcessedCoreVsCore50000.csv"]
# IndePendentFiles = ["ProcessedData/ProcessedToolkitVsToolkit.csv",
#                     "ProcessedData/ProcessedToolkitVsFirefox.csv",
#                     "ProcessedData/ProcessedToolkitVsCore.csv"] 
IndePendentFiles = ["ProcessedData/ProcessedCoreVsCore50000.csv",
                    "ProcessedData/ProcessedCoreVsToolkit.csv",                 
                    "ProcessedData/ProcessedCoreVsFirefox.csv"] 

#clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=.01, kernel='rbf')
#clf = SVC(kernel='rbf')
clf = MultinomialNB()
skf = StratifiedKFold(5)
SIZE = [50,100,200,500,700,1000,1500]#,1700,3000,6000]#,10000,18000]
#SIZE = [100]
CROSS_PROJECT_SIZE=[10,50,100]#,150]
# try a range of k values
all_scores, meanNstd = list(), list()
all_crossproject, meanNstdCross = list(), list()
# define a function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    #print (text)
#     wnl = nltk.WordNetLemmatizer()
#     [wnl.lemmatize(t) for t in text]
    text = str(text)
    words = TextBlob(text).words
    #print "in here"
    #print [word.lemmatize() for word in words]
    #print "\n------------------------\n"
    return [word.lemmatize() for word in words]
    #return

def pickData(Localsize,cross_size,projectName,pro1):
    print("********************************************************************")
    #df_Dep_Alldata = pd.read_csv(DepFile,index_col=[7,15])
    #df_Ind_AVsA = pd.read_csv(IndePendentFiles[0],index_col=[4,12]) #for training
    df_Dep_Alldata = pd.read_csv(DepFile, usecols=['MultiClass', 'req1Id', 'req1',
       'req1Product', 'req1Type', 'req2', 'req2Id', 'req2Product',
       'req2Type'])#, index_col=['req1Id','req2Id'])
    df_Ind_AVsA = pd.read_csv(IndePendentFiles[0], usecols=['MultiClass', 'req1Id', 'req1',
       'req1Product', 'req1Type', 'req2', 'req2Id', 'req2Product',
       'req2Type'])#, index_col=['req1Id','req2Id']) #for training
    
    #print(df_Dep_Alldata.columns)
    #print(df_Ind_AVsA.columns)
    #print(df_Dep_Alldata.columns.get_loc("req1Id"),df_Dep_Alldata.columns.get_loc("req2Id"))
    #print(df_Ind_AVsA.columns.get_loc("req1Id"),df_Ind_AVsA.columns.get_loc("req2Id"))
    
    
    #------------------------------------------------------------------------------
    #find the crossproject file and return the file name
    independtNegFile = "\n".join(s for s in IndePendentFiles  if pro1.lower() in s.lower()) 
    print("Cross project: ",independtNegFile)
    df_Dep_crossProject = df_Dep_Alldata[(df_Dep_Alldata["req1Product"] == projectName) & (df_Dep_Alldata["req2Product"] == pro1)]
    df_Ind_crossProject = pd.read_csv(independtNegFile, usecols=['MultiClass', 'req1Id', 'req1',
                                                                    'req1Product', 'req1Type', 'req2', 'req2Id', 
                                                                    'req2Product','req2Type'])
    
    #balance it 
    '''
    read the cross project data and gradually add to the existing dataset and keep remaining for testing
    remaining pass on for testing
    '''
    #NOTE: this is because the negative set might be smaller 
    if len(df_Dep_crossProject) > len(df_Ind_crossProject):
        size = len(df_Ind_crossProject)
    else:
        size = len(df_Dep_crossProject)
    
    #retain a copy
    df_Dep_cross_ALL= df_Dep_crossProject
    df_Ind_cross_ALL = df_Ind_crossProject
    
    #append later this to training/cross validation
    df_Dep_crossProject = df_Dep_crossProject.sample(cross_size)
    cross_proj_df_TRAIN = df_Dep_crossProject.append(df_Ind_crossProject.sample(cross_size))
    cross_proj_df_TRAIN = cross_proj_df_TRAIN.sample(frac = 1)
    
    #remaining pass on for testing
    remaining = size-cross_size
    df_TEST_Dep_crossProject = df_Dep_cross_ALL.sample(remaining)
    cross_proj_df_TEST = df_TEST_Dep_crossProject.append(df_Ind_cross_ALL.sample(remaining))
    cross_proj_df_TEST = cross_proj_df_TEST.sample(frac = 1)
    
    print("Cross project Data:\n:", len(df_Dep_crossProject), len(df_Ind_crossProject))
    

    #input("hit enter")
    #------------------------------------------------------------------------------

    ## get the pairs from both dataset with the same req1product
    d_pairs = df_Dep_Alldata[(df_Dep_Alldata["req1Product"] == projectName) & (df_Dep_Alldata["req2Product"] == projectName)]
        
    print("total available:", "requires: ", len(d_pairs),"no_Requires: ", len(df_Ind_AVsA))
    #independent will belong to the same project
    i_pairs = df_Ind_AVsA.append(d_pairs)
    print(len(i_pairs), len(i_pairs[i_pairs['MultiClass']=="requires"]))
    i_pairs = i_pairs.drop_duplicates(subset=['req1Id', 'req2Id'], keep=False)
    i_pairs = i_pairs[i_pairs.MultiClass!="requires"]

    #print(len(i_pairs), len(i_pairs[i_pairs['MultiClass']=="requires"]))
    #input("hit enter")
    
    d_pairs = d_pairs.sample(Localsize)
    train_df = d_pairs #.sample(int(train_size/2))
   
    #print("pos",len(train_df))
    train_df = train_df.append(i_pairs.sample(len(d_pairs))) #this is balanced train set
    train_df = train_df.append(cross_proj_df_TRAIN)
    train_df = train_df.sample(frac = 1)
   
    ### randomize the data frame
    print("pos+neg",len(train_df))
    #input("hit enter")
   
    train_df = train_df.sample(frac = 1)
    #print(train_df['req1'].head(5))
    train= train_df[['req1', 'req2']]
    train_lables = np.array(train_df['MultiClass'])

    #binary_class = np.array(train_df["MultiClass"])
             
    tfidf_transformer = TfidfTransformer()
    count_vect = CountVectorizer(tokenizer=lambda doc: doc, analyzer=split_into_lemmas, lowercase=False, stop_words='english')
    
    ## condense the data
    X_train_counts = count_vect.fit_transform(np.array(train))
    X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)

    clf_model = MultinomialNB().fit(X_train_tfidf,train_lables)
    ## k-fold validation
    scores = cross_val_score(clf_model, X_train_tfidf, train_lables, cv = skf, scoring='accuracy', )
    avgValidation = np.average(scores)
    all_scores.append(scores)
    m,s = mean(scores), std(scores)
    meanNstd.append([m,s])
    
    print("Project:",projectName,"  5_CrossValidation: ", avgValidation, "  Balanced Size:", 2*Localsize, " Cross_porject train Size:", cross_size)
    #print(len(d_pairs), len(i_pairs))
    
    f1, precision, recall = crossProjectTesting(cross_proj_df_TEST,count_vect,tfidf_transformer,clf_model)
    print(pro1, f1,"\t", precision,"\t", recall)
    # f1, precision, recall = crossProjectTesting(IndePendentFiles[2], df_Dep_Alldata[(df_Dep_Alldata["req1Product"] == projectName) & (df_Dep_Alldata["req2Product"] == pro2)],count_vect,tfidf_transformer,clf_model)
    # print(pro2, f1,"\t", precision,"\t", recall)
    print("-------------------------------------------------------------------------------")
    
    return

def crossProjectTesting(df_Test,count_vect,tfidf_transformer,clf_model):
    test_df = df_Test
    test_df = test_df.sample(frac = 1)

    test_lables = np.array(test_df["MultiClass"])
    test = test_df[['req1', 'req2']]
           ## condense the data 
    X_test_counts = count_vect.transform(np.array(test))
    X_test_tfidf= tfidf_transformer.fit_transform(X_test_counts)
           ## seperate prediction array from actual value array
    predict_labels = clf_model.predict(X_test_tfidf)
    actualLabels = np.array(test_lables)
            ## create the confusion matrix
    scores = confusion_matrix(actualLabels, predict_labels)
           
    precision = round(precision_score(actualLabels, predict_labels,average='macro'),2)
    recall = round(recall_score(actualLabels, predict_labels,average='macro'),2)
    f1 = round(f1_score(actualLabels, predict_labels,average='macro'),2)
    #print(f1,"\t", precision,"\t", recall)#, cm)
    all_crossproject.append(scores)
    m,s = mean(scores), std(scores)
    meanNstdCross.append([m,s])
    return(f1,precision,recall)

    #f1_scores.append(f1)

for i in SIZE:
    for j in CROSS_PROJECT_SIZE:
        projectName = Projects[1]
        pro1= Projects[2]
        #pro2= Projects[2]
        pickData(i,j, projectName,pro1)
        winsound.Beep(frequency, duration)
print (all_scores)
print (meanNstd)

# plot
pyplot.boxplot(all_scores, labels=SIZE)
pyplot.show()
# pyplot.boxplot(all_crossproject, labels=CROSS_PROJECT_SIZE)
# pyplot.show()

# projectName = Projects[0]
# pro1= Projects[1]
# pro2= Projects[2]
# pickData(SIZE, projectName,pro1,pro2)
# winsound.Beep(frequency, duration)