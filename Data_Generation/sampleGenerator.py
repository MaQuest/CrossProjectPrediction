#program to generate independent samples (requirement pairs) within different projects
#List of projects can be provided in products.csv file with ToBeProcessed Flag set to True/False
#Dataset is available in /NLP/Requires_enhancement_2_enhancementData.csv

import pandas as pd
import os
import winsound
import argparse

pd.set_option("display.max_rows", 1000)
pd.set_option('display.max_colwidth', -1)
pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', None)

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second

def get_args():
    
    parser = argparse.ArgumentParser(description="This script takes the dependent requirements from various projects as input and generates independent requirement pairs among the projects.", formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input","-i",type=str,required = False,default="../NLP/Requires_enhancement_2_enhancementData.csv",help="path to dependent requirement combinations data file")
    parser.add_argument("--maxPairs","-mp",type=int,default=50000, required=False,help="Maximum number of independent requirement pairs you wish to generate for each set of projects.")
    
    return parser.parse_args()

def genPosNegSamples(df_data,products,maxPairs):
    '''
    Generates Positive (dependent) and Negative (Independent) Samples for the dataset.
    
    Saves the Negative Samples for various combinations of products in .csv files
    '''
    print(len(df_data))
    df_iVsj = pd.DataFrame()
    for i in products:
        for j in products:
            print ("Generating results for : "+i+" vs "+j)
            df_iVsj_pos = df_data[(df_data['req1Product'].astype(str) == i) & (df_data['req2Product'].astype(str) == j)]
            print ("Number of positive samples found in data set : ",len(df_iVsj_pos))
        
            print("Generating negative samples...")
            df_iVsj_neg = genNeg(df_iVsj_pos,maxPairs)
            print ("Number of negative samples generated : ",len(df_iVsj_neg))

            df_iVsj_neg.to_csv("./NegativeSamples/"+i+"Vs"+j+"_"+str(len(df_iVsj_neg))+"_Samples.csv",encoding="utf-8")
            print ("Generated negative samples are available at : ","./NegativeSamples/"+i+"Vs"+j+"_"+str(len(df_iVsj_neg))+"_Samples.csv")
            winsound.Beep(frequency, duration)
            input('hit enter to proceed to next set...')

def genNeg(df_data,maxSamples):
    '''
    From the positive samples generates the combinations which are independent of each other.

    for every id get the data from df_data itself and generate the pair 
        req1	req1Id	req1Priority req1Product
        # req1Release	req1Severity	req1Type	req1Ver	
                
        #  req2	req2Id	req2Priority req2Product	
        # req2Release	req2Severity	req2Type	req2Ver

    Limiting the number of pairs to 50000 (default)
    '''

    UniqueIds = df_data['req1Id']
    UniqueIds = UniqueIds.append(df_data['req2Id'])
    UniqueIds = list(set(UniqueIds)) # this is a set of Unique Ids and sorts them
    UniqueIds.sort()
    
    df_neg = pd.DataFrame(columns=['req1','req1Id','req1Priority','req1Product','req1Release','req1Severity','req1Type','req1Ver','req2','req2Id','req2Priority','req2Product','req2Release','req2Severity','req2Type','req2Ver','BinaryClass','MultiClass'])
    List_data_neg = []
    for i in UniqueIds:
        for j in UniqueIds:
            if (i!=j):
                try: 
                    #Look for the elements and get that particular rows from the dataset
                    ele1 = df_data[df_data['req1Id'] == i].iloc[0]
                except:
                    ele1 = df_data[df_data['req2Id'] == i].iloc[0] #search in other half

                try:
                    ele2 = df_data[df_data['req1Id'] == j].iloc[0]
                except:
                    ele2 = df_data[df_data['req2Id'] == j].iloc[0] #search in other half
                
                #Generate Pair

                # req1	req1Id	req1Priority req1Product	
                # req1Release	req1Severity	req1Type	req1Ver	
                
                # req2	req2Id	req2Priority req2Product	
                # req2Release	req2Severity	req2Type	req2Ver
                # BinaryClass	MultiClass	
                
                pair = {'req1':ele1['req1'],
                        'req1Id':ele1['req1Id'], 
                        'req1Type':ele1['req1Type'],
                        'req1Priority':ele1['req1Priority'], 
                        'req1Severity':ele1['req1Severity'],
                        'req1Release':ele1['req1Release'],
                        'req1Product':ele1['req1Product'],
                        'req1Ver':ele1['req1Ver'],

                        'req2':ele2['req2'],
                        'req2Id':ele2['req2Id'],
                        'req2Type':ele2['req2Type'],
                        'req2Priority':ele2['req2Priority'],
                        'req2Severity':ele2['req2Severity'],
                        'req2Release':ele2['req2Release'],
                        'req2Product':ele2['req2Product'],
                        'req2Ver':ele2['req2Ver'],
                        'BinaryClass':0,
                        'MultiClass':"norequires"} 
                try:
                    print (pair)
                except:
                    print("Some print error")
                print("----------------------------------------")
                List_data_neg.append(pair) 
                print ("Length of list_data_neg : ",len(List_data_neg))
                if (len(List_data_neg)>(maxSamples - 1)):  #maxSamples - 1 because list starts with index 0
                    df_neg = pd.DataFrame(List_data_neg)
                    #print (df_neg.tail(5))
                    return df_neg

    df_neg = pd.DataFrame(List_data_neg)
    return df_neg

def main():

    args = get_args()  #Get all the command line arguments

    ifileName = args.input     
    maxNegativeSamples = args.maxPairs

    #Read dataset into dataframe
    dataset = pd.read_csv(ifileName)

    #Get list of products for which negative samples are to be generated
    productsDF = pd.read_csv("Products.csv")
    productsFiltered = productsDF[productsDF['ToBeProcessedFlag']==True] #only use products with ToBeProcessedFlag = True 
    products_list = list(productsFiltered['ProductName'])

    #Create NegativeSamples directory if it doesn't exist
    if not os.path.exists('NegativeSamples'):
        os.makedirs('NegativeSamples')

    #Generate Positive Negative Samples and Save them to the csv file
    genPosNegSamples(dataset,products_list,maxNegativeSamples)

if __name__ == '__main__':
    main()