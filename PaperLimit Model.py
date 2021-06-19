# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Simulation libraries
import numpy as np    #Install numpy
import pandas as pd   #Install pandas
import networkx as nx #Install networkx
import random
import statsmodels.stats.power as smp #Install statsmodels
import math

#Plotting Libraries
import matplotlib.pyplot as plt #Install matplotlib
import time
from scipy import stats


# %%

#Simulation of one generation
def competition(lifeSpan, sampleSize, startupCost, sampleCost, ExpDistributionShape, scoopedCost, negativeResultCost, limit, randomWalkP,acceptP): 
    oneYear = lifeSpan/10 #Duration of one timePeriod in the model
    scientistID = list(np.arange(populationSize))  #ID to keep track of scientists
    amountOfQuestions = round((lifeSpan / (startupCost + minSampleSize)) * populationSize + populationSize) #Generates pool of questions
    questionID = list(np.arange(amountOfQuestions)) #ID to keep track of questions
    effectSizeQuestion = list(np.round(np.array(np.random.exponential(1/ExpDistributionShape, size=amountOfQuestions)), 1)) #Pool of effect sizes Determined by lambda
    drawerQ = list(np.array(np.zeros(shape=[populationSize, 0]),np.int32)) #File drawer of papers
    drawerR = list(np.array(np.zeros(shape=[populationSize, 0]),np.bool_)) #File drawer of results linked to papers

    #Initialize variables
    workedOnQuestions = 0  #To calculate questions worked on
    publishedQuestions = 0 #To calculate questions published
    totalAuthors = 0 #To calculate authors per paper
    PNMatrix = np.array([[0,0,0,0],[0,0,0,0]]) #To store confusion matrix
    drawerPN = list(np.array(np.zeros(shape=[populationSize, 0]),np.int32)) #To store all results, even if removed from file-drawer

    #Dataframe consisting of the scientist population
    d1 = {'scientistID' : scientistID, 'sampleSize' : sampleSize, 'questionID' : 0, 'publications' : 0, 'payoff' : 0.0, 'RandomWalkP': randomWalkP , 'AcceptP': acceptP }
    scientistDataFrame = pd.DataFrame(data=d1)
    
    #Dataframe for storing all the results
    d2 = {'questionID' : questionID, 'scientistID' : -1, 'sampleSize' : 0, 'effectSize': effectSizeQuestion,'result' : 0, 'published': None, 'coWriters':"None"}
    resultsDataFrame = pd.DataFrame(data=d2)
    resultsDataFrame['coWriters'].astype('object') #store object types

    #Generate a co-authorship network
    averageNeighbours = 8 
    rewiringP = 0.1
    scientistNetwork = nx.watts_strogatz_graph(populationSize, averageNeighbours, rewiringP)
    for i in ([e for e in scientistNetwork.edges]): #Add edge weights based on neighbours in common
        scientistNetwork[i[0]][i[1]]['weight'] = len( sorted(  nx.common_neighbors(scientistNetwork, i[0], i[1])  ) ) + 1

    #For the PCI credit system contribution shares have to be stored for each paper worked on
    if payoffMechanism == "PCI":
        PCIcontributions = [None]*amountOfQuestions
    
    #Tracking which questions are published and worked on
    questionIDsPublished = [] 
    questionIDsWorked = [] 
    
    #Initialize time in the model
    timePeriod = 1 #starting time period
    timeCostBaseline = list(scientistDataFrame['sampleSize'] * sampleCost + startupCost)  #Base cost for a scientists study, start-up cost + sample size
    timeCost = [[],[]]                          #Used to keep track of what kind of event takes place and when, because there are two types of events
    timeCost[0] = timeCostBaseline              #Curent event time
    timeCost[1] = np.zeros(populationSize)      #Track whether the current event is a co-write event(0) or lead-author event(1), 
                                                # for the lead-author results have to be calculated while when a collaboration has been finished
                                                # no results have to be calculated
    timeCostBacklog = np.zeros(populationSize)  #Store commitments made to work on collaborations, which have to be completed after current event
    timeCostRealTime = timeCostBaseline         #Track the backlog and current event times combined

    yearProgress = oneYear-1 # time to next year tracker
    unpubQ = questionID # list of all questions that havent been worked on
    
    #Running the simulation in a loop, until the lifeSpan has been reached
    while(timePeriod < lifeSpan):
        timeToNextEvent = min(yearProgress, min( min(timeCost[0]), lifeSpan - timePeriod)) #Check which event is first to take place
        if limit: #used to toggle limit on or off
            yearProgress = yearProgress - timeToNextEvent 
        timeCost[0] = list(timeCost[0] - np.repeat(timeToNextEvent, len(timeCost[0]))) #Forward time for scientists on current event
        timePeriod = timePeriod + timeToNextEvent  #Forward time period
        timeCostRealTime = list(timeCostRealTime - np.repeat(timeToNextEvent, len(timeCost[0]))) #Forward total time for scientists
        if (timePeriod > lifeSpan):
            break

        if (min(timeCost[0]) == 0.0): # studies are completed -> scientists store questions into their drawer
            if limit == False: #used to toggle limit on or off
                yearProgress = 0
            actingScientists = list(np.array([i for i, noTimeLeft in enumerate(timeCost[0]) if noTimeLeft == 0])) #Scientist who have an event
            newQuestionSearchers = list(actingScientists) #Scientists who need a new question, so have no backlog that has to be completed

            
            removeActingScientists = []
            removeQuestionSearchers = []
            for sid in actingScientists: #Check what has to happen for each scientist that has an event
                if timeCost[1][sid] == 0:
                    removeActingScientists.append(sid) # We need not to calculate results, this was a co-write event
                    if timeCostBacklog[sid] != 0:      # We need not to move them on to a new question, they have a backlog to be completed
                        removeQuestionSearchers.append(sid)
                        timeCost[0][sid] = timeCostBacklog[sid] 
                        timeCost[1][sid] = 0
                        timeCostBacklog[sid] = 0
                if (timeCost[1][sid] == 1) and (timeCostBacklog[sid] != 0): #We need to calculate results, but they have to complete a backlog
                    removeQuestionSearchers.append(sid)
                    timeCost[0][sid] = timeCostBacklog[sid]
                    timeCost[1][sid] = 0
                    timeCostBacklog[sid] = 0
            actingScientists = [x for x in actingScientists if x not in removeActingScientists] #These are all scientists who need a new RQ
            newQuestionSearchers = [x for x in newQuestionSearchers if x not in removeQuestionSearchers] #These are all scientists we need to calculate results for


            #Calculate results for the scientists who finished their study
            if len(actingScientists) != 0: 
                amountActingScientists = len(actingScientists) 
                questionActingScientists = list(scientistDataFrame[scientistDataFrame['scientistID'].isin(actingScientists)]['questionID']) #Retrieve the question
                effectSizeQuestion = [] 
                for i in questionActingScientists: 
                    effectSizeQuestion.append(resultsDataFrame.loc[[i],'effectSize']) #Retrieve the effect size
                sampleSizeActingScientists = list(scientistDataFrame[scientistDataFrame['scientistID'].isin(actingScientists)]['sampleSize']) #Retrieve the Sample Size
                sampleSizeIndex = 0 

                powerOfQuestions = [] 
                for qid in questionActingScientists: #Calculate the power they have on a given effect size
                        resultsDataFrame.at[qid,'published'] = False 
                        powerOfQuestion =  smp.ttest_power( effect_size= resultsDataFrame._get_value(qid, 'effectSize'),nobs=sampleSizeActingScientists[sampleSizeIndex], 
                                                        alpha=0.05, alternative='two-sided') 
                        powerOfQuestions.append(powerOfQuestion) 
                        sampleSizeIndex += 1 

                positiveResult = [] 
                for r in range(amountActingScientists): #Calculate whether they get a positive result
                    positiveResult.append(np.random.uniform(0,1) < powerOfQuestions[r]) 

                index = 0
                for sid in actingScientists: #Store whether the result they got was a TP,FP,TN,FN
                    qid = questionActingScientists[np.where(actingScientists == sid)[0][0]] 
                    workedOnQuestions += 1
                    if (resultsDataFrame.at[qid, 'effectSize'] > 0) and (positiveResult[index] == True): # True Positive
                        PNMatrix[0][0] += 1
                        drawerPN[sid] = np.append(drawerPN[sid], 0) 
                    elif  (resultsDataFrame.at[qid, 'effectSize'] == 0) and (positiveResult[index] == True): # False Positive
                        PNMatrix[0][1] += 1
                        drawerPN[sid] = np.append(drawerPN[sid], 1) 
                    elif (resultsDataFrame.at[qid, 'effectSize'] == 0) and (positiveResult[index] == False): # True Negative
                        PNMatrix[0][2] += 1
                        drawerPN[sid] = np.append(drawerPN[sid], 2) 
                    elif (resultsDataFrame.at[qid, 'effectSize'] > 0) and (positiveResult[index] == False):# False Negative
                        PNMatrix[0][3] += 1
                        drawerPN[sid] = np.append(drawerPN[sid], 3) 
                    index += 1

                for sid in actingScientists: #Add the completed questions and results to a scientists drawer
                    qid = questionActingScientists[np.where(actingScientists == sid)[0][0]] 
                    drawerQ[sid] = np.append(drawerQ[sid], [qid]) 
                    drawerR[sid] = np.append(drawerR[sid], [positiveResult[np.where(actingScientists == sid)[0][0]]]) 
                questionIDsWorked = np.concatenate((questionIDsWorked, questionActingScientists)) #Update questions worked on
            
                for sid in actingScientists: #Update results dataFrame with the results
                    qid = questionActingScientists[np.where(actingScientists == sid)[0][0]] 
                    resultsDataFrame.loc[[qid],'scientistID'] = sid 
                    resultsDataFrame.loc[[qid],'sampleSize'] = sampleSizeActingScientists[np.where(actingScientists == sid)[0][0]] 
                    resultsDataFrame.loc[[qid],'result'] = positiveResult[np.where(actingScientists == sid)[0][0]] 
            
            #All scientists who need new questions, get new questions
            if len(newQuestionSearchers) != 0:
                for sid in newQuestionSearchers: 
                    nextQuestion = np.random.choice(unpubQ) #Randomly select an unpublished question
                    scientistDataFrame.at[sid, 'questionID'] = nextQuestion #Update dataframe that scientist has new question

                for lsid in newQuestionSearchers: #Use their collaboration strategy to find collaborators

                    if scientistDataFrame.at[lsid, 'RandomWalkP'] > 0.01: #If they are willing to collaborate, they go on a random walk
                        startNodeRWP = scientistDataFrame.at[lsid, 'RandomWalkP'] #Retrieve RWP strategy
                        startNode = lsid   #Remember start node  
                        currentNode = lsid #Start at own node
                        endWalk = False    #Parameter to check whether random walk has ended
                        path = [startNode] #Keep track of the path walked
                        studyCost = timeCostBaseline[lsid] #Baseline cost of the scientist doing the walk


                        while endWalk == False:
                            randomNumber = np.random.uniform(0,1,1)
                            if randomNumber < startNodeRWP: #Determine whether the scientists keeps walking
                                neighbors = [n for n in scientistNetwork[currentNode]] #Get neighbouring nodes to the node currently at
                                neighbors = [x for x in neighbors if x not in path] #Remove neighbours already visited from path
                                if len(neighbors) != 0: #If there is a neighbour to move to
                                    transitionWeights = list([])
                                    for nsid in neighbors:
                                        transitionWeights.append(scientistNetwork[currentNode][nsid]['weight']) #Get all edge weights
                                    choice = random.choices(neighbors, weights = transitionWeights , k = 1) #Make a choice which direction to go
                                    currentNode = choice[0] #This is the new node the scientist end up at
                                    path.append(currentNode) #Add this node to the path
                                    if len(path) > 40: #End walk if the path lengh is longer than 40
                                        endWalk = True
                                        path.remove(startNode)
                                        candidates = path
                                else: #If no neighbours the path ends as well
                                    endWalk = True
                                    path.remove(startNode)
                                    candidates = path
                            else: #The decision to stop walking was made
                                endWalk = True
                                path.remove(startNode)
                                candidates = path
                        candidates = [x for x in candidates if ((timeCostRealTime[x]) < (studyCost) )] #These are all collaboration candidates

                        candidatesAccepted = [] 
                        candidatesBacklog = []

                        if len(candidates) != 0: #If any candidates are in the path
                            for csid in candidates:
                                if scientistDataFrame.at[csid, 'AcceptP'] > np.random.uniform(0,1,1) :#Check whether the candidates accept the collaboration
                                    candidatesAccepted.append(csid) #ID of accepted scientists
                                    candidatesBacklog.append(timeCostRealTime[csid]) #Current task length to be completed of candidates

                        if len(candidatesAccepted) != 0: #If somebody accepted the request

                            weights = [1*precisionWeights] #Set precision of contribution shares generated
                            numberOfCandidates = len(candidatesAccepted) 
                            piecesContribution = np.zeros((numberOfCandidates, numberOfCandidates+1)) #To store contribution shares

                            for i in range(numberOfCandidates): #Generate possible contribution shares up to the number of candidates
                                                                # e.g. if 3 candidates than shares for 1,2 and 3 candidates are generated to see which one fits
                                if i < 4: #Contribution share weight based on position
                                    weights.append((1/(i+2))* precisionWeights)
                                else: #limit minimum contribution share weight to 1/5 
                                    weights.append((1/(5))* precisionWeights)

                                #Store these contribution shares for each number of candidates in a matrix
                                contributions = np.round(np.random.dirichlet((weights), 1),2) 
                                emptySpace = np.array(  [np.repeat(0,   ( numberOfCandidates - (i+1) ))] )
                                contributions = (np.concatenate((contributions, emptySpace), axis=1)).reshape(numberOfCandidates + 1)
                                piecesContribution[i] = contributions
                            piecesCost = piecesContribution * studyCost #multiply shares with timecost of study to be conducted

                            #Order scientists by the time they have available
                            possibleShares = []
                            orderedList = [list(e) for e in zip(candidatesAccepted, candidatesBacklog)] 
                            orderedList.sort(key = lambda orderedList: orderedList[1])
                            candidatesBacklog = list(np.array(orderedList)[:,1])
                            candidatesAccepted = list(np.array(orderedList)[:,0])

                            #See which candidates can work on the contributions
                            for i in range(numberOfCandidates):
                                if all(  np.array(candidatesBacklog[:i+1] + piecesCost[i][1:i+2]) <= np.array(piecesCost[i][0])  ):
                                    possibleShares.append(i)

                            #If no candidates can work on the paper, the lead-author continues as a single author
                            if len(possibleShares) == 0: #no co-writing can occur
                                timeCost[0][lsid] = timeCostBaseline[lsid] 
                                timeCostRealTime[lsid] = timeCostBaseline[lsid]
                                timeCost[1][lsid] = 1
                                totalAuthors += 1

                            #There are candidates who can work on the contribution, they will become co-authors
                            else: 
                                bestOption = max(possibleShares) #Highest amount of writers possible
                                qid = scientistDataFrame.loc[scientistDataFrame['scientistID'] == lsid]['questionID'].item() #Get question they will work on
                                selectedCandidates = candidatesAccepted[:len(piecesCost[bestOption][1:bestOption+2])] 
                                selectedCandidates = [int(a) for a in selectedCandidates] #These are all available candidates
                                if payoffMechanism == "PCI": #Store contribution shares in case of the PCI payoff mechanism
                                    PCIcontributions[qid] = piecesContribution[bestOption]
                                resultsDataFrame.at[qid, "coWriters"] = selectedCandidates #Set these authors as co-writers on the given question

                                pieceCount = 1
                                for sid in selectedCandidates: #Add the time cost of the contribution share to co-writers timecost
                                    timeCostBacklog[sid] += np.round(piecesCost[bestOption][pieceCount]) #Update backlog with timecost
                                    timeCostRealTime[sid] += np.round(piecesCost[bestOption][pieceCount]) #Update total time cost to be completed
                                    pieceCount += 1
                                timeCost[0][lsid] = np.round(piecesCost[bestOption][0]) + len(selectedCandidates) * 10 #add timecost of lead-author
                                timeCostRealTime[lsid] = np.round(piecesCost[bestOption][0]) + len(selectedCandidates) * 10 #add timecost of lead-author
                                timeCost[1][lsid] = 1 
                                totalAuthors += 1 + len(selectedCandidates)    

                        if len(candidates) == 0 or len(candidatesAccepted)==0: #No co-writing can occur
                            timeCost[0][lsid] = timeCostBaseline[lsid] 
                            timeCost[1][lsid] = 1  
                            timeCostRealTime[lsid] = timeCostBaseline[lsid] 
                            totalAuthors += 1     

                    else:  #This author has a RWP of 0, so they will work as a solo-author
                        timeCost[0][lsid] = timeCostBaseline[lsid]
                        timeCost[1][lsid] = 1
                        timeCostRealTime[lsid] = timeCostBaseline[lsid]
                        totalAuthors += 1

        #A time period has ended, scientists can publish studies             
        if yearProgress == 0: 
            yearProgress = oneYear #Reset timeperiod progress

            for sid in np.repeat(scientistDataFrame['scientistID'], paperLimit): #Scientists can publish equal to the paper limit
                workedQ = drawerQ[sid] #Retrieve worked on questions
                if len(workedQ) != 0:#If the scientist has completed any questions
                    payoff = 0  #Initialize payoff
                    for q in workedQ: #Select the question with the highest payoff
                        priorPublished = questionIDsPublished.count(q) 
                        noveltyResult = pow( (1/(1+priorPublished)), scoopedCost) #Calculate the decreased value, if a scientist got scooped
                        indexQuestionR = list(workedQ).index(q)
                        if drawerR[sid][indexQuestionR]: 
                            possiblepayoff = noveltyResult #Keep track of highest payoff question
                            fullpayoff = possiblepayoff 
                        else: 
                            possiblepayoff = noveltyResult * negativeResultCost #Decrease value if negative result
                            fullpayoff = possiblepayoff

                        if resultsDataFrame.at[q, 'coWriters'] != "None": #Check if the question was co-written on
                            amountCoWriters = len(resultsDataFrame.at[q, 'coWriters'])
                            if payoffMechanism == "EC": #Decrease value because of multiple authors
                                fullpayoff = possiblepayoff
                                possiblepayoff = (1/(amountCoWriters + 1))* possiblepayoff
                            elif payoffMechanism == "PCI": #Decrease value because of multiple authors
                                fullpayoff = possiblepayoff
                                possiblepayoff = PCIcontributions[q][0] * possiblepayoff
                        if possiblepayoff > payoff: #Save question with highest payoff
                            chosenQ = q 
                            payoff = fullpayoff

                           
                    payoff = np.round(payoff,2) #Round the payoff
                    scientistDataFrame.at[sid, 'publications'] += 1 #Add publication to dataframe
                    if resultsDataFrame.at[chosenQ, 'coWriters'] != "None": #Calculate payoff for potential co-writers
                        coWriters = resultsDataFrame.at[chosenQ, 'coWriters']
                        authors = [sid] + coWriters
                        if payoffMechanism == "Fixed": #Fixed payoff for collaborators
                            for csid in authors:
                                scientistDataFrame.at[csid, 'payoff'] += payoff
                        if payoffMechanism == "SDC": #SDC payoff for collaborators
                            positionCount = 1
                            for csid in authors:
                                scientistDataFrame.at[csid, 'payoff'] += np.round( (1/positionCount) * payoff, 2) 
                                positionCount += 1
                        if payoffMechanism == "EC": #EC payoff for collaborators
                            for csid in authors:
                                scientistDataFrame.at[csid, 'payoff'] += np.round( (1/len(authors) * payoff), 2)
                        if payoffMechanism == "PCI": #PCI payoff for collaborators
                            contributionsPayoff = PCIcontributions[chosenQ] #Retrieve contribution shares
                            positionCount = 0
                            for csid in authors:
                                scientistDataFrame.at[csid, 'payoff'] += np.round( contributionsPayoff[positionCount] * payoff, 2)
                                positionCount += 1           
                    else: 
                        scientistDataFrame.at[sid, 'payoff'] += payoff #No collaborations, payoff for single author

                    #Update all drawers and the dataframe that this question is published
                    questionIDsPublished.append(chosenQ) 
                    resultsDataFrame.at[chosenQ, 'published'] = True                  
                    publishedQuestions += 1
                    if chosenQ in unpubQ:
                        unpubQ.remove(chosenQ)
                    indexQuestionQ = list(workedQ).index(chosenQ)
                    PNMatrix[1][drawerPN[sid][indexQuestionQ]] += 1
                    drawerPN[sid] = np.delete(drawerPN[sid], indexQuestionQ) 
                    drawerQ[sid] = np.delete(drawerQ[sid], indexQuestionQ) 
                    drawerR[sid] = np.delete(drawerR[sid], indexQuestionQ) 


    #Track descriptive statistics from this generation
    averageAuthors = np.round( totalAuthors / workedOnQuestions , 2)
    drawerSize = 0
    for i in range(populationSize):
        drawerSize += len(drawerQ[i])
    averageDrawerSize = drawerSize / populationSize

    #Return descriptive statistics to be saved
    return scientistDataFrame, publishedQuestions, averageDrawerSize, averageAuthors, workedOnQuestions, PNMatrix


# %%
#Let scientists in new generations adopt new strategies
def evolution(lifeSpan, generations, startupCost, sampleCost, ExpDistributionShape, scoopedCost, negativeResultCost, limit):
    # initialize population, with uniform sample sizes
    sampleSize = np.round(np.random.uniform(minSampleSize, maxSampleSize, populationSize)) # draw sample sizes from a uniform distribution
    #Possibility to toggle collaborations on and off
    if collaborations == False:
        randomWalkP = np.zeros(populationSize)
        acceptP = np.zeros(populationSize)
    else:
        randomWalkP = np.round( ((np.random.uniform(0, 100, populationSize))/100),2)
        acceptP = np.round( ((np.random.uniform(0, 100, populationSize))/100),2 )
    #Descriptive statistics that are going to be tracked across generations
    meanSampleSizes = []   
    meanPayoffs = []       
    meanPublished = []  
    meanAP = []
    meanRWP = []
    averageAuthors = []   
    questionsWorked = []            
    drawerSizes = []
    finalAP = []
    finalRWP = []
    finalSS = []
    finalPN = []

    #Print statement between model repeats
    print( loopCount, "/10. Working with: // Paper limit =", str(limit), "(",str(paperLimit),") // Collaborations =", str(collaborations), "(", str(payoffMechanism), ") // Generations =", generations)
    print("  |", end="")

    #For every generation we update the strategies
    for g in range(generations): 
        print(g+1, end="|")

        #Retrieve information from completed generation
        result = competition(lifeSpan, sampleSize, startupCost, sampleCost, ExpDistributionShape, scoopedCost, negativeResultCost, limit,                                      randomWalkP, acceptP) 
        outcomeGeneration = result[0]
        publishedQuestions = result[1]
        averageDrawerSizeGeneration = result[2]
        averageAuthorsGeneration = result[3]
        questionsWorkedOn = result[4]
        PNMatrix = result[5]

        #Select the strategies from scientists that were most succesfull based on payoff
        evolutionID = outcomeGeneration.sample(n=populationSize, weights=outcomeGeneration["payoff"], random_state=1, replace= True)['scientistID'].to_numpy()
        sid = 0

        #The new generation scientists will adopt these strategies with some noise
        for evid in evolutionID:
            sampleSize[sid] = np.absolute(np.round(np.random.normal(outcomeGeneration.at[evid, 'sampleSize'], 5, 1))) #Adopting sample size strategy
            if collaborations == False:
                randomWalkP[sid] = 0
                acceptP[sid] = 0
            else:          
                randomWalkP[sid] = np.random.normal(outcomeGeneration.at[evid, "RandomWalkP"] ,0.02 ,1) #Adopting RWP strategy
                acceptP[sid] = np.random.normal(outcomeGeneration.at[evid, "AcceptP"] ,0.05 ,1) #Adopting AP strategy
            if randomWalkP[sid] > 0.95: #Limiting probabilities to 0.95 and making sure they are not negative
                randomWalkP[sid] = 0.95
            if randomWalkP[sid] < 0.00:
                randomWalkP[sid] = 0.00
            if acceptP[sid] > 0.95:
                acceptP[sid] = 0.95
            if acceptP[sid] < 0:
                acceptP[sid] = 0
            sid += 1

        #Descriptive statistics from all generations are saved in these arrays
        combinedPO.append(outcomeGeneration["payoff"].mean())
        combinedQW.append(questionsWorkedOn)
        combinedSS.append(outcomeGeneration["sampleSize"].mean())
        combinedAP.append(outcomeGeneration["AcceptP"].mean()*100)
        combinedRWP.append(outcomeGeneration["RandomWalkP"].mean()*100)
        combinedAA.append(averageAuthorsGeneration)
        combinedDRAW.append(averageDrawerSizeGeneration)
        combinedPUB.append(publishedQuestions)
        
        #Save extra information about the convergence of the last generation
        if (g == generations - 1): 
            if collaborations == True:
                finalAP = outcomeGeneration["AcceptP"].to_numpy() * 100
                finalRWP = outcomeGeneration["RandomWalkP"].to_numpy() * 100
                finalSS = outcomeGeneration["sampleSize"].to_numpy() 
                finalPN = PNMatrix
            else:
                finalAP = np.zeros(populationSize)
                finalRWP = np.zeros(populationSize)
                finalSS = outcomeGeneration["sampleSize"].to_numpy()
                finalPN = PNMatrix 
        
        #Save confusion matrix for published and all questions
        totalW = np.sum(PNMatrix[0])
        totalP = np.sum(PNMatrix[1])
        TP.append(PNMatrix[0][0]/totalW)
        TPP.append(PNMatrix[1][0]/totalP)
        FP.append(PNMatrix[0][1]/totalW)
        FPP.append(PNMatrix[1][1]/totalP)
        TN.append(PNMatrix[0][2]/totalW)
        TNP.append(PNMatrix[1][2]/totalP)
        FN.append(PNMatrix[0][3]/totalW)
        FNP.append(PNMatrix[1][3]/totalP)

    return finalAP, finalRWP, finalSS, finalPN
    


# %%
loopCount = 0

#Initialize all arrays that are used to save descriptive statistics
TP = []
TPP = []
FP = []
FPP = []
TN = []
TNP = []
FN = []
FNP = []
combinedPO = []
combinedSS = []
combinedAP = []
combinedRWP = []
combinedAA = []
combinedDRAW = []
combinedQW = []
combinedPUB = []

#####################################################################################
############################ SET SIMULATION PARAMETERS ##############################
#####################################################################################

#### Simulation Length and Size ####
repeats = 1          #Amount of repeats
generationsT = 100    #Generations per repeat
populationSize = 120  #Size of population in the model
lifeSpanT = 10000     #Amount of time units population is alive for

#### Costs and Effect Sizes #### 
minSampleSize = 2
maxSampleSize = 1000  #Set initialization sample sizes
ExpDistributionShapeT = 8 #Set lambda of exp. distribution
startupCostT = 200    #Start up costs for study
sampleCostT =  1      #Cost for every sample
scoopedCostT = 0.5    #Cost of being scooped
negativeResultCostT = 0.5 #Cost of finding a negative result

#### Paper Limit ####
limit = True    #Toggle Limit
paperLimit = 1  #Height of the paper limit

#### Collaborations ####
collaborations = False   #Toggle collaborations
payoffMechanism = "SDC"  #Select payoff mechanism
precisionWeights = 100   #Set precision of contribution shares

#####################################################################################

#Let the model run for X amount of repeats
for i in range (repeats):
    print()
    loopCount += 1
    start = time.time()
    plot = evolution (lifeSpanT, generationsT, startupCostT, sampleCostT, ExpDistributionShapeT, scoopedCostT, negativeResultCostT, limit)
    end = time.time() #Display time used per repeat
    print("Minutes:", np.round((end - start)/60,2))

# Saving all descriptive statistics in a .npz file, that can be opened with NumPy
finalAP = plot[0]
finalRWP = plot[1]
finalSS = plot[2]
finalPN = plot[3]     
combinedPO = np.array(combinedPO).reshape(repeats, generationsT) 
combinedQW = np.array(combinedQW).reshape(repeats, generationsT)
combinedSS = np.array(combinedSS).reshape(repeats, generationsT)
combinedAP = np.array(combinedAP).reshape(repeats, generationsT)
combinedRWP = np.array(combinedRWP).reshape(repeats, generationsT)
combinedAA = np.array(combinedAA).reshape(repeats, generationsT)
combinedDRAW = np.array(combinedDRAW).reshape(repeats, generationsT)
combinedPUB = np.array(combinedPUB).reshape(repeats, generationsT)
TP = np.array(TP).reshape(repeats, generationsT)
TPP = np.array(TPP).reshape(repeats, generationsT)
FP = np.array(FP).reshape(repeats, generationsT)
FPP = np.array(FPP).reshape(repeats, generationsT)
TN = np.array(TN).reshape(repeats, generationsT)
TNP = np.array(TNP).reshape(repeats, generationsT)
FN = np.array(FN).reshape(repeats, generationsT)
FNP = np.array(FNP).reshape(repeats, generationsT)

name = "BaseOne"
np.savez( name + '.npz', 
finalAP = finalAP,
finalRWP = finalRWP,
finalSS = finalSS,
finalPN = finalPN, 
combinedPO = combinedPO, 
combinedQW = combinedQW,
combinedSS = combinedSS,
combinedAP = combinedAP,
combinedRWP = combinedRWP, 
combinedAA = combinedAA,
combinedDRAW = combinedDRAW,
combinedPUB = combinedPUB,
TP = TP,
TPP = TPP,
FP = FP,
FPP = FPP,
TN = TN,
TNP = TNP,
FN = FN,
FNP = FNP)


