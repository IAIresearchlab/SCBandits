'''
Created on Apr 27, 2018

@author: rafferty
'''
import pandas as pd
import generate_contextual_data
import sys
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from glmnet_python import glmnet
from glmnet_python import glmnetPlot
from glmnet_python import glmnetPrint
from glmnet_python import glmnetCoef
from glmnet_python import glmnetPredict
# from cvglmnet_python import cvglmnet
# from cvglmnetCoef import cvglmnetCoef
# from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict
from sklearn import linear_model
from output_format import *
from ast import literal_eval as make_tuple
import numpy as np
import scipy.stats

VERY_SMALL_SIZE = 6
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', family='serif')          # font family
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=VERY_SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
EST_COLOR = 'gold'
ACTUAL_COLOR = 'navy'
cumulative_reward_scale = 100

def read_input_data(banditOutFile, configFile):
    '''
    Reads in a file output by running a contextual bandit, and returns a dataframe
    where the 0:-2 columns are dummy coded contextual variables, and the -2 column
    is the action (treating actions as flat), and the -1 column is the observed reward.
    '''
    df = pd.read_csv(banditOutFile,header=1)
    
    config = generate_contextual_data.loadConfiguration(configFile)
    # need to get the columns that refer to contextual variables and dummy code them
    structure = generate_contextual_data.ContextualStructure(config)
    numContextualVars = structure.getNumberOfVariables()
    contextualVarCols = df.iloc[:,1:(numContextualVars + 1)]
    dummyCoded = [pd.get_dummies(contextualVarCols.iloc[:,i], prefix='cv'+str(i)).iloc[:,1:] for i in range(numContextualVars)]
    contextualVarDf = pd.concat(dummyCoded, axis=1)
    # then we'll add in the action and the response variable
    xyDf = pd.concat([contextualVarDf, df.loc[:,H_ALGO_ACTION], df.loc[:,H_ALGO_OBSERVED_REWARD]], axis=1)
#     print(xyDf.head())
    return xyDf


def getInteractionHeaders(config):
    numExperiments = len(config["conditions"])
    # number of contextual coefficients per variable: 1 fewer than number of options for categorical, exactly 1 for continuous (denoted -1) 
    numCoeffPerContextualVar = [x - 1 if x > 0 else 1 for x in config["contextualStructure"]]  
    interactionHeaders = []
    for i in range(numExperiments):
        numConditionsInExperiment = len(config["conditions"][i])
        for j in range(numConditionsInExperiment):
            interactionHeaders.append(getInterceptHeaderForExpCondition(i,j))
            for k in range(len(numCoeffPerContextualVar)):
                for l in range(numCoeffPerContextualVar[k]):
                    header = getHeaderForCV(l+1, k, i, j)
                    interactionHeaders.append(header)
    return interactionHeaders
    
def makeActionInteractions(banditOutFile, configFile, useFlat = False):
    '''
    Reads in a file output by running a contextual bandit, and returns a dataframe
    where the 0:-2 columns are dummy coded contextual variables, and the -2 column
    is the action (treating actions as flat), and the -1 column is the observed reward.
    Returns a dataframe where the -1 column is still the reward, but the 0:-1 columns are
    interactions between each contextual variable and the condition for an experiment.
    configFile is used to find out how many experiments there were, and for the filename
    that gives the mapping between action numbers in the outfile (which are flat across
    experiments) and action numbers for each experiment. E.g., action 1 in the outfile
    might correspond to action 0 in experiment 1 and action 1 in experiment 2.
    Future implementation: If useFlat is true, then the flat action structure is used 
    (so we don't separate at all based on the different experiments).
    '''
    df = pd.read_csv(banditOutFile,header=1)
    config = generate_contextual_data.loadConfiguration(configFile)
    actionsToConditionsDict = getActionsToConditionsDictionary(config)

    structure = generate_contextual_data.ContextualStructure(config)
    numContextualVars = structure.getNumberOfVariables()
    interactionHeaders = getInteractionHeaders(config)
#     print(interactionHeaders)
    interactionRows = []
    for rowIndex in range(df.shape[0]):
        action = df.loc[rowIndex,H_ALGO_ACTION]
        conditions = actionsToConditionsDict[action]
        curRow = makeAllZerosRow(interactionHeaders)
        interactionRows.append(curRow)
        for condition, expIndex in zip(conditions, range(len(conditions))):
            # add the intercept term
            interceptHeader = getInterceptHeaderForExpCondition(expIndex, condition)
            curRow[interceptHeader] = 1
            for cvIndex in range(1,numContextualVars + 1):# Second to last column is the action, last column is the outcome variable so is omitted
                header = getHeaderForCV(df.iloc[rowIndex,cvIndex], cvIndex - 1, expIndex, condition)
                if header in interactionHeaders:# won't appear if cv value is 0 because that's baked into the intercept
                    curRow[header] = 1

    
    
    interactionDf = pd.DataFrame(interactionRows, columns = interactionHeaders)
    
#     print(interactionDf.head())
    return pd.concat([interactionDf, df.loc[:,H_ALGO_OBSERVED_REWARD]], axis=1)

def makeActionInteraction(contextualVariableVector, conditions, numContextualVars, interactionHeaders):
    x = makeAllZerosRow(interactionHeaders)
    for condition, expIndex in zip(conditions, range(len(conditions))):
        interceptHeader = getInterceptHeaderForExpCondition(expIndex, condition)
        x[interceptHeader] = 1 # intercept
        for cvValue, cvIndex in zip(contextualVariableVector, range(len(contextualVariableVector))):
            header = getHeaderForCV(cvValue, cvIndex, expIndex, condition)
            if header in interactionHeaders:# won't appear if cv value is 0 because that's baked into the intercept
                x[header] = 1
    xDf = pd.DataFrame([x], columns = interactionHeaders)
    return xDf.iloc[0,:]

def makeAllZerosRow(headers):
    return {header : 0 for header in headers}

def getInterceptHeaderForExpCondition(expIndex, conditionIndex):
    return "intercept_exp" + str(expIndex) + "_condition" + str(conditionIndex)

def getHeaderForCV(cvValue, cvIndex, expIndex, conditionIndex):
    return "cv" +str(cvIndex) + "_value" + str(cvValue) + "_exp"+ str(expIndex) + "_condition" + str(conditionIndex)

def getActionsToConditionsDictionary(config):
    '''
    Returns a dictionary where the keys are actions 0-maxActions and the values are
    a tuple with the action for each condition given this "meta action".
    '''
    actionsToConditionsDict = {}
    actionsToConditionsDf = pd.read_csv(config["conditionsToActionsFile"])
    for i in range(actionsToConditionsDf.shape[0]):
        actionsToConditionsDict[actionsToConditionsDf.iloc[i].actionNumber] = make_tuple(actionsToConditionsDf.iloc[i].conditionVector)
    return actionsToConditionsDict




    
def plotLassoAndActualCoefficients(ax, fittedModel, config):
    # 1 + (# experiments) subplots: coefficients (actual and lasso);  for each experiment, predicted probability of reward and actual
    
    allExperiments = []
    for experiment in config["conditions"]:
        allExperiments += experiment
    coef = np.array(allExperiments).flatten();
    ax.plot(fittedModel.coef_[0], color=EST_COLOR, linewidth=2,
         label='Lasso coeff.')
    ax.plot(coef, '--', color=ACTUAL_COLOR, label='True coeff.')
#     ax.legend()
    tol = .05
    fittedZeroThatShouldBeNonZero = sum([1 for i in range(len(coef)) if abs(coef[i]) >= tol and abs(fittedModel.coef_[0][i]) < tol])
    fittedNonZeroThatShouldBeZero = sum([1 for i in range(len(coef)) if abs(coef[i]) < tol and abs(fittedModel.coef_[0][i]) >= tol])
    avgDiff = np.average(abs(fittedModel.coef_[0] - coef))
    ax.annotate("Extra: " + str(round(fittedNonZeroThatShouldBeZero / len(coef),2)), xy=(-.45, -.35), xycoords='axes fraction')
    ax.annotate("Missing: " + str(round(fittedZeroThatShouldBeNonZero / len(coef),2)), xy=(-.45, -.48), xycoords='axes fraction')
    ax.annotate("Avg diff: " + str(round(avgDiff,2)), xy=(-.45, -.61), xycoords='axes fraction')


def plotProbabilitiesOfReward(fittedModel, config, ax):
    '''
    Plots the fittedModel's estimated probabilities of reward compared to the actual
    probabilitiy of reward for each combination of the contextual variables. Assumes
    that all contextual variables are categorical.
    '''
    structure = generate_contextual_data.ContextualStructure(config)
    numContextualVars = structure.getNumberOfVariables()
    conditionVectors = generate_contextual_data.makeConditionVectorsFromConfig(config)
    contextualVariableCombinations = structure.getAllContextualCombinations()
    interactionHeaders = getInteractionHeaders(config)
    conditions = config["conditions"]
    experiments =[generate_contextual_data.Experiment([generate_contextual_data.Condition(coeff) for coeff in curConditions]) for curConditions in conditions] 
    model = generate_contextual_data.LogisticModel(0, experiments, structure)
    estimatedProbs = []
    actualProbs = [] 
    # 2 series for each combination of contextual variable values (one for the estimated probabilities and one for actual)
    # each series is plotted separately, so we'll look over contextual variable value combinations first, and make
    # a list of the values for each condition combination
    for varValues in contextualVariableCombinations:
        curEstProb = []
        curActualProb = []
        for conditionVector in conditionVectors:
            estProb = fittedModel.predict_proba(makeActionInteraction(varValues,conditionVector, numContextualVars, interactionHeaders).values.reshape(1,-1)).flatten()[1] # get success prob
            curEstProb.append(estProb)
            actualProb = model.getSuccessProbability(varValues, conditionVector)
            curActualProb.append(actualProb)
        estimatedProbs.append(curEstProb)
        actualProbs.append(curActualProb)
    
    # Now we need to do the plotting
    barWidth = 1/(2*len(estimatedProbs) + 1)
    for cvValuesIndex in range(len(estimatedProbs)):
        curEstProb = estimatedProbs[cvValuesIndex]
        xEst = [x + 2*cvValuesIndex*barWidth for x in np.arange(len(curEstProb))] 
        xActual = [x + (2*cvValuesIndex+1)*barWidth for x in np.arange(len(curEstProb))]
        print("CVs: " + str(contextualVariableCombinations[cvValuesIndex]))
        print(estimatedProbs[cvValuesIndex])
        print(actualProbs[cvValuesIndex])

        ax.bar(xEst, curEstProb, width=barWidth, color=EST_COLOR)
        ax.bar(xActual, actualProbs[cvValuesIndex], width=barWidth, color=ACTUAL_COLOR)
    # Add xticks on the middle of the group bars    
    ax.set_xticks([r + len(estimatedProbs)*barWidth for r in range(len(curEstProb))], [str(vector) for vector in conditionVectors])

def calculateJensenShannonDivergence(dist1, dist2):
    avgDist = .5*(dist1 + dist2)
    klDivD1Avg = scipy.stats.entropy(dist1, avgDist)
    klDivD2Avg = scipy.stats.entropy(dist2,avgDist)
    return .5*(klDivD1Avg + klDivD2Avg)

def calculateTotalVariationDistance(dist1, dist2):
    return max(abs(dist1 - dist2))
    
def plotDifferencesInProbabilitiesOfReward(fittedModel, config, ax):
    '''
    Plots the fittedModel's estimated probabilities of reward compared to the actual
    probabilitiy of reward for each combination of the contextual variables. Assumes
    that all contextual variables are categorical.
    '''
    structure = generate_contextual_data.ContextualStructure(config)
    numContextualVars = structure.getNumberOfVariables()
    conditionVectors = generate_contextual_data.makeConditionVectorsFromConfig(config)
    contextualVariableCombinations = structure.getAllContextualCombinations()
    interactionHeaders = getInteractionHeaders(config)
    conditions = config["conditions"]
    experiments =[generate_contextual_data.Experiment([generate_contextual_data.Condition(coeff) for coeff in curConditions]) for curConditions in conditions] 
    model = generate_contextual_data.LogisticModel(0, experiments, structure)
    estimatedProbs = []
    actualProbs = [] 
    tol = .98
    numDistWhereMaxProbIsCorrect = 0
#     jsDivergences = np.zeros(shape=(len(contextualVariableCombinations),1))
#     totalVariationDists = np.zeros(shape=(len(contextualVariableCombinations),1))
    # 2 series for each combination of contextual variable values (one for the estimated probabilities and one for actual)
    # each series is plotted separately, so we'll look over contextual variable value combinations first, and make
    # a list of the values for each condition combination
    for varValues, i in zip(contextualVariableCombinations,range(len(contextualVariableCombinations))):
        curEstProb = []
        curActualProb = []
        for conditionVector in conditionVectors:
            estProb = fittedModel.predict_proba(makeActionInteraction(varValues,conditionVector, numContextualVars, interactionHeaders).values.reshape(1,-1)).flatten()[1] # get success prob
            curEstProb.append(estProb)
            actualProb = model.getSuccessProbability(varValues, conditionVector)
            curActualProb.append(actualProb)
        estimatedProbs.append(curEstProb)
        actualProbs.append(curActualProb)
        # Identify whether maximum estimated prob action is an action with tol of best actual action
        npEstProb = np.asarray(curEstProb)
        npActProb = np.asarray(curActualProb)
        goodEnoughActions = npActProb > np.max(npActProb)*tol
        if any(goodEnoughActions[npEstProb > np.max(npEstProb)*tol]):
            numDistWhereMaxProbIsCorrect += 1
#         else:
#             print("not close")
#         jsDivergences[i] = calculateJensenShannonDivergence(np.asarray(curEstProb), np.asarray(curActualProb))
#         totalVariationDists[i] = calculateTotalVariationDistance(np.asarray(curEstProb), np.asarray(curActualProb))

        
    # Now we need to do the plotting
    barWidth = 1/(len(estimatedProbs) + 1)
    for cvValuesIndex in range(len(estimatedProbs)):
        curEstProb = estimatedProbs[cvValuesIndex]
        curActualProb = actualProbs[cvValuesIndex]
        difference = np.array(curEstProb) - np.array(curActualProb)
        xEst = [x + cvValuesIndex*barWidth for x in np.arange(len(difference))] 
#         xActual = [x + (2*cvValuesIndex+1)*barWidth for x in np.arange(len(curEstProb))]
#         print("CVs: " + str(contextualVariableCombinations[cvValuesIndex]))
#         print(estimatedProbs[cvValuesIndex])
#         print(actualProbs[cvValuesIndex])

        ax.bar(xEst, difference, width=barWidth, color='r')
#         ax.bar(xActual, actualProbs[cvValuesIndex], width=barWidth, color=ACTUAL_COLOR)
    # Add xticks on the middle of the group bars    
#     ax.set_xticks([r + len(estimatedProbs)*barWidth for r in range(len(curEstProb))], [str(vector) for vector in conditionVectors])
    ax.get_xaxis().set_ticks([])
    ax.set_xlabel("Condition and Cont. Var. Value Combos")
    ax.set_ylabel("Est. - Actual")
    
    
    
    # distances
    estimated = np.asarray(estimatedProbs).flatten()
    actual =  np.asarray(actualProbs).flatten()
    
    roundingFigs = 4
    ax.annotate("Pointwise L1: " + str(round(np.average(abs(estimated - actual)),roundingFigs)) + \
                "(" + str(round(np.median(abs(estimated - actual)),roundingFigs)) + ")", 
                xy=(-.4, -.35), xycoords='axes fraction')
    ax.annotate("Euclidean: " + str(round(np.linalg.norm(estimated-actual),roundingFigs)), 
                xy=(-.4, -.48), xycoords='axes fraction')
    ax.annotate("Proportion close to max: " + str(round(numDistWhereMaxProbIsCorrect / len(contextualVariableCombinations),roundingFigs)), 
                xy=(-.4, -.61), xycoords='axes fraction')

def plotResults(configFile, fittedModel):
    config = generate_contextual_data.loadConfiguration(configFile)
    numExperiments = len(config["conditions"])
    fig, axes = plt.subplots(1, 2)
    plotLassoAndActualCoefficients(axes[0], fittedModel, config)
    plotDifferencesInProbabilitiesOfReward(fittedModel, config,axes[1])
    plt.show()


def plotAllResults(configFile):
    config = generate_contextual_data.loadConfiguration(configFile)
    suffixes = ["Random","FactorialArms", "ExpArms"]
    fig, axes = plt.subplots(len(suffixes), 3, sharey='col')
    for suffixIndex in range(len(suffixes)):
        banditOutFile = config["outfilePrefix"] + suffixes[suffixIndex] +".csv"
        interactionDf = makeActionInteractions(banditOutFile, configFile)
        fittedModel = getFittedModel(interactionDf)
        curAxes = axes[suffixIndex]
        curAxes[0].set_ylabel(suffixes[suffixIndex])
        
        plotLassoAndActualCoefficients(curAxes[0], fittedModel, config)
        curAxes[0].set_title("Coefficients")
        if suffixIndex == 0:
            curAxes[0].legend(loc='upper right', bbox_to_anchor=(0, 1.4))
        
        plotDifferencesInProbabilitiesOfReward(fittedModel, config,curAxes[1])
        curAxes[1].set_title("Differences in Success Prob.")
        
        plotCumulativeReward(config["outfilePrefix"] + suffixes[suffixIndex] +".csv", curAxes[2])
        curAxes[2].set_title("Cumulative Reward")
        
        print(suffixes[suffixIndex])
        for experimentIndex in range(len(config["conditions"])):
            print("Experiment", experimentIndex)
            print(countNumSamplesByContextualVariableCombination(banditOutFile, experimentIndex, config))
    plt.subplots_adjust( hspace=0.85,wspace=0.55)
#     plt.show()
    if "regressionOutputFile" in config:
        plt.savefig(config["regressionOutputFile"])
    
def plotCumulativeReward(banditOutFile, ax):
    df = pd.read_csv(banditOutFile,header=1)
    rows = df.shape[0]
    avgCumulativeRewards = [0]
    for i in range(0,rows):
        curRewards = df.loc[:,H_ALGO_OBSERVED_REWARD].iloc[0:i]
        avgCumulativeRewards.append(np.sum(curRewards))

    ax.plot(range(len(avgCumulativeRewards)), avgCumulativeRewards)
    ax.annotate('avg reward = ' + str(avgCumulativeRewards[-1]/rows), xy=(0, -.5), xycoords='axes fraction')
        
def getFittedModel(interactionDf):
    # Code for sklearn's lasso
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, fit_intercept=False)
    clf.fit(interactionDf.iloc[:,0:-1], interactionDf.iloc[:,-1])
    return clf

def countNumSamplesByContextualVariableCombination(banditOutFile, experimentIndex, config):
    df = pd.read_csv(banditOutFile,header=1)
    structure = generate_contextual_data.ContextualStructure(config)
    numContextualVars = structure.getNumberOfVariables()
    numActions = len(config["conditions"][experimentIndex])
    actionsToConditionsDict = getActionsToConditionsDictionary(config)
    contextualVarValuesToActions = [{} for _ in range(numContextualVars)]
    for row in range(df.shape[0]):
        contextualVars = tuple(df.iloc[row,1:(numContextualVars + 1)])
        for i in range(numContextualVars):
            if contextualVars[i] not in contextualVarValuesToActions[i]:
                contextualVarValuesToActions[i][contextualVars[i]] = [0]*numActions
            action = df.iloc[row,:].loc[H_ALGO_ACTION]
            contextualVarValuesToActions[i][contextualVars[i]][actionsToConditionsDict[action][experimentIndex]] += 1
        
    return contextualVarValuesToActions

    
        
def main():
    configFile = sys.argv[1]
    if len(sys.argv) > 2:
        banditOutFile = sys.argv[2]
    #     xyDf = read_input_data(banditOutFile, configFile)
        interactionDf = makeActionInteractions(banditOutFile, configFile)
        interactionDf.to_csv("/Users/rafferty/banditalgorithms/data/contextualBanditData/TestInteractionFile.csv")
        #glmnet does not seem to be installed correctly - not sure why (fortran compilation issues and pip not working)
    #     fit = glmnet(x = xyDf.iloc[:,0:-1].values(), y = xyDf.iloc[:,-1].values(), family = 'binomial')
    #     glmnetPlot(fit, xvar = 'dev', label = True);
    
        # Code for sklearn's lasso
        clf = getFittedModel(interactionDf)
        for i in range(clf.coef_.shape[1]):
            if(abs(clf.coef_[0][i]) > 0.01):
                print(interactionDf.columns[i])
                print(clf.coef_[0][i])
    #     plotLassoAndActualCoefficients(clf, configFile)
        plotResults(configFile,clf)
    else:
        plotAllResults(configFile)
if __name__ == "__main__":
    main()