'''
Module for developing simulated datasets where contextual variables have an impact on arm choices.
Supports simultaneous factorial experiments - two separate choices made at once.
Currently does not support interactions among contextual variables or interactions between
the experiments.


Created on Apr 16, 2018

@author: rafferty

'''
import numpy as np
import itertools
import sys
import csv
import json
import thompson_policy
import logistic_regression
from bandit_data_format import HEADER_ACTUALREWARD, HEADER_OPTIMALACTION, HEADER_TRUEPROB
class Experiment:
    
    def __init__(self, conditions):
        '''
        conditions: iterable of Condition instances
        '''
        self.conditions = conditions
        
    def getValue(self, contextualVariables, conditionAssignment):
        return self.conditions[conditionAssignment].getValue(contextualVariables)
    
class Condition:
    def __init__(self, contextualCoefficients):
        '''
        contextualCoefficients: iterable with an intercept at index 0 and each
        subsequent coefficient corresponds to a contextual variable (either with a particular
        dummy coding or one value for continuous) 
        '''
        self.coeff = contextualCoefficients
        
    def getValue(self, contextualVariables):

        return np.inner(self.coeff, [1] + contextualVariables)# 1 is for the intercept

class LinearModel:
    
    def __init__(self, intercept, experiments, contextualStructure):
        self.intercept = intercept
        self.experiments = experiments
        self.contextualStructure = contextualStructure

    def simulateReward(self, contextualVariableValues, conditionVector, noiseModel=np.random.normal):
        dummyCodedVariableValues = self.contextualStructure.getDummyCodedVersion(contextualVariableValues) 
        outcome = self.intercept
        for experiment, condition in zip(self.experiments, conditionVector):
            outcome += experiment.getValue(dummyCodedVariableValues, condition)
        
        # Add random noise
        outcome += noiseModel()
        return outcome
    
class LogisticModel:
    
    def __init__(self, intercept, experiments, contextualStructure):
        self.intercept = intercept
        self.experiments = experiments
        self.contextualStructure = contextualStructure

    def getSuccessProbability(self, contextualVariableValues, conditionVector):
        dummyCodedVariableValues = self.contextualStructure.getDummyCodedVersion(contextualVariableValues) 
        outcome = self.intercept
        for experiment, condition in zip(self.experiments, conditionVector):
            outcome += experiment.getValue(dummyCodedVariableValues, condition)
        
        # Calculate probability of success
        exponential = np.exp(outcome)
        successProb = exponential / (1 + exponential)
        
        return successProb
    
    def simulateReward(self, contextualVariableValues, conditionVector):
        successProb = self.getSuccessProbability(contextualVariableValues, conditionVector)
        success = np.random.binomial(1, successProb)
        return success

class ContextualStructure:
    
    def __init__(self, config):
        '''
        config is json and must declare:
        contextualStructure: iterable where each entry is a contextual variable.
        entries > 0 indicate a categorical variable with that number of options.
        entries = -1 indicate a continuous variable.
        May also declare: contextualDependencies, contextualGenerationProcess which impact generateContext.
        contextualDependencies: dictionary with string keys corresponding to any variables that are dependent
        on the value of other variables. We require that the original structure be partially ordered such that
        a contextual variable that is dependent on other variables comes later in the structure than those it's
        dependent on. (So variable 0 is always lacking dependencies on other variables.)
        contextualGenerationProcess: dictionary with string keys, where if a variable appears as a key,
        the value specifies the distribution for generating context. (currently only categorical distributions
        are allowed.) If the variable is not specified, its value is generated uniformly at random.  If the variable
        is dependent on another variable, then the keys are of the form "i-v" where i is the index of the variable
        and v is the value. If it's dependent on multiple variables, these are comma separated (e.g., "i-v1,j-v2"). 
        '''
        self.contextualStructure = config["contextualStructure"]
        self.contextualDependencies = {}
        self.contextualGenerationProcess = {}
        if "contextualDependencies" in config:
            self.contextualDependencies = config["contextualDependencies"]
        
        if "contextualGenerationProcess" in config:
            self.contextualGenerationProcess = config["contextualGenerationProcess"]
            #Normalize distributions so we don't have to do it every time
            for key in self.contextualGenerationProcess:
                if not isinstance(self.contextualGenerationProcess[key], dict):
                    # Have the distribution here
                    self.contextualGenerationProcess[key] = self.__getNormalizedDistribution(self.contextualGenerationProcess[key])
                else:
                    for innerKey in self.contextualGenerationProcess[key]:
                        self.contextualGenerationProcess[key][innerKey] = self.__getNormalizedDistribution(self.contextualGenerationProcess[key][innerKey])

    
    def __getNormalizedDistribution(self, dist):
        total = sum(dist)
        normDist = [val / total for val in dist]
        return normDist
    
    def generateContext(self):
        '''
        If contextualGenerationProcess exists and is non-empty (specified on construction), then generates
        contextual variable values according to the probabilities specified there. Otherwise, generates
        contextual variables uniformly at random.
        '''
        if len(self.contextualGenerationProcess) == 0:
            return self.generateUniformlyAtRandom()
        else:
            variableValues = []
            for varIndex in range(len(self.contextualStructure)):
                generatingDistribution = None
                stringVarIndex = str(varIndex)
                if stringVarIndex not in self.contextualDependencies:
                    # Generate this without dependence on other values
                    if stringVarIndex in self.contextualGenerationProcess:
                        generatingDistribution = self.contextualGenerationProcess[stringVarIndex]
                else:
                    # Need to find the values for ones it's dependent on
                    dependentVars = self.contextualDependencies[stringVarIndex]
#                     dependentVars.sort()
                    depValueKey = [dependentVar + "-" + str(variableValues[int(dependentVar)]) for dependentVar in dependentVars]
                    generatingDistribution = self.contextualGenerationProcess[stringVarIndex][",".join(depValueKey)]
                
                if generatingDistribution == None:
                    variableValues.append(self.__generateSingleVariableUniformlyAtRandom(self.contextualStructure[varIndex]))
                else:
                    # Assumes a categorical distribution, summing to 1 handled in constructor
                    variableValues.append(np.random.choice(len(generatingDistribution), p=generatingDistribution))
            return variableValues
    
    def generateUniformlyAtRandom(self):
        '''
        Generates a list of contextual variables (non-dummy-coded) selected uniformly at random from their range.
        If a variable is continuous, assumes the range is [0,1].
        '''
        variableValues = [self.__generateSingleVariableUniformlyAtRandom(varStructure) for varStructure in self.contextualStructure]            
        return variableValues
    
    def __generateSingleVariableUniformlyAtRandom(self, varStructure):
        if varStructure == -1:
            return np.random.uniform(0,1)
        else:
            return np.random.randint(varStructure)
    
    def getAllContextualCombinations(self):
        '''
        Returns a list of all possible combinations of contextual variable values.
        Assumes all contextual variables are categorical.
        '''
        if any(np.array(self.contextualStructure) == -1):
            print("Error: can't get all contextual variable combinations if some variables are continuous.")
        
        return [item for item in itertools.product(*[range(i) for i in self.contextualStructure])]
        
    def getNumberOfVariables(self):
        return len(self.contextualStructure)
    
    def getDummyCodedVersion(self, contextualVariableValues):
        '''
        Converts from iterable contextualVariableValues where categorical variables with
        n options take on values (0,...,n-1) to a dummy coded version where there would
        be n values, all 0/1, and the ith value is 1 for value 1,...,n-1 and no value
        is 1 for the 0th value.
        '''
        dummyCodedVariables = []
        for i in range(len(contextualVariableValues)):
            if self.contextualStructure[i] == -1:
                # Continuous variable - no dummy coding
                dummyCodedVariables.append(contextualVariableValues[i])
            else:
                dummyCoding = np.zeros(self.contextualStructure[i] - 1)
                if contextualVariableValues[i] != 0:
                    dummyCoding[contextualVariableValues[i] - 1] = 1
                dummyCodedVariables.extend(dummyCoding)
        return dummyCodedVariables

    def get_context(self, row, includeIntercept=True):
        contextNames = ['contextualVariable' + str(i) for i in range(self.getNumberOfVariables())]
        contextVector = []
        if includeIntercept:
            contextVector.append(1)
        
        contextualVariableValues = []
        for contextName, i in zip(contextNames, range(len(contextNames))):
            if self.contextualStructure[i] == -1:
                #Continuous variable
                contextualVariableValues.append(float(row[contextName]))
            else:
                contextualVariableValues.append(int(row[contextName]))
        contextVector.extend(self.getDummyCodedVersion(contextualVariableValues))
        return contextVector
    
def generateContextualRewardFile(config):
    numStudents = config["numStudents"]
    # Contextual variables: 1 binary, 2 trinary (so with dummy coding, we'll have two coefficients here for each)
    ## 5 total contextual coefficients, plus an intercept, for each condition
    structure = ContextualStructure(config)

    conditions = config["conditions"]
    numConditions = [len(curConditions) for curConditions in conditions]
    experiments =[Experiment([Condition(coeff) for coeff in curConditions]) for curConditions in conditions] 
    model = LogisticModel(0, experiments, structure)
    
    # Now simulate some data. We'll assume that the contextual variables are distributed
    # uniformly at random.
    conditionVectors = makeConditionVectors(numConditions)
    with open(config["conditionsToActionsFile"], 'w',encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(["actionNumber", "conditionVector"])
        for v, a in zip(conditionVectors, range(len(conditionVectors))):
            writer.writerow([a+1, str(v)])


    with open(config["rewardFile"], 'w',encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['n'] + ['contextualVariable' + str(i) for i in range(structure.getNumberOfVariables())] 
                        + [HEADER_ACTUALREWARD.format(a + 1) for a in range(len(conditionVectors))] + [HEADER_OPTIMALACTION]\
                        + [HEADER_TRUEPROB.format(a + 1) for a in range(len(conditionVectors))])
        

        for n in range(numStudents):
            contextualVariables = structure.generateContext()
            rewards = [model.simulateReward(contextualVariables, conditionVector) for 
                       conditionVector in conditionVectors]
            
            successProbs = [model.getSuccessProbability(contextualVariables, conditionVector)
                            for conditionVector in conditionVectors]
            optimalActions = np.argwhere(successProbs == np.amax(successProbs)).flatten() + 1
            optimalActionsString = ';'.join(str(index) for index in optimalActions)
            writer.writerow([n] + contextualVariables + rewards + [optimalActionsString] + successProbs)
    return structure
            
            
def runThompsonContextualBandit(config):
    '''
    Just for experimenting with running the contextual bandit code.
    '''
    # Initial experiments - treat this as a factorial - one choice at each timestep but lots of actions
    # First, get the right models to include regression
    conditions = config["conditions"]
    numConditions = [len(curConditions) for curConditions in conditions]
    conditionVectors = makeConditionVectors(numConditions)
    models = [logistic_regression.RLogReg(D=6, Lambda=1) for _ in range(len(conditionVectors))]
    # Then, run thompson sampling
    get_context = lambda row: contextualStructure.get_context(row, includeIntercept=True)
    chosen_actions, models = thompson_policy.calculate_thompson_single_bandit(config["rewardFile"], 
                                                                              len(conditionVectors),
                                                                              config["outfilePrefix"] + "ExpArms.csv",
                                                                              models,
                                                                              get_context=get_context)
    print([model.w for model in models])

def makeConditionVectors(numConditions):
    '''
    Creates a list of all combinations of condition assignments given the list numConditions
    that lists the number of conditions in each experiment
    '''
    return [vector for vector in itertools.product(*[range(x) for x in numConditions])]

def makeConditionVectorsFromConfig(config):
    '''
    Creates a list of all combinations of condition assignments given the list numConditions
    that lists the number of conditions in each experiment
    '''
    conditions = config["conditions"]
    numConditions = [len(curConditions) for curConditions in conditions]
    return [vector for vector in itertools.product(*[range(x) for x in numConditions])]

def runUniformRandomBandit(config):
    conditions = config["conditions"]
    numConditions = [len(curConditions) for curConditions in conditions]
    conditionVectors = makeConditionVectors(numConditions)
    models = [logistic_regression.RLogReg(D=6, Lambda=1) for _ in range(len(conditionVectors))]
    # Then, run uniform random sampling
    get_context = lambda row: contextualStructure.get_context(row, includeIntercept=True)
    # epsilon is 1 below, so an action is always chosen uniformly at random
    chosen_actions, models = thompson_policy.calculate_thompson_single_bandit(config["rewardFile"], 
                                                                              len(conditionVectors),
                                                                              config["outfilePrefix"] + "Random.csv",
                                                                              models,
                                                                              epsilon = 1,
                                                                              get_context=get_context)
    
def runFactorialThompsonContextualBandit(config):
    '''
    Just for experimenting with running the contextual bandit code. This uses the factorial version, so
    the actions are chosen separately for each experiment.
    '''
    # Initial experiments - treat this as a factorial - one choice at each timestep but lots of actions
    # First, get the right models to include regression
    conditions = config["conditions"]
    numConditions = [len(curConditions) for curConditions in conditions]
    conditionVectors = makeConditionVectors(numConditions)
    conditionsToActionIndex = {tuple(vector):i for (i, vector) in zip(range(len(conditionVectors)), conditionVectors)}
    models = [[logistic_regression.RLogReg(D=6, Lambda=1)  for _ in range(curNum)] for curNum in numConditions]
#     models = [logistic_regression.RLogReg(D=6, Lambda=1) for _ in range(len(conditionVectors))]
    # Then, run thompson sampling
    get_context = lambda row: contextualStructure.get_context(row, includeIntercept=True)
    chosen_actions, models = thompson_policy.calculate_thompson_single_bandit_factorial(config["rewardFile"], 
                                                                                        numConditions, 
                                                                                        config["outfilePrefix"] + "FactorialArms.csv", 
                                                                                        models, 
                                                                                        conditionsToActionIndex, 
                                                                                        get_context=get_context)
    
    for curModels in models:
        print([model.w for model in curModels])

def loadConfiguration(configurationFile):
    '''
    Returns the JSON object stored in configuration file.
    Used to setup the structure of the generating reward function.
    '''
    with open(configurationFile) as jsonFile:
        config = json.load(jsonFile)
    
    if "subNumStudents" in config and config["subNumStudents"]:
        # Substitute the number of students into the filenames
        for key in config:
            if "file" in key or "File" in key:
                config[key] = config[key].replace("NUMSTUDENTS", str(config["numStudents"]))
    return config
    
if __name__ == "__main__":
    configurationFile = sys.argv[1]
    config = loadConfiguration(configurationFile)
    contextualStructure = generateContextualRewardFile(config)
    runThompsonContextualBandit(config)
    runFactorialThompsonContextualBandit(config)
    runUniformRandomBandit(config)

        