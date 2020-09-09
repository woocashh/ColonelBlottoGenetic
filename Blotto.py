#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:52:47 2020
@author: lj1019
"""
import random
import numpy as np
import matplotlib.pyplot as plt

# Untility Functions
def reverseTuple(lstOfTuple): 
    return [tup[::-1] for tup in lstOfTuple]    

#______________________________________________________________________________ 
#Different types of the game
#(I)    
def playGame(A, B):
    sumA = 0
    comboA = 0
    
    sumB = 0
    comboB = 0
        
    for i in range(10):
        if A[i] > B[i]:
            sumA += i + 1
            comboA += 1
            comboB = 0
            
            if comboA == 3:
                sumA += sumCombo(i + 1)
                break
        elif A[i] < B[i]:
            sumB += i + 1
            comboB += 1
            comboA = 0
            if comboB == 3:
                sumB += sumCombo(i + 1)
                break
        else:
            comboA = 0
            comboB = 0
            
    return (sumA - sumB)

#(II)    
def playGameWithCastlesEqual(A, B):
    sumA = 0
    comboA = 0
    sumB = 0
    comboB = 0
        
    for i in range(10):
        if A[i] > B[i]:
            sumA += 1
            comboA += 1
            comboB = 0
            
            if comboA == 3:
                sumA += 10 - (i + 1)
                break            
        elif A[i] < B[i]:
            sumB += 1
            comboB += 1
            comboA = 0
            
            if comboB == 3:
                sumB += 10 - (i + 1)
                break
        else:
            comboA = 0
            comboB = 0   
            
    return (sumA - sumB)
  
#(III)    
def playGameWithoutCombos(A, B):
    sumA = 0
    sumB = 0
  
    for i in range(10):
        if A[i] > B[i]:
            sumA += i + 1
        elif A[i] < B[i]:
            sumB += i + 1 
            
    return (sumA)

def sumCombo(x):
    return sum(range(x+1, 11))
#______________________________________________________________________________ 
#Generating entries    
def generateTrials(n, generator, 
                   bias = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    trialList = []
    
    for i in range(n):
        trialList.append(generator(bias))
        
    return trialList

def generateMixedTrials(n, gen1, gen2, proportion):
    trialList = []
    
    for i in range(n):
        if i%(proportion + 1) == 0:
            trialList.append(gen2()) 
        else:
            trialList.append(gen1())
            
    return trialList
#______________________________________________________________________________ 
#Generators    
        
def generateTrial():
    indexes = [0,1,2,3,4,5,6,7,8,9]
    trial = [0,0,0,0,0,0,0,0,0,0]
    armyLeft = 100
    
    for i in range(10):
        indexChosen = random.randint(0,len(indexes) - 1)
        armyAllocated = random.randint(0, armyLeft)
        trial[indexes[indexChosen]] = armyAllocated
        del indexes[indexChosen]
        armyLeft -= armyAllocated
        
    return tuple(trial)

def generateBiasedTrial(
        bias = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    army = 100
    indexes = [0,1,2,3,4,5,6,7,8,9]
    trial = [0,0,0,0,0,0,0,0,0,0]
    
    for i in range(army):
        index = np.random.choice(indexes, p = bias)
        trial[index] += 1
        
    return tuple(trial)

def generateXtremlyBiased3Trial():
    
    army = 100
    indexes = [0,1,2,3,4,5,6,7,8,9]
    trial = [0,0,0,0,0,0,0,0,0,0]
    
    
    for i in range(army):
        index = np.random.choice(
                indexes, 
                p=[0.33, 0.33, 0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        trial[index] += 1
        
    return tuple(trial)

def generateDefBiased3Trial():
    army = 100
    indexes = [0,1,2,3,4,5,6,7,8,9]
    trial = [0,0,0,0,0,0,0,0,0,0]
    
    for i in range(army):
        index = np.random.choice(
                indexes, 
                p=[0.07, 0.07, 0.34, 0.07, 0.07, 0.1, 0.07, 0.07, 0.07, 0.07])
        trial[index] += 1
        
    return tuple(trial)
    
#______________________________________________________________________________ 
#Playing the game
        
def playTrials(n, gen1, gen2, proportion, rules):
    
    trialList = generateMixedTrials(n, gen1, gen2, proportion)    
    ranking = {}
    
    for t in trialList:
        ranking[t] = 0
    
    for a in trialList:
        score = 0
        
        for b in trialList:
            score += rules(a,b)
        
        ranking[a] = score
        
    sortedRanking = sorted(reverseTuple(list(ranking.items())))
    top10 = (sortedRanking[-10:])[::-1]
             
    return top10
    
#______________________________________________________________________________    
# Visuals
def avgTop10BarChart(top10):
    avgTop =[0,0,0,0,0,0,0,0,0,0]
        
    for i in top10:
        for j in range(len(i[1])):
            avgTop[j] += i[1][j]
    for i in range(10):
        avgTop[i] /= 10
            
    plt.bar(['1','2','3','4','5','6','7','8','9','10'], avgTop,)
    axes = plt.gca()
    axes.set_ylim([0,80])
    plt.ylabel("Soliders")
    plt.xlabel("Castle")
    plt.show()
    return
            
#______________________________________________________________________________ 
# Genetic algorithm
POPULATION_SIZE = 1000
GENES = [1,2,3,4,5,6,7,8,9,10]

class Individual(object):
    
    '''
    Class representing individual submission to a game
    '''
    
    def __init__(self, chromosome, trials):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness(trials)
        
        
    def create_genome(self, bias = [0.1,0,1,0.1,0,1,0.1,0,1,0.1,0,1,0.1,0,1]):
        '''
        Create a chrmosome
        '''
        return generateBiasedTrial(bias)
    
    
    def mateEqually(self, partner, newTrial):
        '''
        Mate by taking the average of individual genes
        '''
        child_chromosome = []
        
        for gp1, gp2 in zip(self.chromosome, partner.chromosome):
            #Rounding sorted randomly
            upFirst = random.choice([True, False])
            avg = (gp1 + gp2)/2
            if (avg - int(avg)) != 0:    
                if upFirst:        
                    child_chromosome.append(int(avg) + 1)        
                else:        
                    child_chromosome.append(int(avg))       
                upFirst = not upFirst
            else:
                child_chromosome.append(int(avg))
                
        return Individual(tuple (child_chromosome), newTrial)
    
    def cal_fitness(self, trials):
        '''
        Calculate fitness of an individual
        '''
        score = 0
        for i in trials:
            score += playGame(self.chromosome, i)
            
        return score
#______________________________________________________________________________  
#Driver code
#Producing specialised generation (1st method)        
def main(maxGenerations):
    global POPULATION_SIZE    
    #current generation
    generation = 1
    population = []
    trials = generateTrials(
            2500, 
            generateBiasedTrial, [0.11, 0.05, 0.74, 0.01,0.01, 0, 0, 0.04, 0.03, 0.01]) + generateTrials(
            2500, 
            generateBiasedTrial,  [0.24,0.25,0.30,0.05,0.06,0.01,0.03,0.02,0.0,0.04]) + generateTrials(
            2500, 
            generateBiasedTrial,  [0.01,0.0,0.53,0.19,0.20,0.02,0.02,0.01,0.01,0.01]) + generateTrials(
            2500,
            generateBiasedTrial, [0.27, 0.26, 0.13, 0.10, 0.10, 0.08, 0, 0, 0.06, 0])
    
    for _ in range(POPULATION_SIZE):
        gnome = generateBiasedTrial([0.30, 0, 0, 0.20,0.15,0.23, 0.0, 0.04,0.04,0.04])
        population.append(Individual(gnome, trials))
        
    for i in range(maxGenerations):
        population = sorted(
                population, key = lambda x:x.fitness, reverse = True)
        print(generation)
        top10 = []

        for i in range (10):
            print((population[i].fitness, population[i].chromosome))
        
        for i in range (10):
            top10.append((population[i].fitness, population[i].chromosome))
            
        avgTop10BarChart(top10)
            
        #Create offspring
        new_generation = []
        
        #Elitism
        s = int((10*POPULATION_SIZE)/100) 
        
        #Update trials
        trials = generateTrials(
            2500, 
            generateBiasedTrial, [0.11, 0.05, 0.74, 0.01,0.01, 0, 0, 0.04, 0.03, 0.01]) + generateTrials(
            2500, 
            generateBiasedTrial,  [0.24,0.25,0.30,0.05,0.06,0.01,0.03,0.02,0.0,0.04]) + generateTrials(
            2500, 
            generateBiasedTrial,  [0.01,0.0,0.53,0.19,0.20,0.02,0.02,0.01,0.01,0.01]) + generateTrials(
            2500,
            generateBiasedTrial, [0.27, 0.26, 0.13, 0.10, 0.10, 0.08, 0, 0, 0.06, 0])
            
        for i in population[:s]:            
            new_generation.append(Individual(i.chromosome, trials))
            
        # From 30% of fittest population, Individuals  
        # will mate to produce offspring 
        
        s = int((80*POPULATION_SIZE)/100) 
        for _ in range(s): 
            parent1 = random.choice(population[:30]) 
            parent2 = random.choice(population[:30]) 
            child = parent1.mateEqually(parent2, trials) 
            new_generation.append(child) 
        
        #10% random   
        s = int((10*POPULATION_SIZE)/100) 
        for _ in range(s):
            new_generation.append(Individual(generateTrial(),trials))
            
        population = new_generation
        generation += 1
#______________________________________________________________________________       
# Two competing GAs        
def mainWithCompetition(maxGenerations):
    global POPULATION_SIZE
    #current generation
    generation = 1
    
    populationA = []
    populationB = []
    
    trials = generateTrials(
            1000, 
            generateBiasedTrial,  
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    for _ in range(POPULATION_SIZE):
        gnomeA = generateTrial()
        gnomeB = generateTrial()
        populationA.append(Individual(gnomeA, trials))
        populationB.append(Individual(gnomeB, trials))
        
    for i in range(maxGenerations):
        
        populationA = sorted(populationA, 
                             key = lambda x:x.fitness, reverse = True)
        populationB = sorted(populationB, 
                             key = lambda x:x.fitness, reverse = True)
        
        print("\n")
        print(generation)
        
        top10A = []
        top10B = []
        
        for i in range (10):
            top10A.append((populationA[i].fitness, populationA[i].chromosome))
                        
        
        
        for i in range (10):
            top10B.append((populationB[i].fitness, populationB[i].chromosome))   
         
            
        for i in range (10):
            print((populationA[i].fitness, populationA[i].chromosome))
        avgTop10BarChart(top10A)    
        print("___________________________________")    
        for i in range (10):
            print((populationB[i].fitness, populationB[i].chromosome))
        avgTop10BarChart(top10B)
        
        #Create offspring
        new_generationA = []
        new_generationB = []
        
        #A
        #Elitism
        s = int((10*POPULATION_SIZE)/100) 
        
        #Update trials
        trialsA =  []
        
        for i in populationB:
            trialsA.append(i.chromosome)
        
        for i in populationA[:s]:
            
            new_generationA.append(Individual(i.chromosome, trialsA))
            
        # From 30% of fittest population, Individuals  
        # will mate to produce offspring 
        
        s = int((80*POPULATION_SIZE)/100) 
        for _ in range(s): 
            parent1 = random.choice(populationA[:30]) 
            parent2 = random.choice(populationA[:30]) 
            child = parent1.mateEqually(parent2, trialsA) 
            new_generationA.append(child) 
            
        #10% random    
        s = int((10*POPULATION_SIZE)/100) 
        for _ in range(s):
            new_generationA.append(Individual(generateTrial(),trialsA))
            
        populationA = new_generationA
        
        #B
        #Elitism
        s = int((10*POPULATION_SIZE)/100) 
        
        #Update trials
        trialsB =  []
        
        for i in populationA:
            trialsB.append(i.chromosome)
        
        for i in populationB[:s]:
            
            new_generationB.append(Individual(i.chromosome, trialsB))
            
        # From 30% of fittest population, Individuals  
        # will mate to produce offspring 
        
        s = int((80*POPULATION_SIZE)/100) 
        for _ in range(s): 
            parent1 = random.choice(populationB[:30]) 
            parent2 = random.choice(populationB[:30]) 
            child = parent1.mateEqually(parent2, trialsB) 
            new_generationB.append(child) 
              
         #10% random   
        s = int((10*POPULATION_SIZE)/100) 
        for _ in range(s):
            new_generationB.append(Individual(generateTrial(),trialsB))
            
        populationB = new_generationB
        
        generation += 1 
#______________________________________________________________________________  
        
    
        
    
    


