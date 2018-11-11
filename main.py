# Author - Ruturaj Kiran Vaidya
# This is the main file

import re
import pandas as pd
from math import log

# For tab completion
import readline
readline.parse_and_bind("tab: complete")

class Node:
    def __init__(self, dataval = None):
        self.dataval = dataval
        self.nextval = None

class Linked_list:
    def __init__(self):
        self.headval = None

def readFile():
    f = input("Please enter the name of the input file: ")
    print("You entered: " + f)
    inputFile = []
    with open(f, "r") as filename:
        for line in filename:
            # Remove all the comments
            line = re.sub(re.compile("!.*?\n"), "", line)
            # Remove the <> line
            line = re.sub(re.compile("<[^>]+>"), "", line)
            line = line.split()
            if line != []:
                # Removing "[" and "]" from the first line
                if line[0] == "[" or line[-1] == "]":
                    #line = list(map(int, line))
                    inputFile.append(line[1:-1])
                else:
                    #line = list(map(int, line))
                    inputFile.append(line)
    return inputFile

def each_attribute(a):
    attribute = []
    for i in range(len(list(a))):
        for value in a[list(a)[i]].unique():
            attribute.append(a.index[a[list(a)[i]] == value].tolist())
    # Returns something like - [[0, 1, 2], [3, 4], [5], [0, 1, 3, 4], [2, 5]]
    return attribute

def unique_values(d):
    return (d[list(d)[0]].unique())

def all_decisions(d):
    # temp maintains decision
    decision = []
    unique = unique_values(d)
    #print(unique)
    for value in unique:
        decision.append(set(d.index[d[list(d)[0]] == value].tolist()))
    return decision

def all_attributes(a):
    #print(list(a.index.values)) #Print indexes
    # atribute maintains decision
    attribute = []
    columns = list(a.index.values)
    # A used list for used attributes
    used = []
    for i in range(len(columns)):
        temp = [i]
        if i in used:
            continue
        used.append(i)
        for j in range(i+1 , len(columns)):
            if(a.values[i].tolist() == a.values[j].tolist()):
                temp.append(j)
                used.append(j)
                print(a.values[j])
        attribute.append(set(temp))
    return attribute

def subset(attribute, decision):
    check = []
    for i in attribute:
        for j in decision:
            if i.issubset(j):
                check.append(i)
    if (check == attribute):
        return True
    else:
        return False

def consistency(attribute, decision):
    print(attribute)
    print(decision)
    if subset(attribute, decision):
        print("It is consistent")
        return True
    else:
        print("It is not consistent")
        return False

def entropy(num_rows, val_decisions):
    print("Calculating entropy\n")
    print("Num rows: "+str(num_rows))
    print(val_decisions)
    sizes = [len(x) for x in val_decisions]
    ent = 0.0
    for x in sizes:
        ent += -(x/num_rows)*log((x/num_rows),2)
    print(ent)
    print("Entropy calculated")

def value_pass(r, a, d, attribute, decision):
    # Descritize using dominant attribute approach
    num_columns = []
    # Considering only numeric columns
    for column in r.iloc[:, :-1]:
        if r[column].dtype.kind in "bifc":
            num_columns.append(column)
    print(num_columns)
    for column in num_columns:
        print("Trying: " + column)
        val_decisions = all_decisions(r.loc[:,[column]])
        print(val_decisions)
        # List of Unique values in the column
        val_unique = unique_values(r.loc[:,[column]]).tolist()
        temp = {}
        #caluculating the list of all decisions
        all_decision = d.iloc[:,0].tolist()
        for i in range(len(val_unique)):
            temp_decision = []
            for j in val_decisions[i]:
                temp_decision.append(all_decision[j])
            # Calculating H(D|A) value i.e. entropy value
            entropy(len(r.index), temp_decision)
            temp.update({val_unique[i]:temp_decision})
        print(temp)

def descritize(r):
    # Let's compute a* and d*
    a = r.iloc[:, :-1].astype(object)
    attribute = all_attributes(a)
    d = r.iloc[:, -1:].astype(object)
    decision = all_decisions(d)
    print(r)
    consistency(attribute, decision)
    value_pass(r, a, d, attribute, decision)
    
 
def scanFile(inputFile):
    # Select rows and columns
    r = pd.DataFrame(inputFile[1:], columns=inputFile[0])
    # Convert Each column into a numeric object if possible
    numeric = False
    for column in r.iloc[:, :-1]:
        r[column] = pd.to_numeric(r[column], errors="ignore")
        # Check if the attribute is Numerical
        if r[column].dtype.kind in 'bifc':
            numeric = True
    if (numeric):
        descritize(r)

def main():
    inputFile = readFile()
    scanFile(inputFile)

if __name__=="__main__":
    main()
