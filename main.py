# Author - Ruturaj Kiran Vaidya
# This is the main file

import re
import pandas as pd
from math import log
from collections import Counter
import numpy as np

# For tab completion
import readline
readline.parse_and_bind("tab: complete")


# Attribute class contains dominant attribute and a cutpoint
class attr:
    def __init__(self, dominant_attribute = None, cutpoint = 0.0, g_cutpoints = {}):
        self.dominant_attribute = dominant_attribute
        self.cutpoint = cutpoint
        self.g_cutpoints = g_cutpoints

# Class keeps consistency matrix
class ConsistencyMatrix:
    def __init__(self, consistency = True, inconsistent_cases = []):
        self.consistency = consistency
        self.inconsistent_cases = inconsistent_cases

# This function reads the file and deletes some unwanted data
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

# This function returns [[0, 1, 2], [3, 4], [5], [0, 1, 3, 4], [2, 5]]
def each_attribute(a):
    attribute = []
    for i in range(len(list(a))):
        for value in a[list(a)[i]].unique():
            attribute.append(a.index[a[list(a)[i]] == value].tolist())
    # Returns something like - [[0, 1, 2], [3, 4], [5], [0, 1, 3, 4], [2, 5]]
    return attribute

# This function returns unique values - [1.4, 1.8]
def unique_values(d):
    return (d[list(d)[0]].unique())

# This function returns all the decisions - ["Medium", "Low"]
def all_decisions(d):
    # temp maintains decision
    decision = []
    unique = unique_values(d)
    #print(unique)
    for value in unique:
        decision.append(set(d.index[d[list(d)[0]] == value].tolist()))
    return decision

# This function returns all the attributes
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
                #print(a.values[j])
        attribute.append(set(temp))
    return attribute

# Checks if the subset of ....
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

# This function gives conflicting attributes
def conflicting(attribute, decision):
    return [d for d in attribute if not any(d <= a for a in decision)]

def consistency(attribute, decision):
    print(attribute)
    print(decision)
    if subset(attribute, decision):
        print("It is consistent")
        con = ConsistencyMatrix()
        con.consistency = True
        con.inconsistent_cases = []
        return con
    else:
        print("It is not consistent")
        con = ConsistencyMatrix()
        con.consistency = False
        con.inconsistent_cases = conflicting(attribute, decision)
        return con

# This function gives entropy
def entropy(num_rows, val_decisions):
    # List of unique keys
    #print(list(Counter(val_decisions).keys()))
    # Number of unique values
    sizes = list(Counter(val_decisions).values())
    ent = 0.0
    for x in sizes:
        ent += -(x/num_rows)*log((x/len(val_decisions)),2)
    #print("Entropy calculated:")
    #print(ent)
    return ent

# This is a pass which returns the class object
def value_pass(r, alld, a, d, attribute, decision, num_columns):
    #Setting smallest entropy value infinit
    smallest_entropy = float("inf")
    for column in num_columns:
        ent = 0.0
        print("Trying: " + column)
        val_decisions = all_decisions(r.loc[:,[column]])
        #print(val_decisions)
        if len(val_decisions) == 1:
            continue
        # List of Unique values in the column
        val_unique = unique_values(r.loc[:,[column]]).tolist()
        temp = {}
        #caluculating the list of all decisions
        all_decision = alld.iloc[:,0].tolist()
        #print(all_decision)
        #print(alld)
        #print(val_unique)
        for i in range(len(val_unique)):
            temp_decision = []
            for j in val_decisions[i]:
                temp_decision.append(all_decision[j])
            # Calculating H(D|A) value i.e. entropy value
            ent += entropy(len(r.index), temp_decision)
            temp.update({val_unique[i]:temp_decision})
        print("Final Entropy:")
        print(ent)
        #print(val_unique)
        if (ent < smallest_entropy):
            smallest_entropy = ent
            # Dominant attribute
            dominant_attr = column
            # Unique value list
            dom_val_unique = val_unique
            # list of all decisions in the column
            dom_val_decisions = val_decisions
    print(smallest_entropy)
    print(dominant_attr)
    print(dom_val_unique)
    print(dom_val_decisions)
    # Dictionary of values and frequecy
    dom_val_dic = dict(zip(dom_val_unique, dom_val_decisions))
    print(dom_val_dic)
    # Calculating cutpoints
    x = np.array(sorted(dom_val_unique))
    cutpoints = ((x[1:] + x[:-1]) / 2).tolist()
    smallest_entropy = float("inf")
    for cut in cutpoints:
        ent = 0.0
        cut_vals = []
        print(cut)
        #Calculating lower and upper values for the cutpoint
        lower = [x for x in dom_val_unique if x < cut]
        upper = [x for x in dom_val_unique if x > cut]
        lower_val = [dom_val_dic[x] for x in dom_val_dic if x in lower]
        upper_val = [dom_val_dic[x] for x in dom_val_dic if x in upper]
        lv_list = []
        for val in lower_val:
            lv_list.extend(val)
        uv_list = []
        for val in upper_val:
            uv_list.extend(val)
        luv_list = []
        luv_list.append(uv_list)
        luv_list.append(lv_list)
        #print(luv_list)
        for val in luv_list:
            temp = []
            for i in val:
                temp.append(all_decision[i])
            ent += entropy(len(r.index), temp)
        if ent < smallest_entropy:
            smallest_entropy = ent
            dominant_cutpoint = cut
    #print(smallest_entropy)
    #print(dominant_cutpoint)
    ob = attr(dominant_attr, dominant_cutpoint)
    if dominant_attr in ob.g_cutpoints:
        ob.g_cutpoints[dominant_attr].append(dominant_cutpoint)
    else:
        ob.g_cutpoints[dominant_attr] = [dominant_cutpoint]

    return ob

# This function gives descritized dataset
def descritized_dataset(r, a, ob):
    decision = list(r)[-1]
    col = ob.dominant_attribute
    table = r[[col, decision]]
    print(table)
    # List of all the values
    x = table[col].values.tolist()
    lower = str(min(x))+".."+str(ob.cutpoint)
    upper = str(ob.cutpoint)+".."+str(max(x))
    descritized_list = []
    for i in x:
        if i < ob.cutpoint:
            descritized_list.append(lower)
        else:
            descritized_list.append(upper)
    table[[col, decision]] = table[[col,decision]].replace(x, descritized_list)
    print(descritized_list)
    return table

# Computes attributes
def compute_a(r):
    return r.iloc[:, :-1].astype(object)

# Computes decisions
def compute_d(r):
    return r.iloc[:, -1:].astype(object)

# Helper function
def dpass(r, all_d):
    a = compute_a(r)
    attribute = all_attributes(a)
    d = compute_d(r)
    decision = all_decisions(d)
    #consistency(attribute, decision) - Let's leave this for now

    # Descritize using dominant attribute approach
    num_columns = []
    # Considering only numeric columns
    for column in r.iloc[:, :-1]:
        if r[column].dtype.kind in "bifc":
            num_columns.append(column)
    print(num_columns)
    return value_pass(r, all_d, a, d, attribute, decision, num_columns) 

def descritize(r):
    # Let's compute a* and d*
    a = compute_a(r)
    attribute = all_attributes(a)
    d = compute_d(r)
    decision = all_decisions(d)
    print(r)
    
    # rc is the copy
    rc = r
    counter = 0
    dataset = pd.DataFrame()
    # Let's leave the consistency check for now
    while True:
        counter += 1
        # Pass - dominant attribute and cutpoints
        ob = dpass(rc, d)
        print(ob.g_cutpoints)

        if counter == 4:
            break
        # Descritized attribute table using cutpoints
        dataset = pd.concat([dataset.iloc[:,:-1], descritized_dataset(r, a, ob)], axis=1)
        print("This is the final dataset")
        print(dataset)

        # Find consistency
        con = consistency(all_attributes(dataset.iloc[:, :-1]),all_decisions(dataset.iloc[:, -1:]))
        inconsistent_case = con.inconsistent_cases
        if (con.consistency == True):
            print(con.consistency)
            break

        # Get the inconsistent cases
        small = (min(inconsistent_case, key=len))
        inconsistent_case = r.loc[list(small)]
        print(inconsistent_case)
        
        # Now let's change our rc with inconsistent_case
        rc = inconsistent_case

        # For now using a counter logic to break
        #if counter == 3:
        #    break

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
