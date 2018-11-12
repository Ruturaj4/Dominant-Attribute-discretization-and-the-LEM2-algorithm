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

class attr:
    def __init__(self, dominant_attribute = None, cutpoint = 0.0):
        self.dominant_attribute = dominant_attribute
        self.cutpoint = cutpoint

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

def value_pass(r, a, d, attribute, decision, num_columns):
    #Setting smallest entropy value infinit
    smallest_entropy = float("inf")
    for column in num_columns:
        ent = 0.0
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
        print(luv_list)
        for val in luv_list:
            temp = []
            for i in val:
                temp.append(all_decision[i])
            ent += entropy(len(r.index), temp)
        if ent < smallest_entropy:
            smallest_entropy = ent
            dominant_cutpoint = cut
    print(smallest_entropy)
    print(dominant_cutpoint)
    ob = attr(dominant_attr, dominant_cutpoint)
    return ob

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

def descritize(r):
    # Let's compute a* and d*
    a = r.iloc[:, :-1].astype(object)
    attribute = all_attributes(a)
    d = r.iloc[:, -1:].astype(object)
    decision = all_decisions(d)
    print(r)
    consistency(attribute, decision)

    # Descritize using dominant attribute approach
    num_columns = []
    # Considering only numeric columns
    for column in r.iloc[:, :-1]:
        if r[column].dtype.kind in "bifc":
            num_columns.append(column)
    print(num_columns)

    ob = value_pass(r, a, d, attribute, decision, num_columns)
    print(ob.dominant_attribute)
    print(ob.cutpoint)
    # Now with dominant attribute and cutpoint, drawing a table
    dataset = descritized_dataset(r, a, ob)
    print(dataset)
 
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
