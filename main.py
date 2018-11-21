# Author - Ruturaj Kiran Vaidya
# This is the main file

import re
import pandas as pd
from math import log
from collections import Counter
import numpy as np

# Debugging
import sys

# For error handling
import os.path

# For tab completion
import readline
readline.parse_and_bind("tab: complete")


# Attribute class contains dominant attribute and a cutpoint
class attr:
    def __init__(self, dominant_attribute = None, cutpoint = 0.0, g_cutpoints = {}, g_count = 0):
        self.dominant_attribute = dominant_attribute
        self.cutpoint = cutpoint
        self.g_cutpoints = g_cutpoints
        self.g_count = g_count

# Class keeps consistency matrix
class ConsistencyMatrix:
    def __init__(self, consistency = True, inconsistent_cases = []):
        self.consistency = consistency
        self.inconsistent_cases = inconsistent_cases

# This function reads the file and deletes some unwanted data
def readFile():
    while True:
        f = input("Please enter the name of the input file: ")
        if(os.path.exists(f)):
            break
        else:
            print("Check your path!")
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

# Gives a nice dictionary of attributes and cases
def avblock(total):
    dic = {}
    for key in total:
        set = ()
        for value in total[key]:
            set = (key, value)
            dic[set] = total[key][value]
    return dic

# This function calculates and then returns avblock{'Weight': {'Medium': [1, 4], 'High': [0, 3, 5],
# 'Low': [2]}, 'Noise': {'low': [0, 1, 3, 4], 'medium': [2, 5]}, 'Comfort':
# {'low': [0], 'high': [1, 2, 3, 4, 5]}}
def each_attribute(a):
    total = {}
    print(a)
    for i in range(len(list(a))):
        attribute = {}
        print(a[list(a)[i]])
        print(a[list(a)[i]].unique())
        for value in a[list(a)[i]].unique():
            attribute[value] = (a.index[a[list(a)[i]] == value].tolist())
        total[list(a)[i]] = attribute
    return avblock(total)

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
    return ent

# This is a pass which returns the class object
def value_pass(r, alld, a, d, attribute, decision, num_columns):
    #Setting smallest entropy value infinit
    smallest_entropy = float("inf")
    for column in num_columns:
        ent = 0.0
        val_decisions = all_decisions(r.loc[:,[column]])
        #print(val_decisions)
        if len(val_decisions) == 1:
            continue
        # List of Unique values in the column
        val_unique = unique_values(r.loc[:,[column]]).tolist()
        temp = {}
        #caluculating the list of all decisions
        all_decision = alld.iloc[:,0].tolist()
        #print(all_decision) #- list of quality
        #print(alld) #- table of quality
        #print(val_unique) # - [4.7, 4.5, 4.3] - Unique values in the column
        # Here it finds the entropy
        for i in range(len(val_unique)):
            temp_decision = []
            for j in val_decisions[i]:
                temp_decision.append(all_decision[j])
            # Calculating H(D|A) value i.e. entropy value
            ent += entropy(len(r.index), temp_decision)
            temp.update({val_unique[i]:temp_decision})

        if (ent < smallest_entropy):
            smallest_entropy = ent
            # Dominant attribute
            dominant_attr = column
            # Unique value list
            dom_val_unique = val_unique
            # list of all decisions in the column
            dom_val_decisions = val_decisions
    #print(dom_val_unique)
    #print(dom_val_decisions)
    # Dictionary of values and frequecy
    dom_val_dic = dict(zip(dom_val_unique, dom_val_decisions))
    # Calculating cutpoints
    x = np.array(sorted(dom_val_unique))
    cutpoints = ((x[1:] + x[:-1]) / 2).tolist()
    #print(cutpoints) # - Contains all the cutpoints
    smallest_entropy = float("inf")
    for cut in cutpoints:
        ent = 0.0
        cut_vals = []
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
            #print(ent)
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

# Create a global dataset
dataset = pd.DataFrame()
# This function gives descritized dataset
def descritized_dataset(r, a, ob):
    global dataset
    decision = list(r)[-1]
    col = ob.dominant_attribute
    ob_columns = []
    for column in r.iloc[:, :-1]:
        if r[column].dtype.kind not in "bifc":
            ob_columns.append(column)
    table = r[[col, decision]]
    table = (pd.concat([r[col], r[ob_columns], r[decision]], axis=1))
    #print(table)
    # List of all the values
    x = table[col].values.tolist()
    #print(ob.g_cutpoints)
    #print(ob.dominant_attribute)
    if len(ob.g_cutpoints[ob.dominant_attribute]) == 1:
        lower = str(min(x))+".."+str(ob.cutpoint)
        upper = str(ob.cutpoint)+".."+str(max(x))
        descritized_list = []
        for i in x:
            if i < ob.cutpoint:
                descritized_list.append(lower)
            else:
                descritized_list.append(upper)
        table[[col, decision]] = table[[col,decision]].replace(x, descritized_list)
        table = pd.concat([dataset.iloc[:,:-1].copy(), table],axis=1)
    else:
        descritized_list = []
        lower = str(min(x))+".."+str(min(ob.g_cutpoints[ob.dominant_attribute]))
        upper = str(max(ob.g_cutpoints[ob.dominant_attribute]))+".."+str(max(x))
        #print(lower)
        #print(upper)
        for i in x:
            #print(i)
            if i < min(ob.g_cutpoints[ob.dominant_attribute]):
                descritized_list.append(lower)
            elif i > max(ob.g_cutpoints[ob.dominant_attribute]):
                descritized_list.append(upper)
            else:
                for j in range(len(ob.g_cutpoints[ob.dominant_attribute])):
                    if i < ob.g_cutpoints[ob.dominant_attribute][j]:
                        descritized_list.append(str(ob.g_cutpoints[ob.dominant_attribute][j-1]) + ".." + str(ob.g_cutpoints[ob.dominant_attribute][j]))
                        break
        new_df = pd.DataFrame({col:descritized_list})
        #print(dataset)
        #print(descritized_list)
        dataset[col] = descritized_list
        table = dataset.copy()
        print(table)
        #table[[col, decision]] = table[[col,decision]].replace(x,descritized_list)
        #table = pd.concat([dataset.iloc[:,:-1], table],axis=1)
    #print(descritized_list)
    print(table)
    #df_num = table.iloc[:,0]
    #print(df_num)
    #sys.exit()
    print(table)
    return table

# Computes attributes
def compute_a(r):
    return r.iloc[:, :-1].astype(object)

# Computes decisions
def compute_d(r):
    return r.iloc[:, -1:].astype(object)

# Helper function
def dpass(r, all_d):
    a = compute_a(r.copy())
    attribute = all_attributes(a)
    d = compute_d(r.copy())
    decision = all_decisions(d)
    # Descritize using dominant attribute approach
    num_columns = []
    # Considering only numeric columns
    for column in r.iloc[:, :-1]:
        if r[column].dtype.kind in "bifc":
            num_columns.append(column)
    print("num columns:")
    print(num_columns)
    return value_pass(r, all_d, a, d, attribute, decision, num_columns) 

def descritize(r):
    # Let's compute a* and d*
    a = compute_a(r.copy())
    attribute = all_attributes(a)
    d = compute_d(r.copy())
    decision = all_decisions(d)
    #print(r)
    
    # rc is the copy
    rc = r.copy()
    counter = 0
    #dataset = pd.DataFrame()
    # Let's leave the consistency check for now
    while True:
        counter += 1
        #if counter == 2:
        #    break
        # Pass - dominant attribute and cutpoints
        ob = dpass(rc.copy(), d)
        #print(rc)
        # Descritized attribute table using cutpoints
        global dataset
        dataset = (descritized_dataset(r.copy(), a, ob)).copy()
        print("This is the final dataset")
        dataset = dataset.loc[:,~dataset.columns.duplicated()]
        print(dataset)
        decs = False
        print(len(r.columns))
        if(len(dataset.columns)==len(r.columns)):
            print("yes")
            with open("myfile.disc", "wb") as f:
                np.save(f, dataset)
            # Saving the descritized dataset into pickle file
            return dataset.copy()
            break
        # Find consistency
        con = consistency(all_attributes(dataset.iloc[:, :-1]),all_decisions(dataset.iloc[:, -1:]))
        inconsistent_case = con.inconsistent_cases
        if (con.consistency == True):
            # Saving the descritized dataset into pickle file
            with open("myfile.disc", "wb") as f:
                np.save(f, dataset)
            return dataset.copy()
            break

        # Get the inconsistent cases
        small = (min(inconsistent_case, key=len))
        inconsistent_case = r.loc[list(small)].copy()
        
        # Now let's change our rc with inconsistent_case
        rc = inconsistent_case.copy()
        print(rc)

class Selection:
    def __init__(self, selection = {}, index = {}, total = {}, key = {}):
        self.selection = selection
        self.index = index
        self.total = total
        self.key = key

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

selectedj = []
selectedr = []

def casecal(avpairs, v, intersections, selob):
    lowest_case_len = float("inf")
    #print("inter")
    #print(intersections)
    max_len = max([len(i) for i in intersections])
    max_len = ([i for i in intersections if len(i) == max_len])
    #print("max_len")
    #print(max_len)
    for i,j in avpairs.items():
        if len(set(selob.selection)-(set(selob.selection)-set(j))) == len(max(intersections, key=len)):
            #print(j)
            if len(j) < lowest_case_len:
                lowest_case_len = len(j)
                #selob.selection = (set(selob.selection)-(set(selob.selection)-set(j)))
                selob.index = j
                selob.key = i
                #print("Print the index")
                temp = i
    global selectedj
    selectedj.append(temp)
    return selob

def lemcal(avpairs, v, selob):
    # List that maintains intersections
    intersections = []
    for i,j in avpairs.items():
        if j == selob.index:
            intersections.append(set())
        else:
            intersections.append(set(selob.selection)-(set(selob.selection)-set(j)))
    #print(intersections)
    selob = casecal(avpairs, selob.selection, intersections, selob)
    return selob

def calc(lemtable, avp, dpairs):
    for k,v in dpairs.items():
        global selectedj
        selectedj = []
        if not v:
            continue
        print("Calculating for: ")
        print(v)
        selob = Selection(v)
        selob.total = ([i for i in range(len(lemtable.values))])
        #print("Initial total")
        #print(selob.total)
        avpairs = avp
        completed = {}
        lemcount = 0
        while(True):
            #print("Selecting: ")
            #print(selob.selection)
            selob = lemcal(avpairs, v, selob)
            #print(selob.total)
            #print("Intersection")
            #print(selob.total)
            #print(selob.index)
            #print("####")
            avpairs = removekey(avpairs, selob.key)

            selob.total = set(selob.total).intersection(selob.index)
            #print(selob.total)
            #print(str(selob.total)+" is a subset of "+str(v))
            #print(set(selob.total).issubset(set(v)) and set(selob.total))
            #print("###")
            selob.selection = set(selob.total).intersection(set(v))
            #if set(selob.total):
            #print(selob.total)
            if (set(selob.total).issubset(set(v))) and set(selob.total):
                completed = set(completed).union(set(selob.total).intersection(set(v)))
            #print(completed)
            if (set(selob.total).issubset(set(v)) and set(selob.total)):
                strstr = ""
                if set(selob.total) == set(v):
                    if len(selectedj)>0:
                        for p in range(len(selectedj)):
                            strstr = strstr + str(selectedj[p])
                            if p != (len(selectedj)-1):
                                strstr = strstr + " & "
                    else:
                        strstr = strstr + str(selectedj)
                        print(selectedj)
                    strstr = strstr + " -> " + str(k)
                    print(strstr)
                    print(selob.index)
                    with open("test.r", "a") as f:
                        f.write(strstr)
                        f.write("\n")
                    break
                elif (set(completed) == set(v)):
                    if len(selectedj)>0:
                        for p in range(len(selectedj)):
                            strstr = strstr + str(selectedj[p])
                            if p != (len(selectedj)-1):
                                strstr = strstr + " & "
                    else:
                        strstr = strstr + str(selectedj)
                    strstr = strstr + " -> " + str(k)
                    print(strstr)
                    with open("test.r", "a") as f:
                        f.write(strstr)
                        f.write("\n")
                    completed = {}
                    #print(completed)
                    #print(v)
                    #print("Completed!")
                    break
                else:
                    if len(selectedj)>0:
                        for p in range(len(selectedj)):
                            strstr = strstr + str(selectedj[p])
                            if p != (len(selectedj)-1):
                                strstr = strstr + " & "
                    else:
                        strstr = strstr + str(selectedj)
                    strstr = strstr + " -> " + str(k)
                    print(strstr)
                    with open("test.r", "a") as f:
                        f.write(strstr)
                        f.write("\n")
                    selectedj = []
                    #print("Do something and continue")
                    selob = Selection(set(v).difference(set(selob.total)))
                    #print("This is my selection")
                    #print(selob.selection)
                    selob.total = ([i for i in range(len(lemtable.values))])
                    avpairs = avp
                    #print("This is avpairs now")
                    #print(avpairs)
    print("__________________________________________________")
 
def lem2(lemtable):
    a = compute_a(lemtable.copy())
    attribute = all_attributes(a)
    d = compute_d(lemtable.copy())
    decision = all_decisions(d)
    #print("We'll start lem2 algorithm")
    #print("We'll first check if the table is consistent")
    con = consistency(all_attributes(lemtable.iloc[:,:-1]),all_decisions(lemtable.iloc[:, -1:]))
    if (con.consistency):
        avp = each_attribute(a)
        dpairs = each_attribute(d)
        with open("test.r", "w") as f:
            f.write("")
    else:
        avp = each_attribute(a)
        #print("This is d")
        #print(d)
        lx = lowerappx(lemtable, attribute, decision)
        ux = upperappx(lemtable, attribute, decision)
        dpairs = lx
        data = {"Cases" : pd.Series(list(avp.values()), index = avp.keys())}
        lem = pd.DataFrame(data)
        #print(lx)
        print("Starting lower")
        with open("test.r", "w") as f:
            f.write("")
            
        with open("test.r", "a") as f:
            f.write("! Certain rule set\n\n")
        calc(lemtable, avp, dpairs)
        dpairs = ux
        print("Starting upper")
        with open("test.r", "a") as f:
            f.write("\n! Possible rule set\n\n")
        calc(lemtable, avp, dpairs)
        return
    print(lemtable)
    data = {"Cases" : pd.Series(list(avp.values()), index = avp.keys())}
    lem = pd.DataFrame(data)
    #print(lem)
    calc(lemtable, avp, dpairs)
    
def lowerappx(lemtable, attribute, decision):
    #print(lemtable)
    a = lemtable.iloc[:, -1:]
    #print(a)
    total = {}
    for i in range(len(list(a))):
        attributes = {}
        for value in a[list(a)[i]].unique():
            attributes[value] = (a.index[a[list(a)[i]] == value].tolist())
            #print(attributes[value])
            for j in attribute:
                if j.issubset(attributes[value]):
                    continue
                else:
                    attributes[value] = list(set(attributes[value]) - j)
        total[list(a)[i]] = attributes
        #print(attributes)
    return avblock(total)
    

"""
    print("Lower Approximation")
    lx = []
    for i in decision:
        for j in attribute:
            if j.issubset(i):
                continue
            else:
                i = i - j
        if i:
            lx.append(i)
        else:
            lx.append({})
    print(lx)
    return lx
"""
        
def upperappx(lemtable, attribute, decision):
    #print("Upper Approximation")
    a = lemtable.iloc[:, -1:]
    #print(a)
    total = {}
    for i in range(len(list(a))):
        attributes = {}
        for value in a[list(a)[i]].unique():
            temp = []
            attributes[value] = (a.index[a[list(a)[i]] == value].tolist())
            #print(attributes[value])
            for m in attributes[value]:
                for n in attribute:
                    if m in n:
                        temp.extend(n)
            attributes[value] = temp
        total[list(a)[i]] = attributes
    #print("I am here")
    #print(total)
    return avblock(total) 
                        
"""
    
    ux = []

    for i in decision:
        temp = []
        for j in i:
            for k in attribute:
                if j in k:
                    (temp).extend(k)
        ux.append(set(temp))
    return ux

"""

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
        with open("myfile.disc", "wb") as f:
            np.save(f, "")
            dataset = descritize(r)
        lem2(dataset.copy())
    else:
        lem2(r.copy())

def main():
    inputFile = readFile()
    scanFile(inputFile)

if __name__=="__main__":
    try:
        main()
    except:
        print("Please check your input")
        main()
