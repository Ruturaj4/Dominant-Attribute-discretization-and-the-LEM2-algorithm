# Author - Ruturaj Kiran Vaidya
# This is the main file

import re
import pandas as pd

# For tab completion
import readline
readline.parse_and_bind("tab: complete")

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

def descritize(r):
    print(r)

def scanFile(inputFile):
    # Select rows and columns
    r = pd.DataFrame(inputFile[1:], columns=inputFile[0])
    # Convert Each column into a numeric object if possible
    numeric = False
    for column in r:
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
