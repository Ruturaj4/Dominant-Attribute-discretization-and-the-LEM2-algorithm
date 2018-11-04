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

def scanFile(inputFile):
    r = pd.DataFrame(inputFile[1:], columns=inputFile[0])
    print(r)
    print(r["GLS100"])

def main():
    inputFile = readFile()
    scanFile(inputFile)

if __name__=="__main__":
    main()
