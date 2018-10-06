import math
import csv
import operator
import sys

class Attribute(object):
    def __init__(self, index):
        self.index = index
        self.classes = list()

# Recursive class to hold tree
class Node(object):
    def __init__(self):
        self.attribute = None
        self.label = None
        self.nodes = {}

attributes = {}
attributeNames = []

# Splits dataset into partitions by the value of an attribute
def PartitionData( data, splitAttrName ):
    partitions = {}
    splitAttr = attributes[splitAttrName]

    for label in splitAttr.classes:
        partitions[label] = list()

    for row in data:
        partitions[row[splitAttr.index]].append(row)

    return partitions

# Finds the most common target class in a partition, for this dataset it could be 'e' or 'p'
def MostCommonClass(data, targetAttrName):
    classes = {}
    targetAttr = attributes[targetAttrName]
    
    for row in data:
        if row[targetAttr.index] not in classes:
            classes[row[targetAttr.index]] = 0

        classes[row[targetAttr.index]] += 1

    if classes:
        return max(classes.items(), key=operator.itemgetter(1))[0]


def CalculateEntropy(data, targetAttr):
    partitions = PartitionData(data, targetAttr)
    entropy = 0

    for key, partition in partitions.items():
        partitionProbability = len(partition) / len(data)
        if partitionProbability != 0:
            entropy -= partitionProbability * math.log2( partitionProbability )

    return entropy

def CalculateAverageEntropy( data, targetAttr, splitAttr ):
    dataRowCount = len(data)
    partitions = PartitionData( data, splitAttr )
    averageEntropy = 0

    # Totals the average entropy for all partitions
    for key, partition in partitions.items():
        partitionCount = len(partition)
        if partitionCount == 0:
            continue
        partitionEntropy = CalculateEntropy( partition, targetAttr )

        averageEntropy += partitionCount / dataRowCount * partitionEntropy
    return averageEntropy, partitions

def id3(data, targetAttr, remainingAttr ):
    node = Node()
    maxInfoGain = None
    maxInfoGainAttr = None
    maxInfoGainPartitions = None

    # Stop if we have run out of attributes to split on
    if len(remainingAttr) == 0:
        node.label = MostCommonClass(data, targetAttr)
        return node

    # Find entropy for the whole dataset
    ent = CalculateEntropy( data, targetAttr )
  
    # Iterate through attributes and selects the one with the highest information gain
    for attribute in remainingAttr:
        attribute_ent, partitions = CalculateAverageEntropy(data, targetAttr, attribute)
        infoGain = ent - attribute_ent
        if maxInfoGain is None or infoGain > maxInfoGain:
            maxInfoGain = infoGain
            maxInfoGainAttr = attribute
            maxInfoGainPartitions = partitions

    # Terminate if no attribute gains information
    if maxInfoGain is None:
        node.label = MostCommonClass(data, targetAttr)
        return node

    node.attribute = maxInfoGainAttr
    remainingAttr.remove(maxInfoGainAttr)
    
    # Iterates through all possible values of selected attributtes and runs ID3 recursively
    for klass in attributes[maxInfoGainAttr].classes:
        if klass not in maxInfoGainPartitions.keys():
            child = Node()
            child.label = MostCommonClass(maxInfoGainPartitions, targetAttr)
            node.nodes[klass] = child
            continue
        partition = maxInfoGainPartitions[klass]
        child = id3(partition, targetAttr, remainingAttr)
        child.label
        node.nodes[klass] = child
   
    return node

# Creates out natural language rules by recusivly exploring all branches of the tree.
def PrintTreeImpl( tree, stack, rules ):
    if tree.attribute:
        rule = "if " if not stack else " and "
        rule += tree.attribute + ' is '
        stack.append(rule)
        for key in tree.nodes:
            stack.append(key)
            PrintTreeImpl(tree.nodes[key], stack, rules)
            stack.pop()
        stack.pop()
    elif tree.label:
        stack.append(' then ' + tree.label)
        rules.append(''.join(stack))
        stack.pop()


def PrintTree( tree ):
    stack = []
    rules = []

    PrintTreeImpl( tree, stack, rules )

    for rule in rules:
        print(rule)

def main():
    if len(sys.argv) != 2:
        print("Error - Please pass in a filename.")
        return

    dataFile = sys.argv[1]

    data = []
    with open(dataFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_num, row in enumerate(csv_reader):
            if row_num == 0:
                for column in row:
                    attributeNames.append(column)
                    attributteIndex = len(attributes)
                    attributes[column] = Attribute( attributteIndex )
            else:
                data.append(row)
                for column_num, column in enumerate(row):
                    attributeName = attributeNames[column_num]
                    if(column not in attributes[attributeName].classes ):
                        attributes[attributeName].classes.append(column)
        print(f'Read {csv_reader.line_num} lines.')
    
    targetAttr = attributeNames[0]
    remainingAttr = attributeNames[1:]
    decisionTree = id3(data, targetAttr, remainingAttr)

    PrintTree( decisionTree )  

if __name__ == "__main__":
    main()