from pprint import pprint


class TreeNode:
    def __init__(self):
        self.predictions = None
        self.question = None
        self.trueBranch = None
        self.falseBranch = None
        self.isLeaf = False
        self.name = id(self)
        self.cost = 0


class Decision:
    def __init__(self, column, value, cost):
        self.column = column
        self.value = value
        self.cost = cost

    def match(self, example):
        val = example[self.column]
        return val >= self.value

    def __repr__(self):
        return str(header[self.column]) + " ("+str(self.cost)+") >= " + str(self.value)


def getClassFrequencies(data):
    freqs = {}  # label -> count.
    for r in data:
        label = r[-1]
        if label not in freqs:
            freqs[label] = 0
        freqs[label] += 1
    return freqs


def partition(data, question):

    leftSplit = []
    rightSplit = []
    for r in data:
        if question.match(r):
            leftSplit.append(r)
        else:
            rightSplit.append(r)
    return leftSplit, rightSplit


def giniImpurity(data):
    # https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    counts = getClassFrequencies(data)
    impurity = 1
    for label in counts:
        p = counts[label] / float(len(data))
        impurity -= p**2
    return impurity


def informationGain(left, right, uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return uncertainty - p * giniImpurity(left) - (1 - p) * giniImpurity(right)


def getBestSplit(data):

    bestInfoGain = 0
    bestSplitAt = None
    uncertainty = giniImpurity(data)
    numFeatures = len(data[0]) - 1

    for col in range(numFeatures):
        values = set([row[col] for row in data])
        for val in values:

            split = Decision(col, val, costs[col])
            # try splitting the dataset
            trueSubdata, falseSubdata = partition(data, split)
            # Skip this split if it doesn't divide the dataset.
            # if len(trueSubdata) == 0 or len(falseSubdata) == 0:
            # continue
            gain = informationGain(trueSubdata, falseSubdata, uncertainty)
            if useCost:
                gain = gain*gain/costs[col]

            if gain >= bestInfoGain:
                bestInfoGain, bestSplitAt = gain, split

    return bestInfoGain, bestSplitAt


def buildTree(rows, depth):

    if depth != maxDepth:
        gain, question = getBestSplit(rows)

        if gain == 0:
            t = TreeNode()  # leafnode
            t.isLeaf = True
            t.predictions = getClassFrequencies(rows)
            return t
            # return LeafNode(rows)

        trueSubdata, falseSubdata = partition(rows, question)

        trueBranch = buildTree(trueSubdata, depth+1)
        falseBranch = buildTree(falseSubdata, depth+1)

        t = TreeNode()  # splitnode
        t.question = question
        t.trueBranch = trueBranch
        t.falseBranch = falseBranch
        t.cost = question.cost
        return t
    else:
        t = TreeNode()  # leafnode
        t.isLeaf = True
        x = getClassFrequencies(rows)

        t.predictions = {max(x, key=x.get): x[max(x, key=x.get)]}
        return t


def printNodes(node):
    if node.isLeaf:
        print(str(node.name)+"[label=\"" +
              str(node.predictions) + "\",color=\"green\"]")
        return
    else:
        print(str(node.name)+"[label=\"" + str(node.question) + "\"]")
        printNodes(node.trueBranch)
        printNodes(node.falseBranch)
        return


def aprintTree(node, spacing=""):

    s = str(node.name)+" -> "
    s += str(node.trueBranch.name) + "[label=\"yes\"]"
    if not node.trueBranch.isLeaf:
        aprintTree(node.trueBranch, spacing + "\t")
    print(s)

    s = str(node.name)+" -> "
    s += str(node.falseBranch.name) + "[label=\"no\"]"
    if not node.falseBranch.isLeaf:
        aprintTree(node.falseBranch, spacing + "\t")
    print(s)


def predict(row, node, cost):

    # if isinstance(node, LeafNode):
    if not node.isLeaf:
        pass
    else:
        pass
        return node.predictions, cost

    if node.question.match(row):
        return predict(row, node.trueBranch, cost + node.trueBranch.cost)

    else:
        return predict(row, node.falseBranch, cost + node.trueBranch.cost)


def printLeafNode(counts):
    total = sum(counts.values()) * 1.0
    freqs = {}
    for classLabel in counts.keys():
        freqs[classLabel] = str(int(counts[classLabel] / total * 100)) + "%"
    return freqs


def getDataFromFile(filename, s=" "):
    f = open(filename, "r")
    lines = f.readlines()
    data = []
    for l in lines:
        items = l.rstrip().lstrip().split(sep=s)
        t = []
        for i in items:
            t.append(float(i))
        data.append(t)
    f.close()
    return data


def exportToGraphViz(dtRoot):
    f = open("out.gvz", "w")
    import sys
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = f  # Change the standard output to the file we created.
    print("digraph G{")
    printNodes(dtRoot)
    aprintTree(dtRoot)
    print("}")
    sys.stdout = original_stdout  # Reset the standard output to its original value
    f.close()


def getCosts():
    costs = []
    costs.append(1)  # age:				1.00
    costs.append(1)  # sex:				1.00
    costs.append(1)  # on_thyroxine:			1.00
    costs.append(1)  # query_on_thyroxine:		1.00
    costs.append(1)  # on_antithyroid_medication:	1.00
    costs.append(1)  # sick:				1.00
    costs.append(1)  # pregnant:			1.00
    costs.append(1)  # thyroid_surgery:		1.00
    costs.append(1)  # I131_treatment:			1.00
    costs.append(1)  # query_hypothyroid:		1.00
    costs.append(1)  # query_hyperthyroid:		1.00
    costs.append(1)  # lithium:			1.00
    costs.append(1)  # goitre:				1.00
    costs.append(1)  # tumor:				1.00
    costs.append(1)  # hypopituitary:			1.00
    costs.append(1)  # psych:				1.00
    costs.append(22.78)  # TSH:				22.78
    costs.append(11.41)  # T3:				11.41
    costs.append(14.51)  # TT4:				14.51
    costs.append(11.41)  # T4U:				11.41
    costs.append(25.92)  # T4U:				11.41
    return costs

#################################################


maxDepth = 4
useCost = True
trainingSet = getDataFromFile("./data/ann-train.data")
testingSet = getDataFromFile("./data/ann-train.data")
costs = getCosts()


header = ["f"+str(i) for i in range(len(trainingSet[0]))]

dtRoot = buildTree(trainingSet, 0)
exportToGraphViz(dtRoot)

acc = 0
hits = []
misses = []

print(len(testingSet))
for r in testingSet:
    a = r[-1]
    p, c = predict(r, dtRoot, 0)

    if p.get(a):
        hits.append(str(a) + " " + str(p) + " " + str(c))
        acc += 1
    else:
        misses.append(str(a) + " " + str(p) + " " + str(c))


pprint(hits)
pprint(misses)
print("Total: " + str(acc/len(testingSet)*100))

#################################################
