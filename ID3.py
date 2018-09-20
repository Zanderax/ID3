import re

attribute_names = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

attribute_values = [
    ['b','c','x','f','k','s'],
    ['f','g','x','y','s'],
    ['n','b','c','g','r','p','u','e','w','y'],
    ['t','f'],
    ['a','l','c','y','f','m','n','p','s'],
    ['a','d','f','n'],
    ['c','w','d'],
    ['b','n'],
    ['k','n','b','h','g','r','o','p','u','e','w','y'],
    ['e','t'],
    ['b','c','u','e','z','r','?'],
    ['f','y','k','s'],
    ['f','y','k','s'],
    ['n','b','c','g','r','p','u','e','w','y'],
    ['n','b','c','g','r','p','u','e','w','y'],
    ['p','u'],
    ['n','o','w','y'],
    ['n','o','t'],
    ['c','e','f','l','n','p','s','z'],
    ['k','n','b','h','r','o','u','w','y'],
    ['a','c','n','s','v','y'],
    ['g','l','m','p','u','w','d']
]

def main():
    file = open("agaricus-lepiota.data", 'r')
    num = 0
    mushrooms = [[]]
    for line in file:
        line = re.sub(',', '', line)
        mushroom = []
        for character in line:
            mushroom.append(character)
        mushrooms.append(mushroom)

    highest_match = 0
    highest_name = 0
    
    for i, name in enumerate( attribute_names ):
        print(name)
        for j, value in enumerate( attribute_values[i] ):
            numMatch = len(
                [m for m in mushrooms if len(m) > 0 and
                    m[i+1] == attribute_values[i][j] ] )
            if numMatch >  highest_match:
                highest_name = name
                highest_match = numMatch

    print(highest_match)
    print(highest_name)
    
    

if __name__ == "__main__":
    main()