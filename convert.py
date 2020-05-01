import sys
import hashlib
inputPath = sys.argv[1]
edgesPath = sys.argv[2]
attribsPath = sys.argv[3]

inFile = open(inputPath, 'r')
lines = inFile.readlines()

codes = set([])
for line in lines:
	line = line[:-2]
	fields = line.split('\t')
	codes |= set([fields[0]] + fields[9:])

print len(codes)
codes = list(codes)
codes.sort()

mapping = dict()
for i in range(len(codes)):
	# print codes[i]
	mapping[codes[i]] = str(i)

print "Done"

def f(x):
	global mapping
	return mapping[x]
    
outFile = open(edgesPath, 'w')
outFile2 = open(attribsPath, 'w')

outString = ""
outString2 = ""

it = 0.0
for line in lines:
    it += 1
    if it%100==0:
    	pass
    	print it/len(lines)
    line = line[:-2]
    fields = line.split('\t')
    related = fields[9:]
    vidNum = f(fields[0])
    
    toPrint = vidNum
    for field in fields[:9]:
        toPrint += "\t"
        toPrint += field

    outString2+=toPrint+"\n"
    for relatedVid in fields[9:]:
        relatedNum = f(relatedVid)
        outString += vidNum + "\t" + relatedNum + "\n"

outFile.write(outString)
outFile.flush()
outFile2.write(outString2)
outFile2.flush()
outFile.close()
outFile2.close()
