#coding:utf-8
import json

f = open('fixations.json', 'r')

jsonData = json.load(f)

print json.dumps(jsonData, sort_keys = True, indent = 4)

f.close()
