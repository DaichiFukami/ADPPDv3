import fnmatch
import os
from os.path import join, relpath
from glob import glob
import random
import shutil
from PIL import Image
from PIL import ImageStat
import math

import judgeImg

class ListUp:
    def __init__(self):
        pass

    def resetDir(self,path):
        if(os.path.isdir(path)==True):
            shutil.rmtree(path)
        os.mkdir(path)

    def tdList(self,tdPath):
        self.tdPath = tdPath
        allList = [relpath(x, self.tdPath) for x in glob(join(self.tdPath, '*'))]
        inList = []
        outList = []
        ji = judgeImg.JudgeImg()
        #拡張子を削除
        for i in range(0, len(allList)):
            root = os.path.splitext(allList[i])
            inList.append(root[0])
        inList = fnmatch.filter(inList,"*_i")
        for i in range(0, len(inList)):
            inList[i] = inList[i][:-2]
            jPath = self.tdPath+'/'+inList[i]
            quePath,ansPath = ji.judgeImgSize(jPath)
            if(quePath != 'null' and ansPath != 'null'):
                outList.append((quePath,ansPath))
        return outList

    def inList(self,inPath):
        inPath
        allList = [relpath(x, inPath) for x in glob(join(inPath, '*'))]
        inList = []
        outList = []
        ji = judgeImg.JudgeImg()
        #拡張子を削除
        for i in range(0, len(allList)):
            root = os.path.splitext(allList[i])
            inList.append(root[0])
        for i in range(0, len(inList)):
            jPath = ji.judgeImgName(inPath+'/'+inList[i])
            if(inPath != 'null'):
                outList.append((jPath,inList[i]))
        return outList

    def splitData(self,quePath,ansPath,trainQuePath,trainAnsPath,
                  testQuePath,testAnsPath,testLatio):
        self.resetDir(trainQuePath)
        self.resetDir(trainAnsPath)
        self.resetDir(testQuePath)
        self.resetDir(testAnsPath)
        allList = [relpath(x, quePath) for x in glob(join(quePath, '*'))]
        random.shuffle(allList)
        testLength = int(len(allList)*testLatio)
        #テストデータ
        for i in range(0,testLength):
            shutil.move(quePath+'/'+allList[i],testQuePath+'/'+str(i)+'.png')
            shutil.move(ansPath+'/'+allList[i],testAnsPath+'/'+str(i)+'.png')
        for i in range(testLength,len(allList)):
            shutil.move(quePath+'/'+allList[i],trainQuePath+'/'+str(i-testLength)+'.png')
            shutil.move(ansPath+'/'+allList[i],trainAnsPath+'/'+str(i-testLength)+'.png')
        os.rmdir(quePath)
        os.rmdir(ansPath)

    def sortData(self,quePath,ansPath):
        allList = [relpath(x, quePath) for x in glob(join(quePath, '*'))]
        for i in range(0,len(allList)):
            img = Image.open(quePath+'/'+allList[i])
            stat = ImageStat.Stat(img)
            if(stat.sum == [0,0,0]):
                os.remove(quePath+'/'+allList[i])
                os.remove(ansPath+'/'+allList[i])

    def addNullData(self,quePath,ansPath,charSize):
        allList = [relpath(x, quePath) for x in glob(join(quePath, '*'))]
        b =  math.ceil(len(allList)/19)
        for i in range(0,b):
            img = Image.new('RGB',(charSize,charSize),(0,0,0))
            img.save(quePath+'/b_'+str(i)+'.png')
            img.save(ansPath+'/b_'+str(i)+'.png')
"""
listUp = ListUp()
listUp.splitData('que','ans','trainQue',
                 'trainAns','testQue','testAns',0.2)
"""
"""
listUp = ListUp()
listUp.sortData('que','ans')
"""
"""
listUp = ListUp()
listUp.addNullData('que','ans',96)
"""