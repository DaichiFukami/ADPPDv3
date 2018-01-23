import listUp
import editImg

import numpy as np
import cupy as cp
xp = np


#import models
import chainer
import chainer.functions as F
import chainer.links as L

import datetime

inPath = 'indata'
outPath = 'outdata'
charSize = 16*8
moveLength = 1
gpu = 0

def addNumpy(outNumpys,numpys,charSize,ySt,xSt):
    for d in range(0, 3):
        for y in range(0, charSize):
                for x in range(0, charSize):
                    outNumpys[d][ySt+y][xSt+x] = outNumpys[d][ySt+y][xSt+x] + numpys[d][y][x]
    return outNumpys

#modelの設定
model = L.Classifier(models.ADPPD_CNN(),lossfun=F.mean_squared_error)#out-10種類(0-9の数字判別のため)
chainer.serializers.load_npz('mymodel.npz', model)

#GPU有無の判別
if gpu >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
    model.to_gpu()
    xp = cp
    print('I use GPU and cupy')


#ファイルのリストアップ
listUp = listUp.ListUp()
fileList = listUp.inList(inPath)

#一時データ用ディレクトリの初期化
listUp.resetDir(outPath)

#edit = editImg.EditImg(charSize,moveLength,gpu)
print('進行状況:'+str(0)+'/'+str(len(fileList)))
for i in range(0, len(fileList)):
    inImgs = edit.preTeachImgCNN(fileList[i][0])
    outNumpys = xp.array([[[0 for x in range(charSize-1+len(inImgs[0]))]for y in range(charSize-1+len(inImgs))] for d in range(3)] , dtype=xp.float32)
    for y in range(0,len(inImgs)):
            for x in range(0,len(inImgs[0])):
                numpy = edit.img2cnn(inImgs[y][x])
                numpy = model.predictor(xp.array([numpy]).astype(xp.float32))
                numpy = F.relu(numpy)
                numpy2 = numpy.data[0]
                print('出力中-'+str(x)+','+str(y))
                outNumpys = addNumpy(outNumpys,numpy2,charSize,y,x)
    outNumpys = chainer.cuda.to_cpu(outNumpys)
    outImg = edit.pile2img(outNumpys,charSize)
    outImg = outImg.convert("RGB")
    outImg.save(outPath+'/'+fileList[i][1]+'.png')
    print('進行状況:'+str(i+1)+'/'+str(len(fileList)))
    d = datetime.datetime.today()
    print('終了時刻:', d)