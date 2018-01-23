import numpy as np
xp = np

from PIL import Image
import glob
import sys

import chainer

trainQuePath ='trainQue'
trainAnsPath = 'trainAns'
testQuePath ='testQue'
testAnsPath = 'testAns'
charSize = 16*8
outUnit = 3*charSize*charSize
unitSize = round(outUnit*4/3)

class DatasetPourDot(chainer.dataset.DatasetMixin):
    def __init__(self,quePath,ansPath):
        self.quePath = quePath
        self.ansPath = ansPath
        #データの数を吐く
        n1 = len(glob.glob(quePath + '/*.png'))
        n2 = len(glob.glob(ansPath + '/*.png'))
        if n1 != n2:
            print ('エラー:学習用データが不正です')
            sys.exit(1)
        self._size = n1
        return

    def __len__(self):
        #ここで入っているデータの一覧を入れる
        return self._size

    def get_example(self, idx):
        if idx < 0 or idx >= self._size:
            print ('エラー:不正なインデックス')
            sys.exit(1)
        #ここから問題データ読み込み
        queImg = Image.open(self.quePath + ('/')+str(idx)+('.png' )).convert("HSV")
        ansImg = Image.open(self.ansPath + ('/')+str(idx)+('.png' )).convert("HSV")
        """
        queNum = self.img2numpy(queImg)
        ansNum = self.img2numpy(ansImg)

        return queNum,ansNum
        """

        queCnn = self.img2cnn(queImg)
        ansCnn = self.img2cnn(ansImg)

        return queCnn,ansCnn

    def img2cnn(self,Img):
        h,s,v = Img.split()
        h2 = xp.asarray(xp.float32(h)/255.0)
        s2 = xp.asarray(xp.float32(s)/255.0)
        v2 = xp.asarray(xp.float32(v)/255.0)
        cnn = xp.asarray([h2, s2, v2])
        return cnn