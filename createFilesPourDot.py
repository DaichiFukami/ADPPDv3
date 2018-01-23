import listUp
import editImg

tdPath = 'alldata'
quePath ='que'
ansPath = 'ans'
trainQuePath ='trainQue'
trainAnsPath = 'trainAns'
testQuePath ='testQue'
testAnsPath = 'testAns'
charSize = 16*8
moveLength = 16*8/2

#ファイルのリストアップ
listUp = listUp.ListUp()
fileList = listUp.tdList(tdPath)
#一時データ用ディレクトリの初期化
listUp.resetDir(quePath)
listUp.resetDir(ansPath)

edit = editImg.EditImg(charSize,moveLength,-1)
print('進行状況:'+str(0)+'/'+str(len(fileList)))
for i in range(0, len(fileList)):
    #edit.outImg(quePath,fileList[i][0])
    #edit.outImg(ansPath,fileList[i][1])
    edit.outImg2(quePath,fileList[i][0])
    edit.outImg2(ansPath,fileList[i][1])
    print('進行状況:'+str(i+1)+'/'+str(len(fileList)))

#黒一色データの削除
print('黒データの調整中')
listUp.sortData(quePath,ansPath)
listUp.addNullData(quePath,ansPath,charSize)
print('黒一色データの調整完了')
print('シャッフル中')
listUp.splitData(quePath,ansPath,trainQuePath,trainAnsPath,
                 testQuePath,testAnsPath,0.2)
print('シャッフル完了')