import torch
import DataSet
from torchtext import data

def getModel(name):
    model = torch.load('done_model/'+name+'_model.pkl')
    return model

model = getModel('TextRNN_Attention' )
device = torch.device('cpu',map_location=torch.device('cpu'))
model = model.to(device)
sent1 = ['大堂不错，有四星的样子，房间的设施一般，感觉有点旧，卫生间细节不错，各种配套东西都不错，感觉还可以，有机会再去泰山还要入住。',
         '姐永远不卖了。坑爹伤不起。次奥。话说人家名字好吧。 好个性的名字草泥马奇葩，蒙牛姐永远不喝啦',
         '订单里明明有这本书为什么其它书都已收到为什么就没有这一本了？？？为什么了？？？']
for num in range(0,3):
    demo = [data.Example.fromlist(data=[sent1[num],0],fields=[('text',DataSet.getTEXT()),('label',DataSet.getLabel())])]
    demo_iter = data.BucketIterator(dataset=data.Dataset(demo,[('text',DataSet.getTEXT()),('label',DataSet.getLabel())]), batch_size=256, shuffle=True,sort_key=lambda x:len(x.text), sort_within_batch=False, repeat=False)
    for batch in demo_iter:
                feature = batch.text
                target = batch.label
                with torch.no_grad():
                    feature = torch.t(feature)
                feature, target = feature.to(device), target.to(device)
                out = model(feature)
                print(sent1[num])
                if torch.argmax(out, dim=1).item() == 0:
                    print('差评')
                elif torch.argmax(out, dim=1).item() == 1:
                    print('好评')
                else:
                    print('None')

