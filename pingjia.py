
from torchvision import transforms
import natsort
import pandas as pd
from torchvision import transforms as T
import numpy as np
from PIL import Image
import os
from medpy.metric.binary import precision,specificity,recall,dc,jc,sensitivity
def mean_iou(input, target, classes = 2):
    """  compute the value of mean iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        miou: float, the value of miou
    """
    miou = 0
    #for i in range(classes):
    intersection = input * target
    union = input + target - intersection
    temp = np.sum(intersection) / np.sum(union)
    miou += temp
    return  miou


test="./GLAS/test/masks"
#test = r"./newcrag/test/masks"
#test="./CRAG/test/masks"
#test = r"D:\yhj\ai\data\nous\test\masks"
test_files=os.listdir(test)
test_files=natsort.natsorted(test_files)
print(test_files)

gt1="./save6"
gt_files=os.listdir(gt1)
gt_files=natsort.natsorted(gt_files)
print(gt_files)
num=len(test_files)
avg=[]
s=0
jar=0
miou_S=0
re=0
pe=0
se=0
jaccard_all=[]
f1_all=[]
Recall=[]
Specificity=[]
Precision=[]
Dice=[]
Iou=[]
ddice=0
iiou=0
Sensitivity=[]
sensi=0
s_sensi=0
data_transform = transforms.Compose([
            transforms.Resize((256,256),interpolation=T.InterpolationMode.NEAREST),
           # transforms.ToTensor(),
                                        #     transforms.Normalize(mean=mean, std=std)
                                            # transforms.RandomResizedCrop
                                             ])
for i in range(0,num):
    output=Image.open(os.path.join(test,test_files[i]))
    gt=Image.open(os.path.join(gt1,gt_files[i]))
    output=data_transform(output)

    output = np.asarray(output) / 255
    gt = np.asarray(gt) / 255
    dice=dc(output, gt)
    Dice.append(dice)
    iou=jc(output, gt)
    Iou.append(iou)
    sensi=sensitivity(output, gt)
    Sensitivity.append(sensi)
    miou = mean_iou(output, gt)
    rerecall = recall(output, gt)
    Recall.append(rerecall)

    prpre = precision(output, gt)
    Precision.append(prpre)
    speci = specificity(output, gt)
    Specificity.append(speci)
    print("re,pe,se",rerecall,prpre,speci)
    output = list(np.array(output).flatten())
    gt=list(np.array(gt).flatten())


    f1 = 0
    f1_all.append(f1)
    jaccard=0
    jaccard_all.append(jaccard)
    accuracy=0
    print("test_pictiure: {} F1_score: {:.3f},mean_iou: {:.3f}"
    #val_loss: {:.3f}, val_acc: {:.3f}, lf: {:.6f}
          .format(os.path.join(test,test_files[i]), f1, miou))
    se=se+speci
    re=re+rerecall
    pe=pe+prpre
    ddice=ddice+dice
    iiou=iiou+iou
    s=s+f1
    jar=jar+jaccard
    miou_S=miou_S+miou
    s_sensi=s_sensi+sensi
print("test num:{},mean F1 score:{},meam iou:{}".format(num,s/num,miou_S/num))
print("jar",jar/num)
print("miou",miou_S/num)
print("jar-all",jaccard_all)
print("sensi",s_sensi/num)
print("se",se/num)
print("pe",pe/num)
print("re",re/num)
print("iou",iiou/num)
print("dice",ddice/num)
#pd.DataFrame(jaccard_all)
data1 = pd.DataFrame(Dice)
data1.to_csv('dice.csv')
data2 = pd.DataFrame(Iou)
data2.to_csv('iou.csv')
data3 = pd.DataFrame(Recall)
data3.to_csv('Recall.csv')
data4 = pd.DataFrame(Sensitivity)
data4.to_csv('Sensitivity.csv')
data5 = pd.DataFrame(Precision)
data5.to_csv('Precision.csv')
data6 = pd.DataFrame(Specificity)
data6.to_csv('Specificity.csv')
