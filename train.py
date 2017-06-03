import os
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.models as models
import torch.optim as optim
import numpy as np
import scipy
import inception
from scipy import misc
from PIL import Image

MODEL_DIR = "./model/"
DATA_DIR = "./data/"
EPOCH = 10
BATCH_SIZE = 15
h = 299
w = 299
LUT = ["agriculture",
"artisinal_mine",
"bare_ground",
"blooming",
"blow_down",
"clear",
"cloudy",
"conventional_mine",
"cultivation",
"habitation",
"haze",
"partly_cloudy",
"primary",
"road",
"selective_logging",
"slash_burn",
"water"
]

models = os.listdir(MODEL_DIR)
now = 0
if(len(models) == 0):
    mo = inception.inception_v3(pretrained = True).cuda()
else:
    now = 2
    mo = (torch.load(MODEL_DIR + "2_1000.pkl"))
    mo = mo.cuda()
class FineTuneModel(nn.Module):
    def __init__(self, num):
        super(FineTuneModel, self).__init__()
        self.features = nn.Sequential(*list(incept.children())[:-1])
        self.classifier = nn.Sequential( nn.Linear(2048, num))
    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

#mo = FineTuneModel(17).cuda()
transformations = transforms.Compose([transforms.Scale((299, 299)), transforms.ToTensor()])

def import_image(s, path, augmentation = False):
    name, label = s.split()
    name = path + name
    img = Image.open(name)
    img = img.convert('RGB')
    #img = np.float32(np.swapaxes(np.swapaxes(img, 2, 3), 1, 2))
    if augmentation:
        img = np.array([np.array(img)])
        img = utils.data_augmentation(img)
        image = []
        for i in xrange(len(img)):
            image.append(Image.fromarray(img[i].astype('uint8')))
    else:
        image = img
    label = label.split(",")
    label = [int(_) for _ in label]
    label = np.float32(label)

    return image, label

class Data(Dataset):
    def __init__(self, typ):
        f = open(DATA_DIR + typ + ".txt")
        self.typ = typ
        self.l = []
        cnt = 0
        for line in f.readlines():
            cnt += 1
            print cnt
            self.l.append(line)
        print(typ + ".txt read finish")
        np.random.shuffle(self.l)
    def __getitem__(self, i):
        path = DATA_DIR + self.typ + "/"
        image, label = import_image(self.l[i], path)
        return image, label
    def __len__(self):
        return len(self.l)

def dataset(typ, batch_size, shuffle = True, augmentation = False):
    f = open(DATA_DIR + typ + ".txt")
    l = []
    cnt = 0
    for line in f.readlines():
        cnt += 1
        print cnt
        l.append(line)
    print(typ + ".txt read finish")
    np.random.shuffle(l)
    data = []
    target = []
    for i in xrange(len(l)):
        image, label = import_image(l[i], DATA_DIR + typ + "/", augmentation)
        data += [transformations(x) for x in image]
        target += [label for j in xrange(len(image))]
        if(i%BATCH_SIZE == 0 and i!=0):
            yield default_collate(data), torch.from_numpy(np.array(target))
            data = []
            target = []

criterion = nn.MultiLabelSoftMarginLoss().cuda()
optimizer = optim.Adam(mo.parameters(), lr = 0.001)
print "start"
for e in xrange(EPOCH):
    cnt = 0
    mo.train()
    train_loader = dataset("train", batch_size = BATCH_SIZE, shuffle = True, augmentation = True)
    print "switch to train"
    for i, (data, target) in enumerate(train_loader):
        inputs, labels = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        #outputs = mo.forward(inputs)
        output1, output2 = mo(inputs)
        outputs = 0.8 * output1 + 0.2 * output2
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print outputs.data
        print("[%d] Batch: %d, Loss: %.3f" % (e+1+now,i,loss.data[0]))
        if(i % 500 == 0 and i != 0):
            #val()
            torch.save(mo, MODEL_DIR + str(e + now) + "_" + str(i) + ".pkl")

