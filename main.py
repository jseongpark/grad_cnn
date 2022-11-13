import os
import torch

default_dir = "D:\proj/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # GPU 할당

CFG = {
    'IMG_SIZE': 32,  # 이미지 사이즈
    'EPOCHS': 50,  # 에포크
    'LEARNING_RATE': 2e-2,  # 학습률
    'BATCH_SIZE': 12,  # 배치사이즈
    'SEED': 41,  # 시드
}

labels = []
word_to_num = dict()
f = open(default_dir + "ko.txt", 'r', encoding='utf-8')
index = 0
while True:
    line = f.readline()
    if not line: break
    labels.append(line[-2])
    word_to_num[line[-2]] = index
    index = index + 1
f.close()


def invert_dictionary(obj):
  return {value: key for key, value in obj.items()}


num_to_word = invert_dictionary(word_to_num)

from glob import glob


def get_test_data(data_dir):
    img_path_list = []

    # get image path
    img_path_list.extend(glob(os.path.join(data_dir, '*.png')))
    img_path_list.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    return img_path_list


import torchvision.transforms as transforms  # 이미지 변환 툴

from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, train_mode=True, transforms=None):  # 필요한 변수들을 선언
        self.transforms = transforms
        self.train_mode = train_mode
        self.img_path_list = img_path_list
        self.label_list = label_list

    def __getitem__(self, index):  # index번째 data를 return
        img_path = self.img_path_list[index]
        # Get image data
        image = cv2.imread(img_path)
        if self.transforms is not None:
            image = self.transforms(image)

        if self.train_mode:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):  # 길이 return
        return len(self.img_path_list)
import cv2


import torch.nn as nn  # 신경망들이 포함됨


class CNNclassification(nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)

        self.layer2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)

        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.layer6 = torch.nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer7 = torch.nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer8 = torch.nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc_layer = nn.Sequential(
            nn.Linear(2048, 1030)
        )

    def forward(self, x):
        x = self.layer1(x)  # 1층

        x = self.layer2(x)  # 2층

        x = self.layer3(x)  # 3층

        x = self.layer4(x)  # 4층

        x = self.layer5(x)  # 5층

        x = self.layer6(x)  # 6층

        x = self.layer7(x)  # 7층

        x = self.layer8(x)  # 8층

        x = torch.flatten(x, start_dim=1)  # N차원 배열 -> 1차원 배열

        out = self.fc_layer(x)
        return out


def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in iter(test_loader):
            img = img.to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.argmax(dim=1, keepdim=True).squeeze(1)

            model_pred.extend(pred_logit.tolist())
    return model_pred


checkpoint = torch.load(default_dir + 'best_model.pth', map_location=device)
model = CNNclassification().to(device)
model.load_state_dict(checkpoint)


def getSize(txt, font):
    testImg = Image.new('RGB', (1, 1))
    testDraw = ImageDraw.Draw(testImg)
    return testDraw.textsize(txt, font)

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
colorBackground = "white"
colorText = "black"

from hangul_utils import join_jamos
from jamo import h2j,j2hcj


def cnn(text):
    #text=u"ひらがなㄱㄴㄷㄹ"
    jamo_str = j2hcj(h2j(text))

    for i in range(len(text)):
        ch = jamo_str[i]
        font = ImageFont.truetype(default_dir +"ARIALUNI.ttf", 14)
        width, height = getSize(ch, font)
        img = Image.new('L', (width+8, height+8), colorBackground)
        d = ImageDraw.Draw(img)
        d.text((4, height/2-4), ch, fill=colorText, font=font)

        img_dir = default_dir + 'imgs/'
        img.save(img_dir + str(i) + ".png")

    test_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.Resize([CFG['IMG_SIZE'], CFG['IMG_SIZE']]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=0.5, std=0.5)
                        ])
    test_img_path = get_test_data(img_dir)
    test_dataset = CustomDataset(test_img_path, None, train_mode=False, transforms=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)
    preds = predict(model, test_loader, device)
    chs = list(map(lambda pred: num_to_word[pred], preds))
    if chs[0] == chs[1]:
        del chs[0]
        if chs[0] == 'ㄱ':
            chs[0] = 'ㄲ'
        elif chs[0] == 'ㄷ':
            chs[0] = 'ㄸ'
        elif chs[0] == 'ㅂ':
            chs[0] = 'ㅃ'
        elif chs[0] == 'ㅅ':
            chs[0] = 'ㅆ'
        elif chs[0] == 'ㅈ':
            chs[0] = 'ㅉ'
    return chs

# example
chat = u"^^l발ひらがなㄱㄴㄷㄹ"
chs = cnn(chat)
chat = ''.join(chs)
result = join_jamos(chat) # <- 결과물
