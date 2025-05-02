import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class DistillationLoss(nn.Module):
    def __init__(self, lambda_feat=[1.0, 1.0, 1.0, 1.0]):
        super(DistillationLoss, self).__init__()
        self.lambda_feat = lambda_feat
        self.expand_layers = nn.ModuleList([
            nn.Conv2d(44, 50, 1),  # match teacher feature channels
            nn.Conv2d(44, 50, 1),
            nn.Conv2d(44, 50, 1),
            nn.Conv2d(44, 50, 1),
        ])

    def forward(self, student_feats, teacher_feats, student_out, teacher_out, enh_hr):
        feat_loss = 0.0
        for i in range(4):
            s_feat = self.expand_layers[i](student_feats[i])
            t_feat = teacher_feats[i].detach()
            feat_loss += self.lambda_feat[i] * F.l1_loss(s_feat, t_feat)

        out_loss = F.l1_loss(student_out, teacher_out.detach()) + F.l1_loss(student_out, enh_hr)
        total_loss = feat_loss + out_loss
        return total_loss


class StudentWithFeatures(nn.Module):
    def __init__(self, model):
        super(StudentWithFeatures, self).__init__()
        self.model = model

    def forward(self, x):
        feat = self.model.conv_1(x)

        f1 = self.model.block_1(feat)
        f2 = self.model.block_2(f1)
        f3 = self.model.block_3(f2)
        f4 = self.model.block_4(f3)

        out_lr = self.model.conv_2(f4) + feat
        out_hr = self.model.upsampler(out_lr)
        return out_hr, [f1, f2, f3, f4]


class Distiller(nn.Module):
    def __init__(self, student, teacher, loss_fn):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
        self.loss_fn = loss_fn
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x, enh_hr):
        student_out, student_feats = self.student(x)
        with torch.no_grad():
            teacher_out, teacher_feats = self.teacher(x)

        loss = self.loss_fn(student_feats, teacher_feats, student_out, teacher_out, enh_hr)
        return loss, student_out


class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = os.path.join(root_dir, 'LR')
        self.hr_dir = os.path.join(root_dir, 'HR')
        self.filenames = sorted(os.listdir(self.lr_dir))
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')

        if self.transform_lr:
            lr_img = self.transform_lr(lr_img)
        if self.transform_hr:
            hr_img = self.transform_hr(hr_img)

        return lr_img, hr_img


# --- Training script ---
from DIPnet import DIPNet
from RFDN import RFDN

class RFDNWithFeatures(nn.Module):
    def __init__(self, model):
        super(RFDNWithFeatures, self).__init__()
        self.model = model

    def forward(self, x):
        out_fea = self.model.fea_conv(x)
        out1 = self.model.B1(out_fea)
        out2 = self.model.B2(out1)
        out3 = self.model.B3(out2)
        out4 = self.model.B4(out3)

        out_cat = self.model.c(torch.cat([out1, out2, out3, out4], dim=1))
        out_lr = self.model.LR_conv(out_cat) + out_fea
        out_hr = self.model.upsampler(out_lr)

        return out_hr, [out1, out2, out3, out4]


# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 16
learning_rate = 1e-4
lambda_feat = [1.0, 1.0, 1.0, 1.0]

transform_lr = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_hr = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor()
])

train_dataset = PairedImageDataset(
    root_dir='dataset/train',
    transform_lr=transform_lr,
    transform_hr=transform_hr
)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

student_raw = DIPNet().to(device)
teacher_raw = RFDN().to(device)
teacher_raw.load_state_dict(torch.load('trained_model/RFDN_AIM.pth'))

student = StudentWithFeatures(student_raw).to(device)
teacher = RFDNWithFeatures(teacher_raw).to(device)

loss_fn = DistillationLoss(lambda_feat).to(device)
distiller = Distiller(student, teacher, loss_fn).to(device)
optimizer = torch.optim.Adam(distiller.student.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    distiller.train()
    epoch_loss = 0.0
    for lr_img, hr_img in dataloader:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        loss, sr_out = distiller(lr_img, hr_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

torch.save(student_raw.state_dict(), 'student_dipnet_distilled.pth')
