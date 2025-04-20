import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

save_dir = "./data/pokemon/poke"
os.makedirs(save_dir, exist_ok=True)

start_id = 1
end_id = 649

for i in tqdm(range(start_id, end_id + 1)):
    url = f"https://pokeapi.co/api/v2/pokemon/{i}/"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            sprite_url = data["sprites"]["front_default"]
            if sprite_url:
                img_data = requests.get(sprite_url).content
                with open(f"{save_dir}/{i:03d}.png", "wb") as f:
                    f.write(img_data)
    except Exception as e:
        print(f"Hata (ID {i}):", e)

image_size = 64
batch_size = 128
z_dim = 100
num_epochs = 100
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

dataset = datasets.ImageFolder(root="./data/pokemon", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self,z_dim):
        super()._init__()
        self.gen = nn.Sequential(
            self.block(z_dim, 512, 4, 1, 0),
            self.block(512, 256, 4, 2, 1),
            self.block(256, 128, 4, 2, 1),
            self.block(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Tanh()
        )

    def block(self,in_c,out_c,k,s,p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c,out_c,k,s,p,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self,x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            self.block(3, 64, 4, 2, 1, bn=False),
            self.block(64, 128, 4, 2, 1),
            self.block(128, 256, 4, 2, 1),
            self.block(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def block(self, in_c, out_c, k, s, p, bn=True):
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.disc(x)

G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.randn(64,z_dim,1,1).to(device)
os.makedirs("output",exist_ok=True)

print("Training Start")
for epoch in range(1000):
    for batch in tqdm(dataloader):
        real_images = batch[0].to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size,1).to(device)
        fake_labels = torch.zeros(batch_size,1).to(device)

        noise = torch.randn(batch_size,z_dim,1,1).to(device)
        fake_images = G(noise)
        D_real = D(real_images).view(-1,1)
        D_fake = D(fake_images.detach()).view(-1,1)

        loss_D_real = criterion(D_real, real_labels)
        loss_D_fake = criterion(D_fake, fake_labels)
        loss_D = loss_D_real + loss_D_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        output = D(fake_images).view(-1, 1)
        loss_G = criterion(output, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

    with torch.no_grad():
        fake = G(fixed_noise)
        img_grid = make_grid(fake, normalize=True)
        save_image(img_grid, f"outputs3/fake_{epoch+1:03d}.png")

    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")


buna uygun readme md dosyasi yaz ingilizce ne yaptk amacimiz ne nasil kullanabilirsiniz 
