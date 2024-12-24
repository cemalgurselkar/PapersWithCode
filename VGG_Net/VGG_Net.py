import torch
import torch.nn as nn
import torch.nn.functional as F

VGG16 = [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M"]
# Then flatten and 4096x4096x1000 Linear Layers

class VGG_Net(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(VGG_Net,self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )

    def create_conv_layers(self,architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = VGG_Net(in_channels=3,num_classes=1000).to(device)
x = torch.randn(1,3,224,224).to(device)
try:
    print(model(x).shape) # torch.size([1,1000])

except Exception as e:
    print(f"Hata Mesaji = {e}")