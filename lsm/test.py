import torch
import models.fc_layers

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test = torch.rand((10,12,32,32,32)).to(device)
fc_layers = models.fc_layers.FCLayers(in_channels=12, dim=32).to(device)
out = fc_layers(test)
