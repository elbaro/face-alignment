import tinder
import torch
import models

tinder.setup(parse_args=False)

Variable = torch.autograd.Variable

net = models.FAN(4)
x = Variable(torch.zeros(1,3,256,256))
x = net(x)

import pdb
pdb.set_trace()