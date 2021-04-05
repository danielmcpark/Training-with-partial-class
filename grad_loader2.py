import torch
import numpy as np
import matplotlib.pyplot as plt

for idx in range(10):
    ckpt = torch.load('./logs/{}-class_grads.pth.tar'.format(idx))
    locals()['ckpt_{}'.format(idx)] = ckpt.detach().numpy()

#265600:266600
plt.subplot(311)
for i in range(4):
    plt.plot(range(300), locals()['ckpt_{}'.format(i)][0,:300])
plt.legend(loc='upper right')

plt.subplot(312)
for i in range(4):
    plt.plot(range(100), locals()['ckpt_{}'.format(i)][0,300:400])
plt.legend(loc='upper right')

plt.subplot(313)
for i in range(4):
    plt.plot(range(10), locals()['ckpt_{}'.format(i)][0,400:410], label='class-{}'.format(i))
plt.legend(loc='upper right')

plt.show()
