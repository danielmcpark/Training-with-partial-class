import torch
import numpy as np
import matplotlib.pyplot as plt

ckpt = torch.load('./logs/1-class_grads.pth.tar')
ckpt = ckpt.detach().numpy()
print(ckpt.shape)

#265600:266600
plt.subplot(311)
plt.plot(range(20), ckpt[0,:20])
plt.plot(range(20), ckpt[1,:20])
plt.plot(range(20), ckpt[2,:20])
plt.plot(range(20), ckpt[3,:20])
plt.legend(loc='upper right')

#plt.plot(range(300), ckpt[4,:300])

plt.subplot(312)
plt.plot(range(50), ckpt[0,20:70])
plt.plot(range(50), ckpt[1,20:70])
plt.plot(range(50), ckpt[2,20:70])
plt.plot(range(50), ckpt[3,20:70])
plt.legend(loc='upper right')

plt.subplot(313)
plt.plot(range(500), ckpt[0,70:570], label='1-iter')
plt.plot(range(500), ckpt[1,70:570], label='2-iter')
plt.plot(range(500), ckpt[2,70:570], label='3-iter')
plt.plot(range(500), ckpt[3,70:570], label='4-iter')
plt.legend(loc='upper right')

plt.show()
