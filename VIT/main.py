# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

import torch
from src.VIT import VIT
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    gpus = '1'
    device = torch.device('cuda:{}'.format(gpus) if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    
    # gpu 할당 시 os.environ에 사용하고자하는 gpu만 적어두고 torch Tensor를 .cuda()로 해버리면 바로 알아서 잡음.

    torch.cuda.set_device(device)
    print('Current cuda device' , torch.cuda.current_device())
    print('Available devices' , torch.cuda.device_count())
    print(torch.cuda.get_device_name(device))
    test_input = torch.rand(6,3,512,912).cuda() # # of data, channels, w, h
    model = VIT(img_size=(512,912)).cuda()    #input channels , output channels, kw & kh, padding
    
    out = model(test_input)
    print(out.shape)
    



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
