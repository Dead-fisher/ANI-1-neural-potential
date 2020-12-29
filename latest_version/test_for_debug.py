import matplotlib.pyplot as plt
import numpy as np

for i in range(1):
    ls = []
    ls_10 = []
    ls_11 = []
    ls_12 = []
    step = []
    with open('./history/logging_val.txt') as f:
        for i, words in enumerate(f.readlines()):
            if words.split(',')[0].isalpha():
                continue
            ele = words.split(',')
            step.append(i)
            ls.append(float(words.split(',')[2]))
            '''
            if 5 < int(ele[0]) < 17:
                step.append(i)
                ls.append(float(words.split(',')[2]))
            '''
            '''
            if int(ele[0]) == 10:
                step.append(int(ele[1]))
                ls_10.append(float(words.split(',')[2]))
            if int(ele[0]) == 11:
                ls_11.append(float(words.split(',')[2]))
            if int(ele[0]) == 12:
                ls_12.append(float(words.split(',')[2]))
            '''

    # fft_ = np.fft.fft(ls)
    plt.figure()
    '''
    plt.plot(step, ls_10, label='11')
    plt.plot(step, ls_11, label='12')
    plt.plot(step, ls_12, label='13')
    '''
    # plt.plot(fft_[1:int(len(fft_)/2-1)], label='loss')
    plt.plot(step[1:], ls[1:], label='11')
    plt.legend()
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.show()
    # print(input('请按回车'))

