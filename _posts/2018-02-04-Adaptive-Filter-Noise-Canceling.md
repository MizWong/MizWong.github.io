---
layout:     post
title:      "自适应噪声消除器"
subtitle:   "2017电子设计大赛E题"
date:       2018-02-04
author:     "Miz.Wong"
header-img : "img/2018-02-04-Adaptive-Filter-Noise-Canceling/page.png"
catalog: true
tags:
    - Digital Signal Processing
    - Competition 
---


<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 写在前面
最近恰好在做均衡和自适应滤波器方面的东西，忽然想起来2017年电子设计大赛中的E题，便简单做了一下仿真和设计，与各位同学分享。

# 题目回顾
![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 18.42.55.png)
完整的题目可以参考[这里](https://wenku.baidu.com/view/ef3c3e01a66e58fafab069dc5022aaea998f41b2.html)。
简单来说，这道题目的目的就是实现一个信号生成器、移相器和自适应滤波器。信号生成器和移相器都比较简单就不再做多余说明了，本文主要说一下自适应滤波器部分。

# 理论推导
设目标信号（原始信号）为：

$$u_d(t) = cos(\omega_d t + \phi_d )$$

干扰信号为：

$$u_n(t) = cos(\omega_n t + \phi_n )$$

则经过加法器合路后的数据为：

$$u(t) = u_d(t)+u_n(t)+n(t)$$

其中n(t)代表测量噪声和系统噪声的总和。

进一步经过相移器的信号相当于延时，可表述为：

$$u(t) = u_d(t)+u_n(t-\tau)+n(t-\tau)$$



剩下的涉及到自适应滤波器算法的使用，一般来说LMS和RLS是自适应算法的两大族，LMS实现简单但是稳定性和收敛速率都较差，RLS则相反。考虑题目需求简单，不需要较高阶数，且指标要求较高，故选用RLS算法。

看到这里有些同学可能还有一些迷惑，没关系，我们先来看下自适应滤波器的结构：
![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 19.13.52.png)

主要由一个线性抽头延迟线的滤波器结构、一个误差计算模块和一个权值调整模块组成。我们都知道，频域的乘积对应时域的卷积，所以这种抽头延迟线结构是在时域实现FIR滤波器的基础，每一个新的sample进入后都会讲旧的sample向外推，和抽头权值w做乘法后求和便是该时刻的输出。自适应滤波器的这一部分结构和其他抽头延迟线滤波器是相同的。

而误差计算模块很简单，是用来计算期望值和实际输出的差值：

$$e(n) = y_d(n)+\widehat y(n)$$

在得到误差后，便是自适应算法的核心部分了，这里面简单介绍一下RLS，详细的推导以及LMS的实现可以自己找相关资料。

首先是卡尔曼矩阵的计算，由于我们输入的只是一元的数据，故卡尔曼矩阵呈向量的形式，长度和抽头数量相同：

$$K(n) = \frac {P(n-1) u(n)}{ \lambda + u(n)^H P(n-1) u(n)}$$

注意这里的K、u都是Nx1的向量，P是NxN的矩阵，N是滤波器抽头数量，H代表共轭转置。λ代表遗忘因子，是一个0-1的数值，它越小算法收敛越快。P是协相关矩阵，初始时一般是一个单位对角阵。


第n时刻的输出可表示为：

$$\widehat y(n) = u(n) * w(n-1)$$

其中u和w都是向量，代表第n个时刻抽头中寄存的向量和前一时刻抽头的权值（此时还未更新）。

则第n时刻的误差为：

$$e(n) = y_d(n)+\widehat y(n)$$

此时更新权值：

$$w(n) = w(n-1)+K^* e$$

其中＊代表共轭（实数则不需要处理）。

进而再更新协相关向量：

$$P(n) = \frac {P(n-1) - K(n) u(n)^H P(n-1)}{ \lambda }$$

# 系统设计
### 采样速率

首先题目要求可输入的信号范围10kHz~100kHz，根据Nyquist采样定理，最小的采样率应为200kHz，实际一般使用5倍速，这里为了方便仿真观察，我们将采样速率设置为1MHz。
### 框图设计
这里是用Simulink搭建框图。

前面我们推导了自适应滤波器的结构和算法，但是没有提具体怎么用。

首先我们知道了自适应滤波器有几个关键的部分：输入数据、期望数据、输出数据和误差。

题目中规定我们可以拿到经过延迟后的合路数据和原始的干扰数据，这样看来谁都不能充当直接的期望信号。这里要引出自适应滤波器的经典应用场景－－对已知噪声的滤除。

大家都知道现在的手机一半都是有两个Mic的，一个在讲话处，另外一个一般是在背面，讲话的Mic混合有用的语音和背景噪声，而背面的Mic可认为只采集了背景噪声，这和我们的场景就非常相像了。

在这种场景下，一般采用*__合路后的信号作为期望、噪声信号作为滤波器输入，而误差信号即为恢复信号。__*

下面搭建框图如下：
![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 18.29.08.png)

很简单，两路cos信号分别作为期望和干扰，经过合路时加入一些测量噪声，后面加入延迟模块充当相移，最后经过RLS自适应滤波器计算输出并观察恢复的信号和原始信号。

RLS参数设定比较随意，因为输入是两个单音信号的合路，滤波器不需要太长，这里我们是用32个Tap（估计一下2个就够用），同时注意*__遗忘因子λ必须设置为1__*，这也是Noise canceling的一个特殊之处。

题目中要求了两个输入信号的频率偏差范围，那我们分别进行两次测试，第一次是相差1k时，得到的估计信号和原始信号：

![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 18.32.37.png)

可以看出来，除了加性白噪声以外，恢复的信号和期望的原始信号基本一样，我们对它们分别进行测量：

![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 18.32.56.png)

![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 18.33.06.png)

可以看到幅度误差和频率误差都ok。

当然，这里面我们加入的测量噪声为峰峰值0.2的白噪声，信噪比只有10.4dB,实际情况噪声要远小于这个数值，算法的性能也会更好。

将干扰信号和有用信号间隔缩减到10Hz：

![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 18.34.31.png)

![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 18.34.50.png)

此时也是ok的，而且滤波器的收敛时间完全是在us级。

### 一点思考


正常时自适应的滤波器是会向误差为0的方向进行迭代的，但是我们可以看到在噪声消除的应用中它收敛到一定程度后就不再变化，此时观察它的抽头和对应的频率响应：

![image](/img/2018-02-04-Adaptive-Filter-Noise-Canceling/屏幕快照 2018-02-04 20.19.22.png)

对于我们输入的信号（1kHz和1.01kHz）完全是全通的，实际上只相当于一个移相器，将干扰信号移相到延迟后的相位，所以经过减法得到的误差信号才能和有用信号＋噪声一致。

### 最后


自适应滤波器就介绍这么多了，其实属于比较基础的应用，但对于本科阶段的同学来说已经很难了。由于Simulink和matlab工具箱中都不是开源的，这里同时给出matlab的实现方式，有兴趣的同学可以很容易将它实现在C语言中。

``` 
function [ hRLS ] = fnRLSCreate( nTAPs, lamda )
% Create RLS system object
% 
% nTaps : Taps number
% lamda : Forgetting factor
%
% Miz.Wong, 2018

    hRLS.lamda = lamda;
    hRLS.P_last = .1*eye(nTAPs);
end

```

```
function [ delta_w, hRLS ] = fnRLS( hRLS, vu, e, reset )
% Upgrade RLS system object and calculate delta weight
% 
% vu : u
% e  : error
%
% Miz.Wong, 2018

M = length(vu);

I = eye(M);                 

if reset == 1
    hRLS.P_last = .1*I;
end

% Step 1 . Kalman gain
K = (hRLS.P_last * vu)/(hRLS.lamda + vu'* hRLS.P_last * vu);  
% Step 2 . Calc delta weight
delta_w = conj(K) * e;      
% Step 3 . Upgrade P
P = (hRLS.P_last-K*vu'*hRLS.P_last)/hRLS.lamda; 

hRLS.P_last = P;

end
```

DEMO： (当然这个Demo是用来跟踪一个已知滤波器抽头系数的，不过对于我的RLS函数的用法是相同的)

```
clear all;
close all;

N  = 32;

x  = rand(1,500);
b  = fir1(N-1,0.5);     % FIR system to be identified
n  = 0.01 * rand(1,500); % Observation noise signal
d  = filter(b,1,x)+n;  % Desired signal
lam = 0.9;            % RLS forgetting factor
w  = zeros(N, 1);

hMyRLS = fnRLSCreate(N, lam);

for ii = 1:1:length(x)
    if ii < N + 1
        y(ii) = [zeros(1,N-ii),x(1:ii)] * w;
        e = d(ii) - y(ii);
        [deltaW, hMyRLS] = fnRLS( hMyRLS, [zeros(1,N-ii),x(1:ii)].', e, 0 );
        w = w + deltaW;
    else
        y(ii) = x(ii-N+1:ii) * w;
        e = d(ii) - y(ii);
        [deltaW, hRLS] = fnRLS( hMyRLS, x(ii-N+1:ii).', e, 0 );
        w = w + deltaW;
    end
end

figure;
stem(b);
hold on;
plot(w,'*-');

```

最后，有问题欢迎联系我。 MizWong@Foxmail.com
