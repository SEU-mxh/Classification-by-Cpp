clear
close all 
clc

%% 训练集可视化
%对于类似的txt文件，不含有字符，只有数字
data=load('E:\VScode\人工智能作业2\题目\train.txt');
attr1=data(:,1);
attr2=data(:,2);
labels=data(:,3);
figure(1)
gscatter(attr1,attr2,labels,'rkgb','o',4,'on','x1','x2')
title('训练集')

%% 测试集处理
figure(2)
cdata=load('E:\VScode\人工智能作业2\题目\test.txt');
cattr1=cdata(:,1);
cattr2=cdata(:,2);
scatter(cattr1,cattr2,24)
title('测试集')
xlabel('x1'),ylabel('x2')

