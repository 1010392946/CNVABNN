%% 该代码为基于BP-Adaboost的强分类器分类

%% 清空环境变量
clc
clear

%% 导入训练样本数据
data1=load('\tests\SimulationData_mat\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2=load('\tests\SimulationData_mat\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data3=load('\tests\SimulationData_mat\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data4=load('\tests\SimulationData_mat\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data5=load('\tests\SimulationData_mat\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data6=load('\tests\SimulationData_mat\0.4_6x_mat\sim1_6_6100_read_trains.mat');

data_trains1 = data1.('sim1_4_4100_read_trains');
data_trains2 = data2.('sim1_6_6100_read_trains');
data_trains3 = data3.('sim1_4_4100_read_trains');
data_trains4 = data4.('sim1_6_6100_read_trains');
data_trains5 = data5.('sim1_4_4100_read_trains');
data_trains6 = data6.('sim1_6_6100_read_trains');
data_trains=[data_trains1;data_trains2;data_trains3;data_trains5;data_trains6;];
column=[2,3,4,5];
[m1,n1] = size(data_trains);

trainLines = m1;
gdata(1:trainLines,:) = data_trains(1:trainLines,:);

%从1到trainlines间随机排序
t=rand(1,trainLines);
[m,n]=sort(t);

%得到输入输出数据
ginput=gdata(:,column);
goutput1 =gdata(:,6);
% [a,b]=find(goutput1==1);
%输出从一维变成四维：0正常，1gain，2hemi_loss，3homo_loss;
goutput=zeros(trainLines,4);
for i=1:trainLines
    switch goutput1(i)
        case 0
            goutput(i,:)=[1 0 0 0];
        case 1
            goutput(i,:)=[0 1 0 0];
        case 2
            goutput(i,:)=[0 0 1 0];
        case 3
            goutput(i,:)=[0 0 0 1];
    end
end

%找出训练数据和预测数据
ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';

%样本输入输出数据归一化
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

%% 权重初始化
[mm,nn]=size(ginput_train);
D(1,:)=ones(1,nn)/nn;

%% 中间变量的初始化

precision_sum=[];
sensitivition_sum=[];
error_number=0;

%% BP强分类器设计
k=3;
result_yc = zeros(k,nn);
at=zeros(1,k);
result_yc=zeros(k,nn);
% 保存弱分类器预测
% prediction = cell(k, 1);
for i=1:k
    %初始化网络结构
    net=newff(ginputn,goutput_train,7);
    net.trainParam.epochs=200;
    net.trainParam.lr=0.1;
    net.trainParam.goal=0.00004;
    
    %网络训练
    net=train(net,ginputn,goutput_train);
    
    %训练数据预测
    an=sim(net,ginputn);
    test_sim=mapminmax('reverse',an,outputps);
%     test_simu(i)=test_sim;
    %统计输出结果（以输出错误节点为标准）
    erroryc = abs(test_sim-goutput');
        for j=1:nn
            [x,y]=min(erroryc);
            if y(j)==1
                result_yc(i,j)=0;
            elseif y(j)==2
                result_yc(i,j)=1;
            elseif y(j)==3
                result_yc(i,j)=2;
            elseif y(j)==4
                result_yc(i,j)=3;
            end
        end
%    prediction{i} = result_yc; 
   %统计此次分类器的准确率
   %计算误差
   error(i)=0;
   error_number=0;
   TP_number=0;
   for j=1:nn
       if result_yc(i,j)~=goutput1(j) && result_yc(i,j)~=0  
             error_number=error_number+1;  %统计预测错误的数目
             if result_yc(i,j)~=goutput1(j) 
                 error(i)=error(i)+D(i,j);
             end
       elseif (result_yc(i,j)==goutput1(j)) && (goutput1(j)~=0)
             TP_number=TP_number+1;  %统计预测正确的数目
       end
   end
     
    %统计P
    P_count=0;
    for j=1:nn
        if goutput1(j)==1||goutput1(j)==2||goutput1(j)==3
            P_count=P_count+1;
        end
    end
    
    %计算本次分类器的权重
    at(i) = log((1-error(i))/error(i))+log(3);
    
    %调整D值(以输出错误节点为标准)
    for j=1:nn
       D(i+1,j)=D(i,j)*exp(at(i)*(result_yc(j)~=goutput1(j)));
    end
    
    %D值的归一化
    Dsum=sum(D(i+1,:));
    D(i+1,:)=D(i+1,:)/Dsum; 
    
    %统计所以分类器的敏感度
    sensitivition = TP_number/P_count;
    sensitivition_sum=[sensitivition_sum sensitivition];
    %统计所有分类器的准确率
    TPFP_number=TP_number+error_number;
    precision= TP_number/TPFP_number;
    precision_sum = [precision_sum precision];
    
    %训练网络的保存
    save(['net\BP_Ada_',num2str(i)],'net');
    
end
%保存所有弱分类器的权值,分类器数目
save('Parameters\at','at')
save('Parameters\k','k')


%% 组合弱分类器结果
TP_count=0;
TPFP_count=0;
if k~=1
    result_boost=combine_BP(result_yc,at,nn,k);
    boost_1=length(find(result_boost==1));
    boost_2=length(find(result_boost==2));
    boost_3=length(find(result_boost==3));
    
else
    result_boost=result_yc;
end
for q=1:nn
   if (result_boost(q)==1&&goutput1(q)==1||result_boost(q)==2&&goutput1(q)==2||result_boost(q)==3&&goutput1(q)==3)
      TP_count=TP_count+1;
   end
   if (result_boost(q)==1||result_boost(q)==2||result_boost(q)==3)
      TPFP_count=TPFP_count+1; 
   end
end
precision_boost=TP_count/TPFP_count;
sensitivition_boost=TP_count/P_count;
disp(['训练集准确率：' num2str(precision_boost)]);
disp(['训练集敏感度：' num2str(sensitivition_boost)]);
