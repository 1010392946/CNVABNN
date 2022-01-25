%% 该部分为BP_Adaboost算法的测试集分类过程

%% 清空环境变量
clc
clear

%% 导入训练好的BP神经网络参数
load('-mat','Parameters\K');
load('-mat','Parameters\at');
% load('-mat','C:\Users\王煊\Desktop\深度学校多目标\学长\对比算法\mfcnv\mfcnv-master\mfcnv-master\BPNN');
%% 获取训练样本中的归一化说明文件
data1=load('\tests\SimulationData_mat\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2=load('\tests\SimulationData_mat\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data3=load('\tests\SimulationData_mat\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data4=load('\tests\SimulationData_mat\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data5=load('\tests\SimulationData_mat\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data6=load('\tests\SimulationData_mat\0.4_6x_mat\sim1_6_6100_read_trains.mat');

data_trains1 = data1.('sim1_4_4100_read_trains');
data_trains2 = data2.('sim1_4_4100_read_trains');
data_trains3 = data3.('sim1_4_4100_read_trains');
data_trains4 = data4.('sim1_6_6100_read_trains');
data_trains5 = data5.('sim1_6_6100_read_trains');
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
[ginputn,ginputs]=mapminmax(ginput_train);
[goutputn,outputs]=mapminmax(goutput_train);

%获取groundtruth

%% 测试部分

%导入测试样本数据
datat=load('\tests\RealData_mat\NA19240_tests.mat');
data_tests = datat.('NA19240_tests');
[m2,n2]=size(data_tests);
ginput2_bin=data_tests(:,1);
ginput_test = data_tests(:,column);
goutput_test = data_tests(:,end);
goutput_insim = zeros(m2,4);
result_yc=zeros(k,m2);
for j=1:m2
    if goutput_test(j)==0
            goutput_insim(j,:) = [1 0 0 0];
    elseif goutput_test(j)==1
            goutput_insim(j,:) = [0 1 0 0];
    elseif goutput_test(j)==2
            goutput_insim(j,:) = [0 0 1 0];
    elseif goutput_test(j) == 3
            goutput_insim(j,:) = [0 0 0 1];
    end
end
ginput_test=ginput_test((1:m2),:)';
goutput_insim=goutput_insim((1:m2),:)';
    
%获取归一化的测试输入样本数据
input_test=mapminmax('apply',ginput_test,ginputs);
precision_sum=[];
sensitivition_sum=[];   
for i=1:k
    %加载k个弱分类器网络
    load('-mat',['\net\BP_Ada_',num2str(i)]);
    %网络预测输出
    an = sim(net,input_test);
   
    %反归一化预测输出
    BPoutput=mapminmax('reverse',an,outputs);
    
    %统计预测误差
    error=abs(BPoutput-goutput_insim);

    P_count=0;
    TP_count=0;
    TPFP_count=0;
    precision=0;
    sensitivition=0;

    for q=1:m2
        if (( error(2,q) < error(1,q) && error(2,q) < error(3,q) && error(2,q) < error(4,q) && goutput_insim(2,q) == 1) || ( error(3,q) < error(1,q) && error(3,q) < error(2,q) && error(3,q) < error(4,q) && goutput_insim(3,q) == 1) || (error(4,q) < error(1,q) && error(4,q) < error(2,q) && error(4,q) < error(3,q) && goutput_insim(4,q) == 1)) 
            TP_count=TP_count+1; 
        end
        if ( goutput_insim(2,q) == 1 || goutput_insim(3,q) == 1 || goutput_insim(4,q) == 1 )
            P_count=P_count+1;
        end
        if (  (error(2,q) < error(1,q) && error(2,q) < error(3,q) && error(2,q) < error(4,q) ) || ( error(3,q) < error(1,q) && error(3,q) < error(2,q) && error(3,q) < error(4,q)) || (error(4,q) < error(1,q) && error(4,q) < error(2,q) && error(4,q) < error(3,q)) )
            TPFP_count=TPFP_count+1; 
        end
    end
    %分类结果
    for j=1:m2
        [x,y]=min(error);
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

    sensitivition = TP_count/P_count;
    sensitivition_sum=[sensitivition_sum sensitivition];

    precision= TP_count/TPFP_count;
    precision_sum = [precision_sum precision];
end
 
%% 组合弱分类器
TP_count_boost=0;
TPFP_count_boost=0;
result_boost=combine_BP(result_yc,at,m2,k); %生成强预测结果
for q=1:m2
   if (result_boost(q)==1&&goutput_test(q)==1||result_boost(q)==2&&goutput_test(q)==2||result_boost(q)==3&&goutput_test(q)==3)
      TP_count_boost=TP_count_boost+1;
   end

   if (result_boost(q)==1)
      TPFP_count_boost=TPFP_count_boost+1; 
   end
end

%计算强分类敏感度
sensitivition_boost=TP_count_boost/P_count;

%计算强分类准确率
precision_boost = TP_count_boost/TPFP_count_boost;

%输出准确率信息
disp(['真实集的准确率为：' num2str(precision_boost)]);

%输出敏感信息
disp(['真实集的敏感度为：' num2str(sensitivition_boost)]);

%输出F1-score
F1_score=(2*sensitivition_boost*precision_boost)/(sensitivition_boost+precision_boost);
disp(['真实集的F1-score为:' num2str(F1_score)]);
