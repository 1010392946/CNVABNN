%% 该部分为BP_Adaboost算法的测试集分类过程

%% 清空环境变量
clc
clear

%% 导入训练好的BP神经网络参数
% load('-mat','C:\Users\王煊\Desktop\深度学校多目标\学长\测试程序matlab代码\BP\chapter5\BP_Ada');
load('-mat','parameter\K');
load('-mat','parameter\at');
%% 导入groundtruth
data_g=load('data\groundtruth.mat');
data_gt=data_g.('groundtruth');

%% 获取训练样本中的归一化说明文件
data1=load('sim1_4_4100_read_trains.mat');
data2=load('sim1_4_4100_read_trains.mat');
data3=load('sim1_4_4100_read_trains.mat');
data4=load('sim1_6_6100_read_trains.mat');
data5=load('sim1_6_6100_read_trains.mat');
data6=load('sim1_6_6100_read_trains.mat');

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

%预处理边界数据
g_column=[1,2];
groundtruth=data_gt(:,g_column);
gtRev=fliplr(groundtruth(:,2)');
%% 测试部分
sample = 50;
covery=[2,3,4];
purity=[4,6];
boundary=[]; %每个bam的边界，用于生成箱图
count_bias_sum=0;
num_boundary=0;
% precision_sum=[];
% sensitivition_sum=[];
precision_boost_sum=[];
sensitivition_boost_sum=[];
P_count=0;
for temp= covery
    for temp2=purity
        for i=1:sample
            %导入测试样本数据
            data=load(['data\tests\SimulationData_mat\0.',num2str(temp),'_',num2str(temp2),'x_mat\sim', num2str(i) ,'_',num2str(temp2),'_',num2str(temp2),'100_read_trains.mat']);
            data_tests = data.(['sim', num2str(i) ,'_',num2str(temp2),'_',num2str(temp2),'100_read_trains']);
            [m2,n2]=size(data_tests);
            ginput_bin=data_tests(:,1);
            ginput_test = data_tests(:,column);
            goutput_test = data_tests(:,end);
            goutput_insim = zeros(m2,4);
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
          %% 弱分类器训练
            for j=1:k
                %加载k个弱分类器网络
                load('-mat',['net\BP_Ada_',num2str(j)]);
                %网络预测输出
                an = sim(net,input_test);
   
                %反归一化预测输出
                BPoutput=mapminmax('reverse',an,outputs);
    
                %统计预测误差
                error=abs(BPoutput-goutput_insim);

                P_count=0;
                TP_count=0;
                for q=1:m2
                % 4.7 统计TP_count,P_count,TPFP_count,boundary
                    if (( error(2,q) < error(1,q) && error(2,q) < error(3,q) && error(2,q) < error(4,q) && goutput_insim(2,q) == 1) || ( error(3,q) < error(1,q) && error(3,q) < error(2,q) && error(3,q) < error(4,q) && goutput_insim(3,q) == 1) || (error(4,q) < error(1,q) && error(4,q) < error(2,q) && error(4,q) < error(3,q) && goutput_insim(4,q) == 1)) 
                        TP_count=TP_count+1; %统计真阳性值
                    end

                    if ( goutput_insim(2,q) == 1 || goutput_insim(3,q) == 1 || goutput_insim(4,q) == 1 )
                        P_count=P_count+1;
                    end
                end
            %分类结果
                for z=1:m2
                    [x,y]=min(error);
                    if y(z)==1
                        result_yc(j,z)=0;
                    elseif y(z)==2
                        result_yc(j,z)=1;
                    elseif y(z)==3
                        result_yc(j,z)=2;
                    elseif y(z)==4
                        result_yc(j,z)=3;
                    end
                end
            end
          

          %% 组合弱分类器
            TP_count_boost=0;
            TPFP_count_boost=0;
            result_boost=combine_BP(result_yc,at,m2,k);
           %% 统计每个样本仿真数据边界
            k1=1;k2=1;
            bound=0;
            binnumber=[];
            cnv=14;
            for q=1:m2
                if (result_boost(q)==1&&goutput_test(q)==1||result_boost(q)==2&&goutput_test(q)==2||result_boost(q)==3&&goutput_test(q)==3)
                    TP_count_boost=TP_count_boost+1;
                    binnumber(k1)=ginput_bin(q);
                    k1=k1+1;
                end
                if (result_boost(q)==1||result_boost(q)==2||result_boost(q)==3)
                    TPFP_count_boost=TPFP_count_boost+1; %表示全部值，有对有错，表示TP+FP
                end
            end
            binnumberRev=fliplr(binnumber);
            [mm,nn]=size(binnumber);
            %计算边界误差
            j1=1;
            bin_l=[];
            j2=1;
            bin_r=[];
            for i1=1:nn
                if j1==14 %最后一个CNV
                    if binnumber(i1) >= groundtruth(j1,1)
                        bin_l(j1)=binnumber(i1);
                        break;
                    end
                else
                   if binnumber(i1) >= groundtruth(j1,1) && binnumber(i1) < groundtruth(j1+1,1)
                        bin_l(j1)=binnumber(i1);
                        j1=j1+1;
                   elseif binnumber(i1) >= groundtruth(j1+1,1) %筛去预判错误的CNV
                        bin_l(j1)=groundtruth(j1,1);
                        j1=j1+1;
                        bin_l(j1)=binnumber(i1);
                        j1=j1+1;
                        if j1>14
                            break; 
                        end
                   end                
                end    
            end
            for i2=1:nn
                if j2==14 %最后一个CNV
                    if binnumberRev(i2) <= gtRev(j2)
                        bin_r(j2)=binnumberRev(i2);
                        break;
                    end
                else
                   if binnumberRev(i2) <= gtRev(j2) && binnumberRev(i2) > gtRev(j2+1)
                        bin_r(j2)=binnumberRev(i2);
                        j2=j2+1;
                   elseif binnumberRev(i2) <= gtRev(j2+1) %筛去预判错误的CNV
                        bin_r(j2)=gtRev(j2);
                        j2=j2+1;
                        cnv=cnv-1; %实际CNV个数
                        bin_r(j2)=binnumberRev(i2);
                        j2=j2+1;
                        if j2>14
                            break; 
                        end
                   end                
                end           
            end
   
            bin_r=fliplr(bin_r);
            bin=[bin_l;bin_r]'; 
            [m_bin,n_bin]=size(bin);
            c_bias_l=[];
            c_bias_r=[];
            for rr=1:m_bin
                c_bias_l(rr)=bin(rr,1)-groundtruth(rr,1);
                c_bias_r(rr)=groundtruth(rr,2)-bin(rr,2);
            end
            c_bias=sum(c_bias_l)+sum(c_bias_r);
            count_bias=c_bias./cnv;
   
            boundary(i)=count_bias;
            num_boundary = num_boundary + 1;
            count_bias_sum = count_bias_sum+count_bias;
            %计算强分类敏感度
            sensitivition_boost=TP_count_boost/P_count;
            %计算强分类准确率
            precision_boost = TP_count_boost/TPFP_count_boost;
            precision_boost_sum=[precision_boost_sum precision_boost];
            sensitivition_boost_sum=[sensitivition_boost_sum sensitivition_boost];
           
        end
        %输出准确率信息
        disp(['测试集（0.',num2str(temp),'_',num2str(temp2),'）的准确率为：' num2str(mean(precision_boost_sum))]);
        %输出敏感信息
        disp(['测试集（0.',num2str(temp),'_',num2str(temp2),'）的敏感度为：' num2str(mean(sensitivition_boost_sum))]);
        %输出F1-score
        F1_score=(2*mean(sensitivition_boost_sum)*mean(precision_boost_sum))/(mean(sensitivition_boost_sum)+mean(precision_boost_sum));
        disp(['测试集的F1-score为:' num2str(F1_score)]);
        %保存边界信息
        save(['\boundary_0.',num2str(temp),'_',num2str(temp2),'.mat'],'boundary')
    end
end

length_real_gain=(find(goutput_test==1));
length_real_less=(find(goutput_test==2));
length_real_less2=(find(goutput_test==3));
boost_1=find((result_boost==1));
boost_2=find((result_boost==2));
boost_3=find((result_boost==3));