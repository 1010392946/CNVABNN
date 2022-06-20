%% Strong classifier classification based on BP-Adaboost

%% Clear environment variables
clc
clear

%% Importing training sample data
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

%Random sorting
t=rand(1,trainLines);
[m,n]=sort(t);

%Get input and output data
ginput=gdata(:,column);
goutput1 =gdata(:,6);

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

%Find the training data and prediction data
ginput_train=ginput(n(1:trainLines),:)';
goutput_train=goutput(n(1:trainLines),:)';

%Normalization
[ginputn,ginputps]=mapminmax(ginput_train);
[outputn,outputps]=mapminmax(goutput_train);

%Weight initialization
[mm,nn]=size(ginput_train);
D(1,:)=ones(1,nn)/nn;

precision_sum=[];
sensitivition_sum=[];
error_number=0;

%% Design of strong classifier
k=3;
result_yc = zeros(k,nn);
at=zeros(1,k);
result_yc=zeros(k,nn);
for i=1:k
    %Initializing the network structure
    net=newff(ginputn,goutput_train,7);
    net.trainParam.epochs=200;
    net.trainParam.lr=0.1;
    net.trainParam.goal=0.00004;
    
    %Network Training
    net=train(net,ginputn,goutput_train);
    
    %Projections
    an=sim(net,ginputn);
    test_sim=mapminmax('reverse',an,outputps);
    
    %Statistical output results
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

   %Statistical accuracy of this weak classifier
   %Calculation error
   error(i)=0;
   error_number=0;
   TP_number=0;
   for j=1:nn
       if result_yc(i,j)~=goutput1(j) && result_yc(i,j)~=0  
             error_number=error_number+1;  %Number of statistical prediction errors
             if result_yc(i,j)~=goutput1(j) 
                 error(i)=error(i)+D(i,j);
             end
       elseif (result_yc(i,j)==goutput1(j)) && (goutput1(j)~=0)
             TP_number=TP_number+1;  %Number of statistical predictions correct
       end
   end
     
    P_count=0;
    for j=1:nn
        if goutput1(j)==1||goutput1(j)==2||goutput1(j)==3
            P_count=P_count+1;
        end
    end
    
    %Calculate the weights of this classifier
    at(i) = log((1-error(i))/error(i))+log(3);
    
    %Adjustment of D value
    for j=1:nn
       D(i+1,j)=D(i,j)*exp(at(i)*(result_yc(j)~=goutput1(j)));
    end
    
    %Normalization of D values
    Dsum=sum(D(i+1,:));
    D(i+1,:)=D(i+1,:)/Dsum; 
    
    %Counting the sensitivity of all classifiers
    sensitivition = TP_number/P_count;
    sensitivition_sum=[sensitivition_sum sensitivition];
    %Counting the precision of all classifiers
    TPFP_number=TP_number+error_number;
    precision= TP_number/TPFP_number;
    precision_sum = [precision_sum precision];
    
    %Save training network
    save(['net\BP_Ada_',num2str(i)],'net');
    
end
save('Parameters\at','at')
save('Parameters\k','k')


%% Combining weak classifiers
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
disp(['Precisiom of the training set：' num2str(precision_boost)]);
disp(['Sensitivity of the training set：' num2str(sensitivition_boost)]);
