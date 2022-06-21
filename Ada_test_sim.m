%% This script is the test set classification process for CNVABNN

%% Clear environment variables
clc
clear

%% Import the parameters of the trained neural network
load('-mat','\K');
load('-mat','\at');
%% Import groundtruth file
data_g=load('\data\groundtruth.mat');
data_gt=data_g.('groundtruth');

%% Obtain normalized information in training samples
data1=load('\0.2_4x_mat\sim1_4_4100_read_trains.mat');
data2=load('\0.3_4x_mat\sim1_4_4100_read_trains.mat');
data3=load('\0.4_4x_mat\sim1_4_4100_read_trains.mat');
data4=load('\0.2_6x_mat\sim1_6_6100_read_trains.mat');
data5=load('\0.3_6x_mat\sim1_6_6100_read_trains.mat');
data6=load('\0.4_6x_mat\sim1_6_6100_read_trains.mat');

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
[ginputn,ginputs]=mapminmax(ginput_train);
[goutputn,outputs]=mapminmax(goutput_train);

g_column=[1,2];
groundtruth=data_gt(:,g_column);
gtRev=fliplr(groundtruth(:,2)');
%% Test section
sample = 50;
covery=[2,3,4];
purity=[4,6];
boundary=[]; %For generating box plots
count_bias_sum=0;
num_boundary=0;
precision_boost_sum=[];
sensitivition_boost_sum=[];
P_count=0;
for temp= covery
    for temp2=purity
        for i=1:sample
            %Importing test sample data
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

            %Obtain normalized test samples
            input_test=mapminmax('apply',ginput_test,ginputs);
            %% Weak classifier training
            for j=1:k
                %Load k weak classifier networks
                load('-mat',['net\BP_Ada_',num2str(j)]);
                %Predicted output results
                an = sim(net,input_test);
   
                %inverse normalization
                BPoutput=mapminmax('reverse',an,outputs);
    
                %Statistical prediction error
                error=abs(BPoutput-goutput_insim);

                P_count=0;
                TP_count=0;
                for q=1:m2
                % calculate TP_count,P_count,TPFP_count,boundary
                    if (( error(2,q) < error(1,q) && error(2,q) < error(3,q) && error(2,q) < error(4,q) && goutput_insim(2,q) == 1) || ( error(3,q) < error(1,q) && error(3,q) < error(2,q) && error(3,q) < error(4,q) && goutput_insim(3,q) == 1) || (error(4,q) < error(1,q) && error(4,q) < error(2,q) && error(4,q) < error(3,q) && goutput_insim(4,q) == 1)) 
                        TP_count=TP_count+1; %calculate TP
                    end

                    if ( goutput_insim(2,q) == 1 || goutput_insim(3,q) == 1 || goutput_insim(4,q) == 1 )
                        P_count=P_count+1;
                    end
                end
                %Classification results
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
          

            %% Combining weak classifiers
            TP_count_boost=0;
            TPFP_count_boost=0;
            result_boost=combine_BP(result_yc,at,m2,k);
            %% Calculate the boundaries of the samples
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
                    TPFP_count_boost=TPFP_count_boost+1; %TP+FP
                end
            end
            binnumberRev=fliplr(binnumber);
            groundtruth1=fliplr(groundtruth(:,1)');
            [mm,nn]=size(binnumber);
            %Calculate the boundary error
            j1=1;
            bin_l=[];
            j2=1;
            bin_r=[];
            for i1=1:nn
                if binnumber(i1)==binnumber(nn) %Last detection position
                    if length(bin_l)<14
                        if abs(binnumber(i1)-groundtruth(j1,1)) < abs(binnumber(i1)-groundtruth(j1,2))
                            bin_l(j1)=binnumber(i1);
                        else
                            bin_l(j1)=groundtruth(j1);
                        end
                    end
                    break
                end
                if j1==14 %Prevent crossing the border
                    if binnumber(i1) >= groundtruth(j1,1)
                        bin_l(j1)=binnumber(i1);
                        break;
                    end
                else
                   if binnumber(i1) >= groundtruth(j1,1) && binnumber(i1) <= groundtruth(j1,2) && binnumber(i1+1) >= groundtruth(j1,1) && binnumber(i1+1) <= groundtruth(j1,2)%区间内，同时不为一个点的情况
                       bin_l(j1)=binnumber(i1);
                       j1=j1+1;
                   elseif binnumber(i1) == groundtruth(j1,2) %Extreme case where only the right endpoint is detected
                       bin_l(j1)=groundtruth(j1,1);
                       j1=j1+1;                      
                   elseif binnumber(i1) >= groundtruth(j1+1,1) %Large segment CNV detection failure
                       while binnumber(i1) >= groundtruth(j1+1,1)
                           bin_l(j1)=groundtruth(j1,1);
                           j1=j1+1;
                       end
                       if binnumber(i1) >= groundtruth(j1,1) && binnumber(i1) <= groundtruth(j1,2) && binnumber(i1+1) >= groundtruth(j1,1) && binnumber(i1+1) <= groundtruth(j1,2) 
                           bin_l(j1)=binnumber(i1);
                           j1=j1+1;
                       else
                           bin_l(j1)=groundtruth(j1,1);
                           j1=j1+1; 
                       end   
                        if j1>14
                            break; 
                        end
                        
                   end                
                end    
            end
            if length(bin_l)<14 %When all samples are trained and there are still less than 14 bars, press ground truth to complete
                l_l = length(bin_l);
                while(l_l<14)
                    l_l=l_l+1;
                    bin_l(l_l)=groundtruth(l_l,1);                   
                end
            end
            for i2=1:nn
                if binnumberRev(i2)==binnumberRev(nn) %Last detection position
                    if length(bin_r)<14
                        if abs(binnumberRev(i2)-groundtruth1(j2)) > abs(binnumberRev(i2)-gtRev(j2))
                            bin_r(j2)=binnumberRev(i2);
                        else
                            bin_r(j2)=gtRev(j2);
                        end
                    end
                    break
                end
                if j2==14 %The last CNV
                    if binnumberRev(i2) <= gtRev(j2)
                        bin_r(j2)=binnumberRev(i2);
                        break;
                    end
                else
                   if binnumberRev(i2) <= gtRev(j2) && binnumberRev(i2) >= groundtruth1(j2) && binnumberRev(i2+1) <= gtRev(j2) && binnumberRev(i2+1) >= groundtruth1(j2)
                        bin_r(j2)=binnumberRev(i2);
                        j2=j2+1;
                   elseif binnumberRev(i2)==groundtruth1(j2)
                       bin_r(j2)=gtRev(j2);
                       j2=j2+1;
                   elseif binnumberRev(i2) <= gtRev(j2+1) %Screening
                       while binnumberRev(i2) <=gtRev(j2+1) && j2<13
                           bin_r(j2)=gtRev(j2);
                           j2=j2+1;
                           cnv=cnv-1;
                       end
                        if binnumberRev(i2) <= gtRev(j2) && binnumberRev(i2) > groundtruth1(j2) && binnumberRev(i2+1) <= gtRev(j2) && binnumberRev(i2+1) >= groundtruth1(j2)
                            bin_r(j2)=binnumberRev(i2);
                            j2=j2+1;
                        else
                            bin_r(j2)=gtRev(j2);
                            j2=j2+1;
                        end
                        if j2>14
                            break; 
                        end
                   end                
                end           
            end
            if length(bin_r)<14 %Usually not used
                l_r = length(bin_r);
                while(l_r<14)
                    l_r=l_r+1;
                    bin_r(l_r)=grRev(l_r);                   
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
            %Calculating the sensitivity of a strong classifier
            sensitivition_boost=TP_count_boost/P_count;
            %Calculating the precision of a strong classifier
            precision_boost = TP_count_boost/TPFP_count_boost;
            precision_boost_sum=[precision_boost_sum precision_boost];
            sensitivition_boost_sum=[sensitivition_boost_sum sensitivition_boost];
           
        end
        disp(['Test set（0.',num2str(temp),'_',num2str(temp2),'）precision：' num2str(mean(precision_boost_sum))]);
        %sensitivity
        disp(['Test set（0.',num2str(temp),'_',num2str(temp2),'）sensitivity：' num2str(mean(sensitivition_boost_sum))]);
        %F1-score
        F1_score=(2*mean(sensitivition_boost_sum)*mean(precision_boost_sum))/(mean(sensitivition_boost_sum)+mean(precision_boost_sum));
        disp(['Test set/F1-score:' num2str(F1_score)]);
        %save
        save(['\boundary_0.',num2str(temp),'_',num2str(temp2),'.mat'],'boundary')
    end
end

length_real_gain=(find(goutput_test==1));
length_real_less=(find(goutput_test==2));
length_real_less2=(find(goutput_test==3));
boost_1=find((result_boost==1));
boost_2=find((result_boost==2));
boost_3=find((result_boost==3));
