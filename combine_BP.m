function result_boost = combine(result_yc,at,nn,k)
%此函数主要实现若分类器的组合
%归一化at
at=at/sum(at);
result_boost=zeros(1,nn);

%组合所有弱分类器的不同类别占比
for i=1:nn
    p_0=0;p_1=0;p_2=0;p_3=0;
    for j=1:k
%         result_yc = prediction{j};
        if result_yc(j,i)==0
            p_0=p_0+at(j);
        elseif result_yc(j,i)==1
            p_1=p_1+at(j);
        elseif result_yc(j,i)==2
            p_2=p_2+at(j);
        elseif result_yc(j,i)==3
            p_3=p_3+at(j);
        end
    end
    if p_0>=p_1 &&p_0>=p_2&&p_0>=p_3
        result_boost(i)=0;
    elseif p_1>=p_0 &&p_1>=p_2&&p_1>=p_3
        result_boost(i)=1;
    elseif p_2>=p_0 &&p_2>=p_1&&p_2>=p_3
        result_boost(i)=2;
    elseif p_3>=p_0 &&p_3>=p_1&&p_3>=p_2
        result_boost(i)=3;
    end
end
end

