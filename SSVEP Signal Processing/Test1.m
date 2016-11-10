

weight_factor=0.32;
for ch = 1:2
FEAT.low(1,ch)  = weight_factor*PSD(ch,f==7.5,1)+...
            (1-weight_factor)*weight_factor*PSD(ch,f==15,1);
FEAT.high(1,ch) = weight_factor*PSD(ch,f==12.25,1)+...
            (1-weight_factor)*PSD(ch,f==37,1);
FEAT.ref(1,ch) = mean(PSD(ch,40:1:44,1));
end
input = cat[2,FEAT.low,FEAT.high,FEAT.ref];
% 0vs12

Label = FEAT.labels;
FEAT.label0 = Label;
FEAT.label0(FEAT.label0==2)=-1;
FEAT.label0(FEAT.label0==1)=-1;
FEAT.label0(FEAT.label0==0)=1;
% 1vs2
FEAT.allFeature_1 = FEAT.allFeatures;
FEAT.allFeature_1(FEAT.label1==0,:) = [];
FEAT.label1(FEAT.label1==0)=[];
FEAT.label1(FEAT.label1==2)=-1;
FEAT.label1(FEAT.label1==1)=1;

%train 0vs12
FEAT.allFeature0=[FEAT.allFeatures(:,5) FEAT.allFeatures(:,6)];
model.class0 = trainShrinkLDA(FEAT.allFeature0,FEAT.label0,0.1,'');

%train 1vs2
FEAT.allFeature1=[FEAT.allFeatures(:,3) FEAT.allFeatures(:,2) FEAT.allFeatures(:,1)];
model.class1 = trainShrinkLDA(FEAT.allFeature1,FEAT.label1,0.1,'');


y0 = predictShrinkLDA(model.class0,input);
y1 = predictShrinkLDA(model.class1,input);

if y0 == 1
    y = 0;
elseif y1 == 1
    y = 1;
else
    y = 2;
end