close all;

% 设置图框属性，包括图位置和尺寸
set(gcf,'units','inches','position',[0,10.0,14.0,10.0]);
set(gcf,'DefaultAxesFontSize',25,'DefaultAxesFontName','Times New Roman'); % 坐标轴字体大小
%读取数据，sd rmse 和 r方
data=xlsread('C:\Users\小关\Desktop\毕业论文材料\毕业论文图\全球\空间\泰勒图\泰勒图指标.xlsx','Sheet1','B2:D5');%文件路径  

sdev = data(:,1);
crmsd = data(:,2);
ccoef = data(:,3);
%mmodel ID，我这里手动输入是因为要每个单独设置标志
label = containers.Map({'Obs','IMERG-E','IMERG-F','Adb-DNN'}, {'ks','r+', 'kx', 'gs'});
% ID = {'Obs','IMERG-E','IMERG-F','Adb-DNN'};
% label = ID;
%>>绘制 taylor_diagram
[hp, ht, axl] = taylor_diagram(sdev,crmsd,ccoef, ...
    'markerLabel',label, 'markerLegend', 'on', ...
    'styleSTD', '-', 'colOBS','k', 'markerObs','s', ...
    'markerSize',20, 'tickRMS',(2:2:10),'limSTD',10, ...
    'tickRMSangle', 115, 'showlabelsRMS', 'on', ...
    'titleRMS','on', 'titleOBS','Observation');

% 保存文件
% writepng(gcf,'s-1333-2.png');
print(gcf,'s-9-11.tif','-r300','-dtiff');%-r600可改为300dpi分辨率


