close all;

% ����ͼ�����ԣ�����ͼλ�úͳߴ�
set(gcf,'units','inches','position',[0,10.0,14.0,10.0]);
set(gcf,'DefaultAxesFontSize',25,'DefaultAxesFontName','Times New Roman'); % �����������С
%��ȡ���ݣ�sd rmse �� r��
data=xlsread('C:\Users\С��\Desktop\��ҵ���Ĳ���\��ҵ����ͼ\ȫ��\�ռ�\̩��ͼ\̩��ͼָ��.xlsx','Sheet1','B2:D5');%�ļ�·��  

sdev = data(:,1);
crmsd = data(:,2);
ccoef = data(:,3);
%mmodel ID���������ֶ���������ΪҪÿ���������ñ�־
label = containers.Map({'Obs','IMERG-E','IMERG-F','Adb-DNN'}, {'ks','r+', 'kx', 'gs'});
% ID = {'Obs','IMERG-E','IMERG-F','Adb-DNN'};
% label = ID;
%>>���� taylor_diagram
[hp, ht, axl] = taylor_diagram(sdev,crmsd,ccoef, ...
    'markerLabel',label, 'markerLegend', 'on', ...
    'styleSTD', '-', 'colOBS','k', 'markerObs','s', ...
    'markerSize',20, 'tickRMS',(2:2:10),'limSTD',10, ...
    'tickRMSangle', 115, 'showlabelsRMS', 'on', ...
    'titleRMS','on', 'titleOBS','Observation');

% �����ļ�
% writepng(gcf,'s-1333-2.png');
print(gcf,'s-9-11.tif','-r300','-dtiff');%-r600�ɸ�Ϊ300dpi�ֱ���


