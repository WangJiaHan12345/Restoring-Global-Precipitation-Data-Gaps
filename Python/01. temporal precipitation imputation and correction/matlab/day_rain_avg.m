FolderPath=input('���������ݴ洢�ļ���:','s'); %�����ַ�����FolderPath������s��ΪĬ��������ֵ
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('G:\��ҵ����ͼ\��ظ�ԭ\�ռ�\�·���������ͼ\��ˮ��\rain\','��'); %����ļ���·�� cpc Early Final ANN
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

% grid = 16210;
% day = 731; 
% %ȫ��ռ�3-5 6-8:368  9-11:364 12-2:361   1-12:1461
% %ȫ��ʱ�� 1-12:361  3-5 6-8:92  9-11:90  12-2:87
% %��ظ�ԭ�ռ� 3-5 6-8��184  9-11��182   12-2��181
% rain=zeros(day,1);
% parfor k=3:FilesCount 
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%   
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,day,1);
%       fclose(fid); 
%       
%       rain = rain + data
%      
% end
% 
% rain = rain / grid;
% 
% outfile=strcat(SaveFolder,'\','gsmap_mvk_day_rain.txt');  % ANN gsmap_gauge gsmap_mvk  ���������
% 
%  if exist(outfile,'file')~=0 
%     delete(outfile);     
%  end
%  fid1=fopen(outfile,'w');
%      
%  for i=1:1:day
%      for j=1:1:1
%           fprintf(fid1,'%g\r\n',rain(i,j));
%      end   
%  end
%  fclose(fid1);    
% disp('�������')


%����һ���õ�����ƽ����ˮת��Ϊ��ƽ����ˮ
day = 731;
for k=3:FilesCount 
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,day,1);
      fclose(fid); 
      
      goal_day = 24;
      rain=zeros(goal_day,1);
      
      step = [0,31,59,90,120,151,181,212,243,273,304,334,365,396,425,456,486,517,547,578,609,639,670,700,731];
      
      for a = 1:1:goal_day
          for i = step(a)+1:1:step(a+1)
             rain(a,1) = rain(a,1) + data(i,1);
          end
      end
      
      SaveFiles=strcat(Name(1:location(end)-1),'.txt');
      outfile=strcat(SaveFolder,'\',SaveFiles);  % ANN gsmap_gauge gsmap_mvk  ���������

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');

     for i=1:1:goal_day
         for j=1:1:1
              fprintf(fid1,'%g\r\n',rain(i,j));
         end   
     end
     fclose(fid1);    
      
     
end
disp('�������')
