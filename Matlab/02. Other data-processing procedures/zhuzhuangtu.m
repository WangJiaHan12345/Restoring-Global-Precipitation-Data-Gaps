%��״ͼ
FolderPath=input('���������ݴ洢�ļ���:','s'); %����ԭʼ����
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ
SaveFolder=strcat('G:\��ظ�ԭ\ʱ��\����\ֻ����վ������final��\','12-2'); 
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');


grid = 168; %������
sum = 0;

for k=3:FilesCount 
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,day,1);
      fclose(fid); 
          
      
       for i=1:1:length(data)
           for j=1:1:1
               sum = sum + data(i,j);
           end
       end
     
end

% cpc  Early  Final  ANN
 outfile=strcat(SaveFolder,'\','ANN_sum_rain.txt');  % ANN gsmap_gauge gsmap_mvk  ���������

 if exist(outfile,'file')~=0 
    delete(outfile);     
 end
 fid1=fopen(outfile,'w');
     
 for i=1:1:1
     for j=1:1:1
          fprintf(fid1,'%g\r\n',sum/grid);
     end   
 end
 fclose(fid1);    
disp('�������')


% % γ�Ƚ�ˮ ��ƽ����ˮ
% FolderPath=input('���������ݴ洢�ļ���:','s'); %�����ַ�����FolderPath������s��ΪĬ��������ֵ
% index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
% SaveFolder=strcat('F:\ȫ��\�ռ�У�����\global_climate\��DNN\1-12\','ANN_lat'); %����ļ���·�� cpc Early Final NN
% if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
%     mkdir(SaveFolder);
% end
% Files=dir(FolderPath);
% FilesCount=length(Files);
% disp('������...');

%��һ��
% day = 1461; %�ռ�1461  ʱ�䣺361
% for i=1:1:232   %�ռ�1��232  ʱ�䣺2��231
%     sum=0;
%     count=0;
%     for k=3:FilesCount 
%           FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%           Name=Files(k).name;
%           location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%           
%           ss = str2num(Name(location(end)-6:location(end)-4));
%           if str2num(Name(location(end)-6:location(end)-4))==i
%               fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%               data = cell2mat(textscan(fid,'%f','headerlines',0));
%               data = reshape(data,day,1);   %������Ҫ����
%               fclose(fid); 
%               
%               rain = 0;
% 
%               for m=1:1:day 
%                    for n=1:1:1
%                        rain = rain + data(m,n);
%                    end
%                end
%  
%               sum = sum + rain;
%               count=count+1;
%           end
%     end
% 
%      SaveFiles=strcat(num2str(i,'%03d'),'.txt'); %CPC����ļ���·��
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
% 
%      fid1=fopen(outfile,'w');
%      
%      %�ռ���4��
%      fprintf(fid1,'%g\r\n',sum/(4*count)); %��ƽ����ˮ��
%      %ʱ����һ��
% %      fprintf(fid1,'%g\r\n',sum/count);
% 
%      fclose(fid1);
% end


% ��һ��������͵Ĳ���
% ���ڵڶ�����
% day = 232;  %ȫ��ռ���232 ʱ����230��
% rain=zeros(day,1);
% a=0;
% 
% for k=3:FilesCount 
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,1);
%       fclose(fid); 
% 
%       a=a+1;
%       rain(a,1)=data;
% end
% 
% 
%  outfile=strcat(SaveFolder,'\','sum.txt');
% 
%  if exist(outfile,'file')~=0 
%     delete(outfile);     
%  end
% 
%  fid1=fopen(outfile,'w');
% 
%  for i=1:1:day
%      for j=1:1:1
%           fprintf(fid1,'%g\r\n',rain(i,j));
%      end   
%  end
%  fclose(fid1);

 disp('�������')
% 
% % 
% % % �·ݺ���
% % FolderPath=input('���������ݴ洢�ļ���:','s'); %�����ַ�����FolderPath������s��ΪĬ��������ֵ
% % index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
% % SaveFolder=strcat('F:\���\shirun\global_s\12-2\global\','NN_rain_sum'); %����ļ���·�� cpc Early Final NN
% % if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
% %     mkdir(SaveFolder);
% % end
% % Files=dir(FolderPath);
% % FilesCount=length(Files);
% % disp('������...');
% % 
% % sum = zeros(364,1);
% % for k=3:FilesCount 
% %       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
% %       Name=Files(k).name;
% %       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
% %   
% %       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
% %       data = cell2mat(textscan(fid,'%f','headerlines',0));
% %       data = reshape(data,364,1);
% %       
% %     
% %       
% %        for i=1:1:364
% %            for j=1:1:1
% %                sum(i,j) = sum(i,j) + data(i,j);
% %            end
% %        end
% %        fclose(fid);  
% % end
% %        
% % sum = sum/(FilesCount-2);
% %  
% % outfile=strcat(SaveFolder,'\','rain_sum_heng_yuefen.txt'); 
% % fid_1=fopen(outfile,'w');
% % 
% % a=0;b=0;c=0;d=0;e=0;f=0;g=0;h=0;m=0;n=0;o=0;p=0;
% % data = zeros(12,1);
% % for i=1:1:31                      
% %     a=a+sum(i,1);
% % end
% % 
% % for i=32:1:59
% %     b=b+sum(i,1);
% % end
% % 
% % for i=60:1:90
% %     c=c+sum(i,1);
% % end
% % 
% % 
% % 
% % for i=91:1:121
% %     d=d+sum(i,1);
% % end
% % 
% % for i=122:1:150
% %     e=e+sum(i,1);
% % end
% % 
% % for i=151:1:181           
% %     f=f+sum(i,1);
% % end
% % 
% % 
% % for i=182:1:212
% %     g=g+sum(i,1);
% % end
% % 
% % for i=213:1:240
% %     h=h+sum(i,1);
% % end
% % 
% % for i=241:1:271
% %     m=m+sum(i,1);
% % end
% % 
% % 
% % 
% % for i=272:1:302
% %     n=n+sum(i,1);
% % end
% % 
% % for i=303:1:330
% %     o=o+sum(i,1);
% % end
% % 
% % 
% % for i=331:1:364
% %     p=p+sum(i,1);
% % end
% % 
% % data(1,1)=a;data(2,1)=b;data(3,1)=c;
% % data(4,1)=d;data(5,1)=e;data(6,1)=f;
% % data(7,1)=g;data(8,1)=h;data(9,1)=m;
% % data(10,1)=n;data(11,1)=o;data(12,1)=p;
% % 
% % 
% %  for i=1:1:12
% %          fprintf(fid_1,'%g\r\n',data(i,1));
% %  end
% %  
% %  fclose(fid_1); 
% %  
% % disp('�������')