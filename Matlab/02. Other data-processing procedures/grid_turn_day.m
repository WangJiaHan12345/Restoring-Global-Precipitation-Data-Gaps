% % ���������ʽ��Ϊ���ڵ���ʽ
FolderPath=input('���������ݴ洢�ļ���:','s');  %�������ڱ��
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
Files=dir(FolderPath);
FilesCount=length(Files);
% 
SaveFolder=strcat('G:\ȫ��\ʱ��Ԥ����\����վ��_final\global_climate\1-12\������ʽ\','Early����'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end

%ֻ���ڵ�һ��������
FolderPath1=input('���������ݴ洢�ļ���:','s');  %������������  
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

% ����������תΪ����
% for k=3:357
%     
% %      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
% %      Name=Files(k).name;
% %      location=strfind(Name,'.');
%       
%         
%      result = zeros(6420,1);  %168��������ĸ���
%       
%      for m = 3:FilesCount1
%          
%           FilePath1 = strcat(FolderPath1,'\',Files1(m).name);
%           fid = fopen(FilePath1,'rb','l');
%           data = cell2mat(textscan(fid,'%f','headerlines',0));
%           data = reshape(data,1,355);   
%           data = data'; 
%           fclose(fid); 
%           
%           
%           result(m-2,1) = data(k-2,1); 
%          
%      end
%       
%      
%      SaveFiles=strcat(num2str(k-2),'.txt'); %CPC����ļ���·��
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%      
%      
%      fid1=fopen(outfile,'w');
%      
%      for i=1:1:6420
%          for j=1:1:1
%              if j==1
%                  fprintf(fid1,'%g\r\n',result(i,j));
%              else
%                  fprintf(fid1,'%g ',result(i,j));
%              end
%          end   
%      end
%      fclose(fid1);         
%     
% end

%�������ݴ�������ȡ����
for k=3:FilesCount   %  6420���ݵ��ļ�
    
     FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
     Name=Files(k).name;
     location=strfind(Name,'.');
      
        
     result = zeros(240,720);  %168��������ĸ���
         
        for a=1:1:240
            for b = 1:1:720
                result(a,b) = -9999;
            end
        end
     
    
      fid = fopen(FilePath,'rb','l');
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,6420);   
      data = data'; 
      fclose(fid); 
          
      for m =3:FilesCount1   % 6420������ʽ���ļ�
          Name1=Files1(m).name;
          location1=strfind(Name1,'.');  %����ַ�'.'��FilePath��λ��


          i= str2num(Name1(location1(end)-6:location1(end)-4));
          j= str2num(Name1(location1(end)-3:location1(end)-1));
          
          result(i,j)= data(m-2,1);
      end
          
  
     SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC����ļ���·��

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     
     
     fid1=fopen(outfile,'w');
     fprintf(fid1,'NCOLS        720\r\nNROWS        240\r\nXLLCORNER   0\r\nYLLCORNER   -60\r\nCELLSIZE    0.5\r\nNODATA_VALUE    -9999\r\n');
     
     for i=1:1:240
         for j=1:1:720
             if j==1
                 fprintf(fid1,'%g\r\n',result(i,j));
             else
                 fprintf(fid1,'%g ',result(i,j));
             end
         end   
     end
     fclose(fid1);         
    
end




% % ������תΪ����
% %������ת��Ϊ��������
% % ���� gsmap�����ļ�  H:\GSMAP����\��ȡ�������-0.5\gsmap_mvk
% FolderPath1=input('���������ݴ洢�ļ���:','s'); 
% index1=strfind(FolderPath1,'\');  %����ַ�'\'��FolderPath��λ��
% SaveFolder=strcat('H:\��ظ�ԭ����\ʱ��Ԥ��\2015-2017\result(���ֿ�)\���������������\������ʽ\','ANN'); %����ļ���·��
% if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
%     mkdir(SaveFolder);
% end
% Files1=dir(FolderPath1);
% FilesCount1=length(Files1);
% 
% % �����涨��Ч����
% FolderPath2=input('���������ݴ洢�ļ���:','s');  %������������  
% Files2=dir(FolderPath2);
% FilesCount2=length(Files2);
% 
% %�����������ļ���
% for k=3:FilesCount2
%       Name=Files2(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%       
%       
% %       i= str2num(Name(location(end)-6:location(end)-4));
% %       j= str2num(Name(location(end)-3:location(end)-1));
%       
%       
%       result = zeros(109,1);  % 365  1096  1461
%       
%       for m =3:FilesCount1
%           FilePath1 = strcat(FolderPath1,'\',Files1(m).name);
%           fid = fopen(FilePath1,'rb','l');
%           data = cell2mat(textscan(fid,'%f','headerlines',0));
%           data = reshape(data,1,168);   %700 440   ���� 700 400
%           data = data'; 
%           fclose(fid); 
%           
% 
%           result(m-2,1)=data(k-2,1); 
%           
%       end
%       
%      SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%           
%      fid1=fopen(outfile,'w');
%      
%      for i=1:1:109 
%          for j=1:1:1
%              if j==1
%                  fprintf(fid1,'%g\r\n',result(i,j));
%              else
%                  fprintf(fid1,'%g ',result(i,j));
%              end
%          end   
%      end
%      fclose(fid1);         
% end
% 
% 
% 


% %ɾ������-999������
% FolderPath1=input('���������ݴ洢�ļ���:','s'); 
% index1=strfind(FolderPath1,'\');  %����ַ�'\'��FolderPath��λ��
% Files1=dir(FolderPath1);
% FilesCount1=length(Files1);
% 
% 
% for k=3:FilesCount1
%       FilePath1 = strcat(FolderPath1,'\',Files1(k).name);
%       
%       fid = fopen(FilePath1,'rb','l');
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,168);   %700 440   ���� 700 400
%       data = data'; 
%       fclose(fid);
%       
%       for m =1:1:168
%           if data(m,1) <0
%               delete(FilePath1);
%           end
%       end
%         
% end
