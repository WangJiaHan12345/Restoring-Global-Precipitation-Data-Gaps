% %��tifͼƬ����ȡ�����ݣ���Ϊtxt��ʽ
% FolderPath=input('���������ݴ洢�ļ���:','s');  %����tif�ļ�������
% index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
% 
% SaveFolder=strcat('H:\�й���������\�¶�\','���������'); %����ļ���·��
% if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
%     mkdir(SaveFolder);
% end
% 
% Files=dir(FolderPath);
% FilesCount=length(Files);
% 
% disp('������...');
% 
% 
% parfor k=3:FilesCount
%   
%      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%      Name=Files(k).name;
%      location=strfind(Name,'.');
%      
%      data = imread(FilePath);
%      data(data<-100)=-9999;
%      
%      [m,n] = size(data);
%     
%      SaveFiles=strcat(Name(1:8),'.txt'); %CPC����ļ���·��
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%        
%      fid=fopen(outfile,'w');
%      fprintf(fid,'NCOLS        700\r\nNROWS        400\r\nXLLCORNER   70\r\nYLLCORNER    15\r\nCELLSIZE    0.100\r\nNODATA_VALUE   -9999.0000\r\n');
%      
%      for i=1:1:m  
%          for j=1:1:n
%              if j==n
%                  fprintf(fid,'%g\r\n',data(i,j));
%              else
%                  fprintf(fid,'%g ',data(i,j));
%              end
%          end   
%      end
%      fclose(fid);         
% end
% 
% disp('�������');


