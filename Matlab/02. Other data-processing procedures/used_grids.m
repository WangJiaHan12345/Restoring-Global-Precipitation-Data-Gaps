%���¶�����ת��Ϊ��������
FolderPath=input('���������ݴ洢�ļ���:','s'); %���� �¶������ļ�
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('H:\ʱ��Ԥ��\�ĸ������������\shirun\02_grid_data\xunlian_features\','wendu'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

 %�߳�
fid_1 = fopen('H:\ʱ��Ԥ��\������\shirun\dem\global\DEM.txt');
data1 = cell2mat(textscan(fid_1,'%f','headerlines',6));
data1 = reshape(data1,720,229);
data1 = data1';
fclose(fid_1);  

%�¶� ����
fid_2 = fopen('H:\ʱ��Ԥ��\�������\�¶�����\slope.txt');
data2 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data2 = reshape(data2,720,240);
data2 = data2';
fclose(fid_2); 
 
% %�¶�
% fid_3 = fopen('H:\ʱ��Ԥ��\�������\�¶�\xunlian\slope.txt');
% data3 = cell2mat(textscan(fid_3,'%f','headerlines',6));
% data3 = reshape(data3,720,240);
% data3 = data3';
% fclose(fid_3); 

% xunlain
for k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      if data1(i,j)~=-9999 && data2(i,j)~=-9999
          fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
          data = cell2mat(textscan(fid,'%f','headerlines',0));
          data = reshape(data,1,1096);
          data = data'; 

          data3=zeros(271,1); %%%%%%%%%%%%%%%%%%%%

           a=0;
           for i=1:1:59
               for j=1:1:1
                   a=a+1;
                   data3(a,1)=data(i,j);
               end
           end

            for i=335:1:425
               for j=1:1:1
                   a=a+1;
                   data3(a,1)=data(i,j);
               end
            end

             for i=701:1:790
               for j=1:1:1
                   a=a+1;
                   data3(a,1)=data(i,j);
               end
             end
             
              for i=1066:1:1096
               for j=1:1:1
                   a=a+1;
                   data3(a,1)=data(i,j);
               end
             end

             SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
             SaveFiles=strcat(Name(1:location(end)-1),'.txt');

             outfile=strcat(SaveFolder,'\',SaveFiles);

             if exist(outfile,'file')~=0 
                delete(outfile);     
             end
             fid1=fopen(outfile,'w');


             for i=1:1:273
                 for j=1:1:1
                     if j==1
                         fprintf(fid1,'%g\r\n',data3(i,j));
                     else
                        fprintf(fid1,'%g ',data3(i,j));
                     end
                 end   
             end
             fclose(fid1); 
             fclose(fid); 
        
         
      end
end

% %ceshi
% for k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%       
%       
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
%       if data1(i,j)~=-9999 && data2(i,j)~=-9999
%           fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%           data = cell2mat(textscan(fid,'%f','headerlines',0));
%           data = reshape(data,1,365);
%           data = data';
%       
%           data3=zeros(90,1); %%%%%%%%%%%%%%%%%%%%
% 
%            a=0;
%            for i=1:1:59
%                for j=1:1:1
%                    a=a+1;
%                    data3(a,1)=data(i,j);
%                end
%            end
%            
%             for i=335:1:365
%                for j=1:1:1
%                    a=a+1;
%                    data3(a,1)=data(i,j);
%                end
%            end
% 
%          SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
%          %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
% 
%          outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%          if exist(outfile,'file')~=0 
%             delete(outfile);     
%          end
%          fid1=fopen(outfile,'w');
% 
% 
%          for i=1:1:91
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data3(i,j));
%                  else
%                     fprintf(fid1,'%g ',data3(i,j));
%                  end
%              end   
%          end
%          fclose(fid1); 
%          fclose(fid); 
%       end
%          
% end
% disp('�������')

