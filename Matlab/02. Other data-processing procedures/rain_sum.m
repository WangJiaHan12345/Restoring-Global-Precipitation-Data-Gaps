FolderPath=input('���������ݴ洢�ļ���:','s'); %����
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('H:\ʱ��Ԥ��\������\23��\02_grid_data\Final\sum_rain\','xunlian'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');


% data2=zeros(562,1); 
% for k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%   
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,562);
%       data = data';
%       
% 
%       
%        
%        for i=1:1:562
%            for j=1:1:1
%                   data2(i,j)=data2(i,j)+data(i,j);
%            end
%        end
%       
%      fclose(fid); 
% 
% end
% 
%   outfile=strcat(SaveFolder,'\','rain_sum.txt');
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%      fid1=fopen(outfile,'w');
%      
%      
%  for i=1:1:562
%      for j=1:1:1
%          if j==1
%              fprintf(fid1,'%g\r\n',data2(i,j));
%          else
%             fprintf(fid1,'%g ',data2(i,j));
%          end
%      end   
%  end
% fclose(fid1); 
% disp('�������')
% 
% 

for k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,1096);
      data = data';
      
      data2=zeros(1,1); 
      
       
       for i=1:1:1096
           for j=1:1:1
                  data2(1,1)=data2(1,1)+data(i,j);
           end
       end
       
      SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
      outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');
     
     
         for i=1:1:1
             for j=1:1:1                
                     fprintf(fid1,'%g\r\n',data2(i,j));
             end
         end  
          fclose(fid1);
           fclose(fid);
end
       

disp('�������')


 