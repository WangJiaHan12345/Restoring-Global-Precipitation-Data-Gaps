%ֻѡȡ�е���վ������������֤
%������Ҫ�޳�������ļ���
FolderPath1=input('���������ݴ洢�ļ���:','s');  % ANN cpc Early Final
index1=strfind(FolderPath1,'\');  %����ַ�'\'��FolderPath��λ��
Files=dir(FolderPath1);
FilesCount=length(Files);

FolderPath2=input('���������ݴ洢�ļ���:','s'); 
index2=strfind(FolderPath2,'\');  %����ַ�'\'��FolderPath��λ��


FolderPath3=input('���������ݴ洢�ļ���:','s'); 
index3=strfind(FolderPath3,'\');  %����ַ�'\'��FolderPath��λ��


FolderPath4=input('���������ݴ洢�ļ���:','s'); 
index4=strfind(FolderPath4,'\');  %����ַ�'\'��FolderPath��λ��

FolderPath5=input('���������ݴ洢�ļ���:','s'); 
index5=strfind(FolderPath5,'\');  %����ַ�'\'��FolderPath��λ��


fid = fopen('G:\ȫ��\ʱ��Ԥ����\վ����Ϣ.txt','rb','l');
data = cell2mat(textscan(fid,'%f','headerlines',6));
data = reshape(data,720,240);
data = data'; 
fclose(fid); 

disp('������...');

%��һ���ļ�����ɾ��
% for k=3:FilesCount
%      FilePath1=strcat(FolderPath1,'\',Files(k).name);  %�ļ�·��\�ļ���
%      Name=Files(k).name;
%      location=strfind(Name,'.');
%      
%      FilePath2=strcat(FolderPath2,'\',Files(k).name);
%      FilePath3=strcat(FolderPath3,'\',Files(k).name);
%      FilePath4=strcat(FolderPath4,'\',Files(k).name);
%       
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
%       
%       if data(i,j)<=0          
%           delete(FilePath1);
%           delete(FilePath2);
%           delete(FilePath3);
%           delete(FilePath4);
%       end 
%      
% end


%��һ���ļ��и��Ƶ���һ���ļ�����
for k=3:FilesCount
     FilePath1=strcat(FolderPath1,'\',Files(k).name);  %�ļ�·��\�ļ���
     Name=Files(k).name;
     location=strfind(Name,'.');
     
     FilePath2=strcat(FolderPath2,'\',Files(k).name);
     FilePath3=strcat(FolderPath3,'\',Files(k).name);
     FilePath4=strcat(FolderPath4,'\',Files(k).name);
     FilePath5=strcat(FolderPath5,'\',Files(k).name);
     
     toPath1='H:\ʱ��Ԥ��\�ĸ������������\ganhan\ֻѡ���е���վ��\02_grid_data\xunlian\cpc\';
     if exist(toPath1,'dir')~=7  %���·�����������½�·��
       mkdir(toPath1);
     end
     toPath2='H:\ʱ��Ԥ��\�ĸ������������\ganhan\ֻѡ���е���վ��\02_grid_data\xunlian\Early\';
     if exist(toPath2,'dir')~=7  %���·�����������½�·��
       mkdir(toPath2);
     end
     toPath3='H:\ʱ��Ԥ��\�ĸ������������\ganhan\ֻѡ���е���վ��\02_grid_data\xunlian\Final\';
     if exist(toPath3,'dir')~=7  %���·�����������½�·��
       mkdir(toPath3);
     end
     toPath4='H:\ʱ��Ԥ��\�ĸ������������\ganhan\ֻѡ���е���վ��\02_grid_data\xunlian_features\wendu\';
     if exist(toPath4,'dir')~=7  %���·�����������½�·��
       mkdir(toPath4);
     end
     toPath5='H:\ʱ��Ԥ��\�ĸ������������\ganhan\ֻѡ���е���վ��\02_grid_data\xunlian_features\lat_wei\';
     if exist(toPath5,'dir')~=7  %���·�����������½�·��
       mkdir(toPath5);
     end
      
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      if data(i,j)>0          
          copyfile(FilePath1,toPath1);
          copyfile(FilePath2,toPath2);
          copyfile(FilePath3,toPath3);
          copyfile(FilePath4,toPath4);
          copyfile(FilePath5,toPath5);
      end 
     
end
disp('�������');