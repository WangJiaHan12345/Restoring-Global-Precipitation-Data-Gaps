FolderPath=input('请输入数据存储文件夹:','s'); %输入
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('G:\结果\global\季节\','s'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

sum=zeros(1461,1);


for k=21463:FilesCount
          FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
          Name=Files(k).name;
          location=strfind(Name,'.');  %输出字符'.'在FilePath的位置

          fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
          data = cell2mat(textscan(fid,'%f','headerlines',0));
          data = reshape(data,1,1461);
          data = data';
          fclose(fid); 
          
          for i=1:1:1461
              sum(i,1)=sum(i,1)+ data(i,1);
          end
end
    
SaveFiles=strcat('cpc','.txt'); %CPC输出文件夹路径
outfile=strcat(SaveFolder,'\',SaveFiles);

if exist(outfile,'file')~=0 
delete(outfile);     
end
fid1=fopen(outfile,'w');

days=8181;

for i=1:1:1461
 for j=1:1:1
     if j==1
         fprintf(fid1,'%g\r\n',sum(i,j)/days);
     else
         fprintf(fid1,'%g ',sum(i,j)/days);
     end
 end 
end
 fclose(fid1); 
 disp('处理结束');

% 
% FolderPath=input('请输入数据存储文件夹:','s'); %输入
% index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
% SaveFolder=strcat('G:\结果\shirun\global\1-12\','区域尺度'); %输出文件夹路径
% if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
%     mkdir(SaveFolder);
% end
% Files=dir(FolderPath);
% FilesCount=length(Files);
% disp('处理中...');
% 
% 
% sum=zeros(1461,7140);
% 
% for k=3:FilesCount
%           FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%           Name=Files(k).name;
%           location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
% 
%           fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%           data = cell2mat(textscan(fid,'%f','headerlines',0));
%           data = reshape(data,1,1461);
%           data = data';
%           fclose(fid); 
%           
%           
%           for i=1:1:1461
%                 sum(i,k-2)=data(i,1);
%           end
% end
%     
% SaveFiles=strcat('NN','.txt'); %CPC输出文件夹路径
% outfile=strcat(SaveFolder,'\',SaveFiles);
% 
% if exist(outfile,'file')~=0 
% delete(outfile);     
% end
% fid1=fopen(outfile,'w');
% 
% 
% for i=1:1:1461
%  for j=1:1:7140
%      if j==7140
%          fprintf(fid1,'%g\r\n',sum(i,j));
%      else
%          fprintf(fid1,'%g ',sum(i,j));
%      end
%  end 
% end
%  fclose(fid1); 
%  disp('处理结束');
% 
%   