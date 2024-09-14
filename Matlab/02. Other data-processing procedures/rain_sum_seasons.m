FolderPath=input('请输入数据存储文件夹:','s'); %输入
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('G:\结果\ganhan\global\','1-12'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

day=1461;
grid=4147;
rain=zeros(1,1);

for k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
  
      fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,day);
      data = data';
      fclose(fid);
       
       for i=1:1:day
           for j=1:1:1
                  rain(1,1)=rain(1,1)+data(i,j);
           end
       end
end
       
SaveFiles=strcat('Final_rain_sum','.txt'); %CPC输出文件夹路径
outfile=strcat(SaveFolder,'\',SaveFiles);

if exist(outfile,'file')~=0 
delete(outfile);     
end
fid1=fopen(outfile,'w');

     
 for i=1:1:1
     for j=1:1:1                
             fprintf(fid1,'%g\r\n',rain(i,j)/grid);
     end
 end  
fclose(fid1);


      
disp('处理完成')


 