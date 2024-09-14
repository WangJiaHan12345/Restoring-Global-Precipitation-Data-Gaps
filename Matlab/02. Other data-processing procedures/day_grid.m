FolderPath=input('请输入数据存储文件夹:','s'); %输入
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('H:\时间预测\不成熟\24区\02_grid_data\Early\','xunlian'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

grid_count=562;
for i=1:1:562
    data2=zeros(365,1); 
    a=0;
    for k=3:FilesCount
          a=a+1;
          FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
          Name=Files(k).name;
          location=strfind(Name,'.');  %输出字符'.'在FilePath的位置

          fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
          data = cell2mat(textscan(fid,'%f','headerlines',0));
          data = reshape(data,1,562);
          data = data';
          fclose(fid); 
          
          data2(a,1)=data(i,1);     
    end
    
    SaveFiles=strcat(num2str(i),'.txt'); %CPC输出文件夹路径
    outfile=strcat(SaveFolder,'\',SaveFiles);

    if exist(outfile,'file')~=0 
    delete(outfile);     
    end
    fid1=fopen(outfile,'w');


     for i=1:1:365
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data2(i,j));
             else
                 fprintf(fid1,'%g ',data2(i,j));
             end
         end 
     end
     fclose(fid1); 

end
  
disp('处理完成')


 