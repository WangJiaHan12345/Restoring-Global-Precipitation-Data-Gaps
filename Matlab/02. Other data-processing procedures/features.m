%features
FolderPath=input('请输入数据存储文件夹:','s'); %输入可用网格的标准
FolderPath1=input('请输入数据存储文件夹:','s'); %输入所需整理的数据 
SaveFolder=strcat('H:\时间预测\结果\shirun\02_final_data\季节\日\6-8\','ceshi_feature\wendu'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);

Files1=dir(FolderPath1);
FilesCount1=length(Files1);
disp('处理中...');

parfor k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
  
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      data1=zeros(FilesCount1-2,1); %%%%12-2=272  3-5=276  6-8=276 9-11=273
      
      for m=3:FilesCount1  
          FilePath1=strcat(FolderPath1,'\',Files1(m).name);  %文件路径\文件名
          fid=fopen(FilePath1,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
          data = cell2mat(textscan(fid,'%f','headerlines',6));
          data = reshape(data,720,240);
          data = data'; 

          data1(m-2,1) = data(i,j);
          fclose(fid); 
      end

     SaveFiles=strcat(Name(1:location(end)-1),'.txt');

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');


     for i=1:1:FilesCount1-2
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data1(i,j));
             else
                fprintf(fid1,'%g ',data1(i,j));
             end
         end   
     end
     fclose(fid1); 
    
end
disp('处理完成');
