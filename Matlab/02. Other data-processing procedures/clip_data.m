FolderPath=input('请输入数据存储文件夹:','s'); %输入
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('H:\时间预测\24区\01_clip_data\','Early'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

fid_2 = fopen('H:\时间预测\28个区\24.txt');
data1 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data1 = reshape(data1,720,240);
data1 = data1';
fclose(fid_2);  

parfor k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
  
      fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data = cell2mat(textscan(fid,'%f','headerlines',6));
      data = reshape(data,720,240);
      data = data';
      
      data2=zeros(240,720); %%%%%%%%%%%%%%%%%%%%%%%
      
       
       for i=1:1:240
           for j=1:1:720
               if data1(i,j)~=-999
                  data2(i,j)=data(i,j);
               else
                   data2(i,j)=-9999;
               end
           end
       end
           
     SaveFiles=strcat(Name(location(end)-8:location(end)-1),'.txt'); %CPC输出文件夹路径
     %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
    
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');
     
     fprintf(fid1,'NCOLS        720\r\nNROWS        240\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.5\r\nNODATA_VALUE    -9999\r\n');
    
     %fprintf(fid1,strcat(repmat('%9.4f\t',1,720),'\r\n'),data2');   %  %f 浮点方式标示,\t:tab键 跳格, \r\n:回车换行，为设置输出格式，B = repmat(A,m,n)：将矩阵 A 复制 m×n 块
     
     for i=1:1:240
         for j=1:1:720
             if j==720
                 fprintf(fid1,'%g\r\n',data2(i,j));
             else
                fprintf(fid1,'%g ',data2(i,j));
             end
         end   
     end
     fclose(fid1); 
     fclose(fid); 

end
disp('处理完成')


 