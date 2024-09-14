%将温度数据转换为网格数据
%输入 温度数据文件
FolderPath1=input('请输入数据存储文件夹:','s'); 
index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('H:\时间预测\四个区域可用数据\ganhan\02_grid_data\ceshi_features\','wendu'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

% 输入可以表示可用网格的数据，这里可以用dem
% H:\时间预测\四个区域可用数据\shirun\02_grid_data\xunlian_features\dem
FolderPath2=input('请输入数据存储文件夹:','s'); 
index2=strfind(FolderPath2,'\');  %输出字符'\'在FolderPath的位置                                  
Files2=dir(FolderPath2);
FilesCount2=length(Files2);

disp('处理中...');


for k=3:FilesCount2
      Name=Files2(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
      
      
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      
      result = zeros(365,1);  % 365  1096
      
      parfor m =3:FilesCount1
          FilePath = strcat(FolderPath1,'\',Files1(m).name);
          fid = fopen(FilePath,'rb','l');
          data = cell2mat(textscan(fid,'%f','headerlines',6));
          data = reshape(data,720,240);
          data = data'; 
          
          result(m-2,1)=data(i,j); 
          
          fclose(fid); 
          
      end
      
     SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     
     
     fid1=fopen(outfile,'w');
     for i=1:1:365  % 365
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',result(i,j));
             else
                fprintf(fid1,'%g ',result(i,j));
             end
         end   
     end
     fclose(fid1);         
end

disp('处理完成');