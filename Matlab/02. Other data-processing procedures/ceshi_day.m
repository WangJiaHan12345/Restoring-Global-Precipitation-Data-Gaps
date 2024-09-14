% %测试结果转为每日各个点结果
% SaveFolder=strcat('H:\时间预测\不成熟\24区\result\','predict'); %输出文件夹路径
% if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
%     mkdir(SaveFolder);
% end
% 
% disp('处理中...');
% 
% fid_2 = fopen('H:\时间预测\不成熟\24区\result\predict\out_data.txt');
% data1 = cell2mat(textscan(fid_2,'%f','headerlines',0));
% data1 = reshape(data1,1,85775);
% data1 = data1';
% fclose(fid_2);  
% 
% grid_count=235;
% a=0;
% for k=1:1:365 %2018年的天数
%    data2=zeros(grid_count,1); 
%   
%     for i=1:1:grid_count
%        for j=1:1:1
%             a=a+1;
%             data2(i,j)=data1(a,j); 
%        end
%     end
% 
%     SaveFiles=strcat(num2str(k),'.txt'); %CPC输出文件夹路径
% 
% 
%     outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%     if exist(outfile,'file')~=0 
%     delete(outfile);     
%     end
%     fid1=fopen(outfile,'w');
% 
% 
%     for i=1:1:grid_count
%      for j=1:1:1
%          if j==1
%              fprintf(fid1,'%g\r\n',data2(i,j));
%          else
%             fprintf(fid1,'%g ',data2(i,j));
%          end
%      end   
%     end
%     fclose(fid1); 
%  
% 
% end
% disp('处理完成')
% 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%day转为grid
FolderPath=input('请输入数据存储文件夹:','s'); %day
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('H:\时间预测\不成熟\23区\02_grid_data\','Final\ceshi'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

day_count=365;  %1096  365
grid_count=562;

fid_2 = fopen('H:\时间预测\不成熟\23区\dem\DEM.txt');  %找到格点的位置
data1 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data1 = reshape(data1,720,229);
data1 = data1';
fclose(fid_2);  

grid=zeros(grid_count,2);

a=0;
for i=1:1:229
    for j=1:1:720
        if data1(i,j)~=-9999
            a=a+1;
            grid(a,1)=i;
            grid(a,2)=j;
        end
    end
end

            
           
for i=1:1:grid_count
    data2=zeros(day_count,1); 
    a=0;
    for k=3:FilesCount
          a=a+1;
          FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
          Name=Files(k).name;
          location=strfind(Name,'.');  %输出字符'.'在FilePath的位置

          fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
          data = cell2mat(textscan(fid,'%f','headerlines',0));
          data = reshape(data,1,grid_count);
          data = data';
          fclose(fid); 
          
          data2(a,1)=data(i,1);     
    end
    
    
    SaveFiles=strcat(num2str(grid(i,1),'%03d'),num2str(grid(i,2),'%03d')); %CPC输出文件夹路径
    SaveFiles=strcat(SaveFiles,'.txt'); %CPC输出文件夹路径
    outfile=strcat(SaveFolder,'\',SaveFiles);

    if exist(outfile,'file')~=0 
    delete(outfile);     
    end
    fid1=fopen(outfile,'w');


     for i=1:1:day_count
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


 
