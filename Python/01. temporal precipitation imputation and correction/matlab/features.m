%特征值
SaveFolder=strcat('H:\时间预测\四个区域可用数据\ganhan\02_grid_data\xunlian_features\','slope'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
disp('处理中...');

%高程
fid_1 = fopen('H:\时间预测\四个区域可用数据\ganhan\dem\DEM.txt');
data1 = cell2mat(textscan(fid_1,'%f','headerlines',6));
data1 = reshape(data1,720,240);   %不同区域的dem的行数 要改
data1 = data1';
fclose(fid_1); 
% % 
% fid_4 = fopen('H:\时间预测\四个区域可用数据\banshirun\dem\MRDEM.txt');
% data4 = cell2mat(textscan(fid_4,'%f','headerlines',6));
% data4 = reshape(data4,720,232);  %不同区域的dem的行数
% data4 = data4';
% fclose(fid_4);
%坡度 坡向
fid_2 = fopen('H:\时间预测\最初数据\坡度坡向\slope.txt');
data2 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data2 = reshape(data2,720,240); %这里就不用换了 坡度和坡向任选一个即可
data2 = data2';
fclose(fid_2);  
%这里放置所需的各种特征的文件
fid_3 = fopen('H:\时间预测\最初数据\坡度坡向\slope.txt');  
data3 = cell2mat(textscan(fid_3,'%f','headerlines',6));
data3 = reshape(data3,720,240);  %这里就不用换了
data3 = data3';
fclose(fid_3);  
  
      
count= 1096; %代表天数 训练1096或者测试365    
data=zeros(count,1); 
for i=1:1:240   %这里的行数也是根据dem来进行判断的  要改
   for j=1:1:720
       if data1(i,j)~=-9999 && data2(i,j)~=-9999
          for a=1:1:count
              for b=1:1:1
                 data(a,b)=data3(i,j);
              end
          end
      

          SaveFiles=strcat(num2str(i,'%03d'),num2str(j,'%03d')); %CPC输出文件夹路径
          SaveFiles=strcat(SaveFiles,'.txt');

         outfile=strcat(SaveFolder,'\',SaveFiles);

         if exist(outfile,'file')~=0 
            delete(outfile);     
         end
         fid1=fopen(outfile,'w');

         for c=1:1:count
             for d=1:1:1
                 if d==1
                     fprintf(fid1,'%g\r\n',data(c,d));
                 else
                     fprintf(fid1,'%g ',data(c,d));
                 end
             end   
         end
         fclose(fid1); 
       end
     
   end
end

disp('处理完成')


