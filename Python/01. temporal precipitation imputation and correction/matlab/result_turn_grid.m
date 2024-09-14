%将预测结果转换为网格形式
%输入每块对应的网格
% 10  10   10  5 
FolderPath=input('请输入数据存储文件夹:','s');  %输入得到的每一块的可用网格
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
Files=dir(FolderPath);
FilesCount=length(Files);


SaveFolder1=strcat('G:\全球\时间预测结果\地面站点_Early\','ganhan\北\12-2\early'); %输出文件夹路径
if exist(SaveFolder1,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder1);
end

SaveFolder2=strcat('G:\全球\时间预测结果\地面站点_Early\','ganhan\北\12-2\ANN'); %输出文件夹路径
if exist(SaveFolder2,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder2);
end


SaveFolder3=strcat('G:\全球\时间预测结果\地面站点_Early\','ganhan\北\12-2\final'); %输出文件夹路径
if exist(SaveFolder3,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder3);
end
% 
SaveFolder4=strcat('G:\全球\时间预测结果\地面站点_Early\','ganhan\北\12-2\cpc'); %输出文件夹路径
if exist(SaveFolder4,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder4);
end
% 
% % SaveFolder5=strcat('G:\青藏高原\时间\季度\新想法（ANN+gauge)\','12-2\mvk'); %输出文件夹路径
% % if exist(SaveFolder5,'dir')~=7  %如果路径不存在则新建路径
% %     mkdir(SaveFolder5);
% % end
% 

% SaveFolder6=strcat('G:\青藏高原\时间\季度\新想法（ANN+gauge)\','12-2\国家气象局'); %输出文件夹路径
% if exist(SaveFolder6,'dir')~=7  %如果路径不存在则新建路径
%     mkdir(SaveFolder6);
% end


% 这里输入预测的结果
% 全球
count =85*(FilesCount-2);  %这里也要改 90 90 90 85

%青藏高原
% count = 30*(FilesCount-2); %这里也要改 30 30 30 30

fid1 = fopen('H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\result(early)\北\12-2\early\D_data.txt','rb','l');
data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
data1 = reshape(data1,1,count);
data1 = data1'; 
fclose(fid1); 

fid2 = fopen('H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\result(early)\北\12-2\ANN\D_data.txt','rb','l');
data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
data2 = reshape(data2,1,count);
data2 = data2';
fclose(fid2); 

fid3 = fopen('H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\result(early)\北\12-2\final\D_data.txt','rb','l');
data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
data3 = reshape(data3,1,count);
data3 = data3';
fclose(fid3); 

fid4 = fopen('H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\result(early)\北\12-2\cpc\D_data.txt','rb','l');
data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
data4 = reshape(data4,1,count);
data4 = data4';
fclose(fid4); 

% fid5 = fopen('H:\青藏高原数据\时间预测\新想法\result(有辅助因子)\12-2\mvk\E_data.txt','rb','l');
% data5 = cell2mat(textscan(fid5,'%f','headerlines',0));
% data5 = reshape(data5,1,count);
% data5 = data5';
% fclose(fid5); 

% fid6 = fopen('H:\青藏高原数据\时间预测\新想法\result(有辅助因子)\12-2\国家气象局\E_data.txt','rb','l');
% data6 = cell2mat(textscan(fid6,'%f','headerlines',0));
% data6 = reshape(data6,1,count);
% data6 = data6';
% fclose(fid6); 

disp('处理中...');


for k=3:FilesCount
     FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
     Name=Files(k).name;
     location=strfind(Name,'.');
      
     
     day =85; % 全球：3-5 6-8 9-11:90  12-2:85   青藏： 30

     rain1 = data1(day*(k-3)+1:day*(k-2),1);
     rain2 = data2(day*(k-3)+1:day*(k-2),1);
     rain3 = data3(day*(k-3)+1:day*(k-2),1);
     rain4 = data4(day*(k-3)+1:day*(k-2),1);
%      rain5 = data5(day*(k-3)+1:day*(k-2),1);
%      rain6 = data6(day*(k-3)+1:day*(k-2),1);
     
     
     SaveFiles=strcat(Name(1:6),'.txt'); %CPC输出文件夹路径
     
     outfile1=strcat(SaveFolder1,'\',SaveFiles);
     outfile2=strcat(SaveFolder2,'\',SaveFiles);
     outfile3=strcat(SaveFolder3,'\',SaveFiles);
     outfile4=strcat(SaveFolder4,'\',SaveFiles);
%      outfile5=strcat(SaveFolder5,'\',SaveFiles);
%      outfile6=strcat(SaveFolder6,'\',SaveFiles);


     
     fid1=fopen(outfile1,'w');
     for i=1:1:day  
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',rain1(i,j));
             else
                 fprintf(fid1,'%g ',rain1(i,j));
             end
         end   
     end
     fclose(fid1);       
     
     fid2=fopen(outfile2,'w');
     for i=1:1:day  
         for j=1:1:1
             if j==1
                 fprintf(fid2,'%g\r\n',rain2(i,j));
             else
                 fprintf(fid2,'%g ',rain2(i,j));
             end
         end   
     end
     fclose(fid2);  
     
     fid3=fopen(outfile3,'w');
     for i=1:1:day 
         for j=1:1:1
             if j==1
                 fprintf(fid3,'%g\r\n',rain3(i,j));
             else
                 fprintf(fid3,'%g ',rain3(i,j));
             end
         end   
     end
     fclose(fid3);  
     
     
     fid4=fopen(outfile4,'w');
     for i=1:1:day % 
         for j=1:1:1
             if j==1
                 fprintf(fid4,'%g\r\n',rain4(i,j));
             else
                 fprintf(fid4,'%g ',rain4(i,j));
             end
         end   
     end
     fclose(fid4);  
     
%      fid5=fopen(outfile5,'w');
%      for i=1:1:day % 
%          for j=1:1:1
%              if j==1
%                  fprintf(fid5,'%g\r\n',rain5(i,j));
%              else
%                  fprintf(fid5,'%g ',rain5(i,j));
%              end
%          end   
%      end
%      fclose(fid5); 
%      
%      fid6=fopen(outfile6,'w');
%      for i=1:1:day % 
%          for j=1:1:1
%              if j==1
%                  fprintf(fid6,'%g\r\n',rain5(i,j));
%              else
%                  fprintf(fid6,'%g ',rain5(i,j));
%              end
%          end   
%      end
%      fclose(fid6); 
   
end
disp('处理完成');