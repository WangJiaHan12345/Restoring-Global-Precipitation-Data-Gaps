FolderPath=input('请输入数据存储文件夹:','s'); %输入字符串给FolderPath，不带s则为默认输入数值
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('G:\毕业论文图\青藏高原\空间\月份性能折线图\降水量\rain\','月'); %输出文件夹路径 cpc Early Final ANN
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

% grid = 16210;
% day = 731; 
% %全球空间3-5 6-8:368  9-11:364 12-2:361   1-12:1461
% %全球时间 1-12:361  3-5 6-8:92  9-11:90  12-2:87
% %青藏高原空间 3-5 6-8：184  9-11：182   12-2：181
% rain=zeros(day,1);
% parfor k=3:FilesCount 
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%   
%       fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,day,1);
%       fclose(fid); 
%       
%       rain = rain + data
%      
% end
% 
% rain = rain / grid;
% 
% outfile=strcat(SaveFolder,'\','gsmap_mvk_day_rain.txt');  % ANN gsmap_gauge gsmap_mvk  国家气象局
% 
%  if exist(outfile,'file')~=0 
%     delete(outfile);     
%  end
%  fid1=fopen(outfile,'w');
%      
%  for i=1:1:day
%      for j=1:1:1
%           fprintf(fid1,'%g\r\n',rain(i,j));
%      end   
%  end
%  fclose(fid1);    
% disp('处理完成')


%将上一步得到的日平均降水转换为月平均降水
day = 731;
for k=3:FilesCount 
      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
  
      fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,day,1);
      fclose(fid); 
      
      goal_day = 24;
      rain=zeros(goal_day,1);
      
      step = [0,31,59,90,120,151,181,212,243,273,304,334,365,396,425,456,486,517,547,578,609,639,670,700,731];
      
      for a = 1:1:goal_day
          for i = step(a)+1:1:step(a+1)
             rain(a,1) = rain(a,1) + data(i,1);
          end
      end
      
      SaveFiles=strcat(Name(1:location(end)-1),'.txt');
      outfile=strcat(SaveFolder,'\',SaveFiles);  % ANN gsmap_gauge gsmap_mvk  国家气象局

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');

     for i=1:1:goal_day
         for j=1:1:1
              fprintf(fid1,'%g\r\n',rain(i,j));
         end   
     end
     fclose(fid1);    
      
     
end
disp('处理完成')
