%只选取有地面站点的网格进行验证
%输入需要剔除网格的文件夹
FolderPath1=input('请输入数据存储文件夹:','s');  % ANN cpc Early Final
index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
Files=dir(FolderPath1);
FilesCount=length(Files);

FolderPath2=input('请输入数据存储文件夹:','s'); 
index2=strfind(FolderPath2,'\');  %输出字符'\'在FolderPath的位置


FolderPath3=input('请输入数据存储文件夹:','s'); 
index3=strfind(FolderPath3,'\');  %输出字符'\'在FolderPath的位置


FolderPath4=input('请输入数据存储文件夹:','s'); 
index4=strfind(FolderPath4,'\');  %输出字符'\'在FolderPath的位置

FolderPath5=input('请输入数据存储文件夹:','s'); 
index5=strfind(FolderPath5,'\');  %输出字符'\'在FolderPath的位置


fid = fopen('G:\全球\时间预测结果\站点信息.txt','rb','l');
data = cell2mat(textscan(fid,'%f','headerlines',6));
data = reshape(data,720,240);
data = data'; 
fclose(fid); 

disp('处理中...');

%从一个文件夹中删除
% for k=3:FilesCount
%      FilePath1=strcat(FolderPath1,'\',Files(k).name);  %文件路径\文件名
%      Name=Files(k).name;
%      location=strfind(Name,'.');
%      
%      FilePath2=strcat(FolderPath2,'\',Files(k).name);
%      FilePath3=strcat(FolderPath3,'\',Files(k).name);
%      FilePath4=strcat(FolderPath4,'\',Files(k).name);
%       
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
%       
%       if data(i,j)<=0          
%           delete(FilePath1);
%           delete(FilePath2);
%           delete(FilePath3);
%           delete(FilePath4);
%       end 
%      
% end


%从一个文件夹复制到另一个文件夹中
for k=3:FilesCount
     FilePath1=strcat(FolderPath1,'\',Files(k).name);  %文件路径\文件名
     Name=Files(k).name;
     location=strfind(Name,'.');
     
     FilePath2=strcat(FolderPath2,'\',Files(k).name);
     FilePath3=strcat(FolderPath3,'\',Files(k).name);
     FilePath4=strcat(FolderPath4,'\',Files(k).name);
     FilePath5=strcat(FolderPath5,'\',Files(k).name);
     
     toPath1='H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\02_grid_data\xunlian\cpc\';
     if exist(toPath1,'dir')~=7  %如果路径不存在则新建路径
       mkdir(toPath1);
     end
     toPath2='H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\02_grid_data\xunlian\Early\';
     if exist(toPath2,'dir')~=7  %如果路径不存在则新建路径
       mkdir(toPath2);
     end
     toPath3='H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\02_grid_data\xunlian\Final\';
     if exist(toPath3,'dir')~=7  %如果路径不存在则新建路径
       mkdir(toPath3);
     end
     toPath4='H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\02_grid_data\xunlian_features\wendu\';
     if exist(toPath4,'dir')~=7  %如果路径不存在则新建路径
       mkdir(toPath4);
     end
     toPath5='H:\时间预测\四个区域可用数据\ganhan\只选用有地面站点\02_grid_data\xunlian_features\lat_wei\';
     if exist(toPath5,'dir')~=7  %如果路径不存在则新建路径
       mkdir(toPath5);
     end
      
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      if data(i,j)>0          
          copyfile(FilePath1,toPath1);
          copyfile(FilePath2,toPath2);
          copyfile(FilePath3,toPath3);
          copyfile(FilePath4,toPath4);
          copyfile(FilePath5,toPath5);
      end 
     
end
disp('处理完成');