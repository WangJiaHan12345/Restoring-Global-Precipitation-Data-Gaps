% % 将网格的形式改为日期的形式
FolderPath=input('请输入数据存储文件夹:','s');  %输入日期标尺
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
Files=dir(FolderPath);
FilesCount=length(Files);
% 
SaveFolder=strcat('G:\全球\时间预测结果\地面站点_final\global_climate\1-12\日期形式\','Early日期'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end

%只有在第一部分有用
FolderPath1=input('请输入数据存储文件夹:','s');  %输入网格数据  
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

% 将网格数据转为天数
% for k=3:357
%     
% %      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
% %      Name=Files(k).name;
% %      location=strfind(Name,'.');
%       
%         
%      result = zeros(6420,1);  %168代表网格的个数
%       
%      for m = 3:FilesCount1
%          
%           FilePath1 = strcat(FolderPath1,'\',Files1(m).name);
%           fid = fopen(FilePath1,'rb','l');
%           data = cell2mat(textscan(fid,'%f','headerlines',0));
%           data = reshape(data,1,355);   
%           data = data'; 
%           fclose(fid); 
%           
%           
%           result(m-2,1) = data(k-2,1); 
%          
%      end
%       
%      
%      SaveFiles=strcat(num2str(k-2),'.txt'); %CPC输出文件夹路径
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%      
%      
%      fid1=fopen(outfile,'w');
%      
%      for i=1:1:6420
%          for j=1:1:1
%              if j==1
%                  fprintf(fid1,'%g\r\n',result(i,j));
%              else
%                  fprintf(fid1,'%g ',result(i,j));
%              end
%          end   
%      end
%      fclose(fid1);         
%     
% end

%将天数据从天中提取出来
for k=3:FilesCount   %  6420数据的文件
    
     FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
     Name=Files(k).name;
     location=strfind(Name,'.');
      
        
     result = zeros(240,720);  %168代表网格的个数
         
        for a=1:1:240
            for b = 1:1:720
                result(a,b) = -9999;
            end
        end
     
    
      fid = fopen(FilePath,'rb','l');
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,6420);   
      data = data'; 
      fclose(fid); 
          
      for m =3:FilesCount1   % 6420网格形式的文件
          Name1=Files1(m).name;
          location1=strfind(Name1,'.');  %输出字符'.'在FilePath的位置


          i= str2num(Name1(location1(end)-6:location1(end)-4));
          j= str2num(Name1(location1(end)-3:location1(end)-1));
          
          result(i,j)= data(m-2,1);
      end
          
  
     SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC输出文件夹路径

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     
     
     fid1=fopen(outfile,'w');
     fprintf(fid1,'NCOLS        720\r\nNROWS        240\r\nXLLCORNER   0\r\nYLLCORNER   -60\r\nCELLSIZE    0.5\r\nNODATA_VALUE    -9999\r\n');
     
     for i=1:1:240
         for j=1:1:720
             if j==1
                 fprintf(fid1,'%g\r\n',result(i,j));
             else
                 fprintf(fid1,'%g ',result(i,j));
             end
         end   
     end
     fclose(fid1);         
    
end




% % 将天数转为网格
% %将数据转换为网格数据
% % 输入 gsmap数据文件  H:\GSMAP数据\读取后的数据-0.5\gsmap_mvk
% FolderPath1=input('请输入数据存储文件夹:','s'); 
% index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
% SaveFolder=strcat('H:\青藏高原数据\时间预测\2015-2017\result(不分块)\随机部分日期数据\网格形式\','ANN'); %输出文件夹路径
% if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
%     mkdir(SaveFolder);
% end
% Files1=dir(FolderPath1);
% FilesCount1=length(Files1);
% 
% % 用来规定有效网格
% FolderPath2=input('请输入数据存储文件夹:','s');  %输入网格数据  
% Files2=dir(FolderPath2);
% FilesCount2=length(Files2);
% 
% %可用网格是文件夹
% for k=3:FilesCount2
%       Name=Files2(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%       
%       
% %       i= str2num(Name(location(end)-6:location(end)-4));
% %       j= str2num(Name(location(end)-3:location(end)-1));
%       
%       
%       result = zeros(109,1);  % 365  1096  1461
%       
%       for m =3:FilesCount1
%           FilePath1 = strcat(FolderPath1,'\',Files1(m).name);
%           fid = fopen(FilePath1,'rb','l');
%           data = cell2mat(textscan(fid,'%f','headerlines',0));
%           data = reshape(data,1,168);   %700 440   或者 700 400
%           data = data'; 
%           fclose(fid); 
%           
% 
%           result(m-2,1)=data(k-2,1); 
%           
%       end
%       
%      SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%           
%      fid1=fopen(outfile,'w');
%      
%      for i=1:1:109 
%          for j=1:1:1
%              if j==1
%                  fprintf(fid1,'%g\r\n',result(i,j));
%              else
%                  fprintf(fid1,'%g ',result(i,j));
%              end
%          end   
%      end
%      fclose(fid1);         
% end
% 
% 
% 


% %删除含有-999的天数
% FolderPath1=input('请输入数据存储文件夹:','s'); 
% index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
% Files1=dir(FolderPath1);
% FilesCount1=length(Files1);
% 
% 
% for k=3:FilesCount1
%       FilePath1 = strcat(FolderPath1,'\',Files1(k).name);
%       
%       fid = fopen(FilePath1,'rb','l');
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,168);   %700 440   或者 700 400
%       data = data'; 
%       fclose(fid);
%       
%       for m =1:1:168
%           if data(m,1) <0
%               delete(FilePath1);
%           end
%       end
%         
% end
