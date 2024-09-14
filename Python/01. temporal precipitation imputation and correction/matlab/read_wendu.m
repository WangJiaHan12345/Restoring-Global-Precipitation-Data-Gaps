% %从tif图片中提取中数据，成为txt格式
% FolderPath=input('请输入数据存储文件夹:','s');  %输入tif文件的数据
% index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
% 
% SaveFolder=strcat('H:\中国区域数据\温度\','整理后数据'); %输出文件夹路径
% if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
%     mkdir(SaveFolder);
% end
% 
% Files=dir(FolderPath);
% FilesCount=length(Files);
% 
% disp('处理中...');
% 
% 
% parfor k=3:FilesCount
%   
%      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%      Name=Files(k).name;
%      location=strfind(Name,'.');
%      
%      data = imread(FilePath);
%      data(data<-100)=-9999;
%      
%      [m,n] = size(data);
%     
%      SaveFiles=strcat(Name(1:8),'.txt'); %CPC输出文件夹路径
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%        
%      fid=fopen(outfile,'w');
%      fprintf(fid,'NCOLS        700\r\nNROWS        400\r\nXLLCORNER   70\r\nYLLCORNER    15\r\nCELLSIZE    0.100\r\nNODATA_VALUE   -9999.0000\r\n');
%      
%      for i=1:1:m  
%          for j=1:1:n
%              if j==n
%                  fprintf(fid,'%g\r\n',data(i,j));
%              else
%                  fprintf(fid,'%g ',data(i,j));
%              end
%          end   
%      end
%      fclose(fid);         
% end
% 
% disp('处理完成');


