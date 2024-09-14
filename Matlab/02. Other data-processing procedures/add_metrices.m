%添加地形辅助因子
% FolderPath=input('请输入数据存储文件夹:','s');  %输入可用网格
% index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
% Files=dir(FolderPath);
% FilesCount=length(Files);
% 
% SaveFolder=strcat('H:\青藏高原数据\时间预测\2015-2016\01_clip_data\测试\','高程\12-2'); %输出文件夹路径
% if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
%     mkdir(SaveFolder);
% end
% 
% fid = fopen('H:\青藏高原数据\时间预测\2015-2016\dem\dem.txt','rb','l');  %dem  Slope
% data = cell2mat(textscan(fid,'%f','headerlines',6));
% data = reshape(data,700,440);
% data = data'; 
% fclose(fid); 
% 
% 
% for k=3:FilesCount
%     
%       result =[];
%        
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%       
%       
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
%       
%       % 全球
%       % 训练：3-5:276  6-8:276  9-11:273  12-2:271  
%       % 测试：3-5:92  6-8:92  9-11:91  12-2:90
%       
%       %青藏高原
%       % 训练: 3-5: 153  6-8: 153   9-11:152  12-2: 150
%       % 测试：3-5：31  6-8：31  9-11：30  12-2：31
%       
%       day =31; 
%       
%       for m = 1:1:day
%           result(m,1) = data(i,j);
%       end
%       
%      SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC输出文件夹路
%       
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
% 
%      fid1=fopen(outfile,'w');
% 
%      for i=1:1:day
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
% disp('处理完成');
% 
%   

