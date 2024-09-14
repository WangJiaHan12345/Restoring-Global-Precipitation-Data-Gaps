% FolderPath=input('请输入数据存储文件夹:','s'); %输入
% index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
% SaveFolder=strcat('H:\时间预测\不成熟\23区\02_grid_data\cpc\季节\月\','6-8'); %输出文件夹路径
% if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
%     mkdir(SaveFolder);
% end
% Files=dir(FolderPath);
% FilesCount=length(Files);
% disp('处理中...');
% 
% 
% fid_2 = fopen('H:\时间预测\最初数据\dem\global_DEM.txt');
% data1 = cell2mat(textscan(fid_2,'%f','headerlines',6));
% data1 = reshape(data1,720,240);
% data1 = data1';
% fclose(fid_2);  
% 
% 
% count=2;
% step=[31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,29,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,29,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31,31,28,31,30,31,30,31,31,30,31,30,31];
% for i=1:120
%     count=count+step(i);
%     data2=zeros(240,720); 
%     for k=3:count
%           FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%           Name=Files(k).name;
%           location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
% 
%           fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%           data = cell2mat(textscan(fid,'%f','headerlines',6));
%           data = reshape(data,720,240);
%           data = data';
% 
% 
%            for i=1:1:240
%                for j=1:1:720
%                    if data1(i,j)~=-9999 
%                       data2(i,j)=data2(i,j)+data(i,j);
%                    else
%                        data2(i,j)=-9999;
%                    end
%                end
%            end
%          fclose(fid);     
%     end
% 
%      SaveFiles=strcat(Name(location(end)-8:location(end)-3),'.txt'); %CPC输出文件夹路径
% 
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%      fid1=fopen(outfile,'w');
% 
%      fprintf(fid1,'NCOLS        720\r\nNROWS        240\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.5\r\nNODATA_VALUE    -9999\r\n');
% 
%     
% 
%      for i=1:1:240
%          for j=1:1:720
%              if j==720
%                  fprintf(fid1,'%g\r\n',data2(i,j));
%              else
%                 fprintf(fid1,'%g ',data2(i,j));
%              end
%          end   
%      end
%      fclose(fid1); 
% end
% disp('处理完成')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FolderPath=input('请输入数据存储文件夹:','s'); %输入
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('H:\时间预测\不成熟\shirun\02_final_data\季节\日\','6-8\ceshi_feature\adem'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...'); 


% step=[31,61,92,123,153,184,215,245,276];
step=[31,61,92];
 
for k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置

      fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,92);
      data = data';
      fclose(fid);
      
      data1=zeros(3,1);
      %特征
      
      for a=1:1:3      
           data1(a,1)=data(step(a),1);
      end

      %降水
%       count=1;
%       for a=1:1:9
%           for i=count:1:step(a);
%               data1(a,1)=data1(a,1)+data(i,1);
%           end
%           count=1+step(a);
%       end
    

     SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');



     for i=1:1:3
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data1(i,j));
             else
                 fprintf(fid1,'%g ',data1(i,j));
             end
         end   
     end
     fclose(fid1); 
end
    disp('处理完成')



