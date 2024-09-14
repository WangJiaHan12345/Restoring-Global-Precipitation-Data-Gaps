%将数据转换为网格数据
%输入 gsmap数据文件  H:\GSMAP数据\读取后的数据-0.5\gsmap_mvk
FolderPath1=input('请输入数据存储文件夹:','s'); 
index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('H:\青藏高原数据\GSMaP\','gsmap_rnt_year'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

% 输入可以表示可用网格的数据，这里可以用dem
% H:\时间预测\四个区域可用数据\shirun\02_grid_data\xunlian_features\dem
% H:\空间预测\shirun\02_final_data\cpc4\3-5
FolderPath2=input('请输入数据存储文件夹:','s'); 
index2=strfind(FolderPath2,'\');  %输出字符'\'在FolderPath的位置                                  
Files2=dir(FolderPath2);
FilesCount2=length(Files2);


% fid = fopen('G:\青藏高原\中国-青藏高原-440.txt','rb','l');
% data = cell2mat(textscan(fid,'%f','headerlines',6));
% data = reshape(data,700,440);
% data = data'; 
% fclose(fid); 
% disp('处理中...');

%可用网格是文件夹
for k=11885:FilesCount2
      Name=Files2(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
      
      
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      
      result = zeros(731,1);  % 365  1096  1461
      
      parfor m =3:FilesCount1
          FilePath = strcat(FolderPath1,'\',Files1(m).name);
          fid = fopen(FilePath,'rb','l');
          data = cell2mat(textscan(fid,'%f','headerlines',6));
          data = reshape(data,700,440);   %700 440   或者 700 400
          data = data'; 
          fclose(fid); 
          
          
          % 700 440
          result(m-2,1)=data(i,j); 
          % 700 400
%           result(m-2,1)=data(i-40,j); 
          
      end
      
     SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     
     
     fid1=fopen(outfile,'w');
     for i=1:1:731  % 365  1096  1461  731
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


% 可用网格是txt文件
% for i = 271:1:271
%     for j =89:1:700
%         if data(i,j) >= 0
%             rain = zeros(731,1);
%             parfor k = 3:FilesCount1
%               FilePath = strcat(FolderPath1,'\',Files1(k).name);
%               Name=Files1(k).name;
%               location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%               
%               fid = fopen(FilePath,'rb','l');
%               data1 = cell2mat(textscan(fid,'%f','headerlines',6));
%               data1 = reshape(data1,700,400);
%               data1 = data1';
%               fclose(fid); 
%               
%               rain(k-2,1) = data1(i-40,j);
%             end
%             
%             
%             SaveFiles= [num2str(i,'%03d'),num2str(j,'%03d'),'.txt']; %CPC输出文件夹路径
% 
%              outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%              if exist(outfile,'file')~=0 
%                 delete(outfile);     
%              end
%      
%      
%              fid1=fopen(outfile,'w');
%              
%              for m=1:1:731  
%                  for n=1:1:1
%                      if n==1
%                          fprintf(fid1,'%g\r\n',rain(m,n));
%                      else
%                          fprintf(fid1,'%g ',rain(m,n));
%                      end
%                  end   
%              end
%              
%              fclose(fid1);             
%         end  
%     end
% end
% disp('处理完成');



% for k=3:FilesCount2
%        Name=Files2(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%       
%       
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
%       
%       SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt');
%       
%       path = strcat(SaveFolder,'\',SaveFiles);
%       
%       if (exist(path)==0)
%           result = zeros(1461,1);  % 365  1096  1461
%           
%           parfor m =3:FilesCount1
%               FilePath = strcat(FolderPath1,'\',Files1(m).name);
%               fid = fopen(FilePath,'rb','l');
%               data = cell2mat(textscan(fid,'%f','headerlines',6));
%               data = reshape(data,720,240);
%               data = data'; 
%               fclose(fid); 
%               
%               result(m-2,1)=data(i,j); 
%               
%           end
%           
% %          SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
%     
%          outfile=strcat(SaveFolder,'\',SaveFiles);
%     
%          if exist(outfile,'file')~=0 
%             delete(outfile);     
%          end
%          
%          
%          fid1=fopen(outfile,'w');
%          for i=1:1:1461  % 365  1096  1461
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',result(i,j));
%                  else
%                      fprintf(fid1,'%g ',result(i,j));
%                  end
%              end   
%          end
%          fclose(fid1);   
%       end
% end
% 
% disp('处理完成')