%将整年的数据分季节
FolderPath=input('请输入数据存储文件夹:','s'); %输入
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder=strcat('H:\时间预测\不成熟\shirun\02_final_data\季节\日\6-8\','xunlian\Early'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

 

%xunlain
for k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
  
      fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,1096);
      data = data';
      
      data1=zeros(276,1); %%%%%%%%%%%%%%%%%%%%
      
       a=0;
       for i=152:1:243
           for j=1:1:1
               a=a+1;
               data1(a,1)=data(i,j);
           end
       end
        
        for i=518:1:609
           for j=1:1:1
               a=a+1;
               data1(a,1)=data(i,j);
           end
        end
        
         for i=883:1:974
           for j=1:1:1
               a=a+1;
               data1(a,1)=data(i,j);
           end
         end
         
     SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
     %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
    
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');
    
     
     for i=1:1:276
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data1(i,j));
             else
                fprintf(fid1,'%g ',data1(i,j));
             end
         end   
     end
     fclose(fid1); 
     fclose(fid); 
        
         
end

% %ceshi
% for k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%   
%       fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,365);
%       data = data';
%       
%       data1=zeros(92,1); %%%%%%%%%%%%%%%%%%%%
%       
%        a=0;
%        for i=60:1:151
%            for j=1:1:1
%                a=a+1;
%                data1(a,1)=data(i,j);
%            end
%        end
%        
%      SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
%      %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
%     
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%      fid1=fopen(outfile,'w');
%     
%      
%      for i=1:1:92
%          for j=1:1:1
%              if j==1
%                  fprintf(fid1,'%g\r\n',data1(i,j));
%              else
%                 fprintf(fid1,'%g ',data1(i,j));
%              end
%          end   
%      end
%      fclose(fid1); 
%      fclose(fid); 
%         
%          
% end
 disp('处理完成')


 