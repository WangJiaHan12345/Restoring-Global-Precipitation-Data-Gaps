
SaveFolder=strcat('G:\毕业论文图\青藏高原\空间\月份性能折线图\','1-12'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end

FolderPath1=input('请输入数据存储文件夹:','s'); %输入3-5数据
index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

FolderPath2=input('请输入数据存储文件夹:','s'); %输入6-8数据
index2=strfind(FolderPath2,'\');  %输出字符'\'在FolderPath的位置
Files2=dir(FolderPath2);
FilesCount2=length(Files2);

FolderPath3=input('请输入数据存储文件夹:','s'); %输入9-11数据
index3=strfind(FolderPath3,'\');  %输出字符'\'在FolderPath的位置
Files3=dir(FolderPath3);
FilesCount3=length(Files3);

FolderPath4=input('请输入数据存储文件夹:','s'); %输入12-2数据
index4=strfind(FolderPath4,'\');  %输出字符'\'在FolderPath的位置
Files4=dir(FolderPath4);
FilesCount4=length(Files4);

disp('处理中...');

 %15-18年
% parfor k=3:FilesCount1
%       FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %文件路径\文件名
%       FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %文件路径\文件名
%       FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %文件路径\文件名
%       FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %文件路径\文件名
%       
%       Name=Files1(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%       
% 
%       fid1=fopen(FilePath1,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
%       data1 = reshape(data1,1,12);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,12);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,12);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,12);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(48,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:2
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=3:1:5
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=6:1:8
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=9:1:11
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=12:1:14
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=15:1:17
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=18:1:20
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=21:1:23
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%         
%        for i=24:1:26
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=27:1:29
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=30:1:32
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=33:1:35
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%        for i=36:1:38
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end       
%         
%         for i=39:1:41
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end   
%        
%         for i=42:1:44
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%         end   
%         
%         for i=45:1:47
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%         for i=48:1:48
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end   
%         
%         SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC输出文件夹路径
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:48
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid,'%g\r\n',data(i,j));
%                  else
%                     fprintf(fid,'%g ',data(i,j));
%                  end
%              end   
%          end
%          fclose(fid); 
%                
% end


%18
% parfor k=3:FilesCount1
%       FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %文件路径\文件名
%       FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %文件路径\文件名
%       FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %文件路径\文件名
%       FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %文件路径\文件名
%       
%       Name=Files1(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%       
% 
%       fid1=fopen(FilePath1,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
%       data1 = reshape(data1,1,3);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,3);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,3);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,3);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(12,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:2
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=3:1:5
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=6:1:8
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=9:1:11
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=12:1:12
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
%         
%         SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC输出文件夹路径
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:12
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid,'%g\r\n',data(i,j));
%                  else
%                     fprintf(fid,'%g ',data(i,j));
%                  end
%              end   
%          end
%          fclose(fid); 
%                
% end
% disp('处理完成')


%15-16
parfor k=3:FilesCount1
      FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %文件路径\文件名
      FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %文件路径\文件名
      FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %文件路径\文件名
      FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %文件路径\文件名
      
      Name=Files1(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
      

      fid1=fopen(FilePath1,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,6);
      data1 = data1'; 
      fclose(fid1);
      
      fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
      data2 = reshape(data2,1,6);
      data2 = data2'; 
      fclose(fid2);
      
      fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
      data3 = reshape(data3,1,6);
      data3 = data3'; 
      fclose(fid3);
      
      fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
      data4 = reshape(data4,1,6);
      data4 = data4'; 
      fclose(fid4);
      
      
      data =zeros(24,1); 

      
       a=0; %3-5
       b=0; %6-8
       c=0; %9-11
       d=0; %12-2
       
       for i=1:1:2
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
       end

       for i=3:1:5
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=6:1:8
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=9:1:11
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
       
        for i=12:1:14
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end
        
         for i=15:1:17
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=18:1:20
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=21:1:23
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
       
        for i=24:1:24
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end
        
        SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC输出文件夹路径
        outfile=strcat(SaveFolder,'\',SaveFiles);      
        fid=fopen(outfile,'w');      

         for i=1:1:24
             for j=1:1:1
                 if j==1
                     fprintf(fid,'%g\r\n',data(i,j));
                 else
                    fprintf(fid,'%g ',data(i,j));
                 end
             end   
         end
         fclose(fid); 
               
end
disp('处理完成')