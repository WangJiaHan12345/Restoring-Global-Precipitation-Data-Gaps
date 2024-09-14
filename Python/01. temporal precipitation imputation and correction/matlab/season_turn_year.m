%将季节数据转为全年数据
SaveFolder=strcat('H:\青藏高原数据\空间+时间\01_data\2015-2016\gsmap_gauge\','1-12'); %输出文件夹路径
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

%  15-18年
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
%       data1 = reshape(data1,1,368);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,368);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,364);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,361);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(1461,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:59
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=426:1:517
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=518:1:609
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=610:1:700
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%         
%        for i=701:1:790
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=791:1:882
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=883:1:974
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=975:1:1065
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%        for i=1066:1:1155
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end       
%         
%         for i=1156:1:1247
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end   
%        
%         for i=1248:1:1339
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%         end   
%         
%         for i=1340:1:1430
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%         for i=1431:1:1461
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end   
%         
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:1461
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

% 15-16
parfor k=3:FilesCount1
      FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %文件路径\文件名
      FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %文件路径\文件名
      FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %文件路径\文件名
      FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %文件路径\文件名
      
      Name=Files1(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
      

      fid1=fopen(FilePath1,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,184);
      data1 = data1'; 
      fclose(fid1);
      
      fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
      data2 = reshape(data2,1,184);
      data2 = data2'; 
      fclose(fid2);
      
      fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
      data3 = reshape(data3,1,182);
      data3 = data3'; 
      fclose(fid3);
      
      fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
      data4 = reshape(data4,1,181);
      data4 = data4'; 
      fclose(fid4);
      
      
      data =zeros(731,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271

      
       a=0; %3-5
       b=0; %6-8
       c=0; %9-11
       d=0; %12-2
       
       for i=1:1:59
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
       end

       for i=60:1:151
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=152:1:243
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=244:1:334
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
       
        for i=335:1:425
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end

       for i=426:1:517
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=518:1:609
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=610:1:700
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
        
       for i=701:1:731
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end

        
        SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
        outfile=strcat(SaveFolder,'\',SaveFiles);      
        fid=fopen(outfile,'w');      

         for i=1:1:731
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
%       data1 = reshape(data1,1,90);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,90);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,90);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,85);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(355,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:55
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=56:1:145
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=146:1:235
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=236:1:325
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=326:1:355
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%         
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:355
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

% 16年
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
%       data1 = reshape(data1,1,30);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,30);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,30);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,30);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(120,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:30
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%        end
% 
%        for i=31:1:60
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%         end
% 
%         for i=61:1:90
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%          end
% 
%         for i=91:1:120
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
%        
%         
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:120
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
