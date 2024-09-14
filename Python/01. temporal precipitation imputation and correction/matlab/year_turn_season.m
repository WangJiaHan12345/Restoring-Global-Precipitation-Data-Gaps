%将全年数据转为季节数据
FolderPath=input('请输入数据存储文件夹:','s'); %输入全年数据
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
SaveFolder1=strcat('H:\青藏高原数据\时间预测\2015-2016\01_clip_data\测试\','gsmap_rnt\12-2'); %输出文件夹路径
if exist(SaveFolder1,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder1);
end

SaveFolder2=strcat('H:\青藏高原数据\时间预测\2015-2016\01_clip_data\测试\','gsmap_rnt\3-5'); %输出文件夹路径
if exist(SaveFolder2,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder2);
end

SaveFolder3=strcat('H:\青藏高原数据\时间预测\2015-2016\01_clip_data\测试\','gsmap_rnt\6-8'); %输出文件夹路径
if exist(SaveFolder3,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder3);
end

SaveFolder4=strcat('H:\青藏高原数据\时间预测\2015-2016\01_clip_data\测试\','gsmap_rnt\9-11'); %输出文件夹路径
if exist(SaveFolder4,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder4);
end

Files=dir(FolderPath);
FilesCount=length(Files);
disp('处理中...');

% xunlain  15-17年
% parfor k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%   
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,1096);
%       data = data'; 
%       
%       
%       data_1 =zeros(271,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
%       data_2 =zeros(276,1);
%       data_3 =zeros(276,1);
%       data_4 =zeros(273,1);
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=426:1:517
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=518:1:609
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=610:1:700
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%         
%        for i=701:1:790
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=791:1:882
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=883:1:974
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=975:1:1065
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end   
%         
%        for i=1066:1:1096
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end        
%         
% 
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:271
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          fclose(fid1); 
%          
%          for i=1:1:276
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end   
%          fclose(fid2);
%          
%          for i=1:1:276
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end      
%          fclose(fid3); 
%          
%          for i=1:1:273
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end         
%          
%          fclose(fid4); 
%          fclose(fid); 
%                
% end
% disp('处理完成')


%ceshi  18年
% parfor k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%   
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,365);
%       data = data'; 
%       
%       
%       data_1 =zeros(90,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
%       data_2 =zeros(92,1);
%       data_3 =zeros(92,1);
%       data_4 =zeros(91,1);
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:365
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
% 
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:90
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end         
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end         
%          
%          for i=1:1:91
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end            
%          
%          fclose(fid1); 
%          fclose(fid2); 
%          fclose(fid3); 
%          fclose(fid4); 
%          fclose(fid); 
%                
% end
% disp('处理完成')


% 15-18年
% for k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%   
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,1461);
%       data = data'; 
%       
%       
%       data_1 =zeros(361,1); % 每个季节有多少天  春季：3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
%       data_2 =zeros(368,1);
%       data_3 =zeros(368,1);
%       data_4 =zeros(364,1);
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=426:1:517
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=518:1:609
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=610:1:700
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%         
%        for i=701:1:790
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=791:1:882
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=883:1:974
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=975:1:1065
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end   
%         
%        for i=1066:1:1155
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end       
%         
%        for i=1156:1:1247
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%        end  
%         
%        for i=1248:1:1339
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%        end  
%         
%                 
%        for i=1340:1:1430
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%        end  
%         
%        for i=1431:1:1461
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end  
%         
% 
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:361
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          fclose(fid1); 
%          
%          for i=1:1:368
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end   
%          fclose(fid2);
%          
%          for i=1:1:368
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end      
%          fclose(fid3); 
%          
%          for i=1:1:364
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end         
%          
%          fclose(fid4); 
%          fclose(fid); 
%                
% end
% disp('处理完成')

%15-16年
parfor k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
  
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));

      fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,731);
      data = data'; 
      fclose(fid);
      
      %训练集    
%       data_1 =zeros(150,1); %12-2
%       data_2 =zeros(153,1); %3-5
%       data_3 =zeros(153,1); %6-8
%       data_4 =zeros(152,1); %9-11
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
%         for i=426:1:486
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
%         
%         
%         for i=518:1:578
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%         end
%         
%         
%          for i=610:1:670
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%          end
        
         
%          %测试集
      data_1 =zeros(31,1); %12-2
      data_2 =zeros(31,1); %3-5
      data_3 =zeros(31,1); %6-8
      data_4 =zeros(30,1); %9-11
      
       a=0;
       b=0;
       c=0;
       d=0;
       
       for i=701:1:731
           for j=1:1:1
               a=a+1;
               data_1(a,1)=data(i,j);
           end
       end

       for i=487:1:517
           for j=1:1:1
               b=b+1;
               data_2(b,1)=data(i,j);
           end
        end

        for i=579:1:609
           for j=1:1:1
               c=c+1;
               data_3(c,1)=data(i,j);
           end
         end

        for i=671:1:700
           for j=1:1:1
               d=d+1;
               data_4(d,1)=data(i,j);
           end
        end
%        


% 完整数据
%       data_1 =zeros(181,1); %12-2
%       data_2 =zeros(184,1); %3-5
%       data_3 =zeros(184,1); %6-8
%       data_4 =zeros(182,1); %9-11
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
%         for i=426:1:517
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
%         
%         
%         for i=518:1:609
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%         end
%         
%         
%          for i=610:1:700
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%          end
%          
%          for i=701:1:731
%            for j=1:1:1
%                a=a+1;
%                data_1(d,1)=data(i,j);
%            end
%          end
%          
        SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径

        outfile1=strcat(SaveFolder1,'\',SaveFiles);
        outfile2=strcat(SaveFolder2,'\',SaveFiles);
        outfile3=strcat(SaveFolder3,'\',SaveFiles);
        outfile4=strcat(SaveFolder4,'\',SaveFiles);

        
        fid1=fopen(outfile1,'w');
        fid2=fopen(outfile2,'w');
        fid3=fopen(outfile3,'w');
        fid4=fopen(outfile4,'w');
        

         for i=1:1:31
             for j=1:1:1
                 if j==1
                     fprintf(fid1,'%g\r\n',data_1(i,j));
                 else
                    fprintf(fid1,'%g ',data_1(i,j));
                 end
             end   
         end
         fclose(fid1); 
         
         for i=1:1:31
             for j=1:1:1
                 if j==1
                     fprintf(fid2,'%g\r\n',data_2(i,j));
                 else
                    fprintf(fid2,'%g ',data_2(i,j));
                 end
             end   
         end   
         fclose(fid2);
         
         for i=1:1:31
             for j=1:1:1
                 if j==1
                     fprintf(fid3,'%g\r\n',data_3(i,j));
                 else
                    fprintf(fid3,'%g ',data_3(i,j));
                 end
             end   
         end      
         fclose(fid3); 
         
         for i=1:1:30
             for j=1:1:1
                 if j==1
                     fprintf(fid4,'%g\r\n',data_4(i,j));
                 else
                    fprintf(fid4,'%g ',data_4(i,j));
                 end
             end   
         end         
         
         fclose(fid4); 
               
end
disp('处理完成')



%17年
% parfor k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %文件路径\文件名
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
%   
% %       i= str2num(Name(location(end)-6:location(end)-4));
% %       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,365);
%       data = data'; 
%       fclose(fid);
%       
%       data_1 =zeros(90,1); %12-2
%       data_2 =zeros(92,1); %3-5
%       data_3 =zeros(92,1); %6-8
%       data_4 =zeros(91,1); %9-11
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:365
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
%          
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:90
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          fclose(fid1); 
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end   
%          fclose(fid2);
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end      
%          fclose(fid3); 
%          
%          for i=1:1:91
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end         
%          
%          fclose(fid4); 
%                
% end
% disp('处理完成')