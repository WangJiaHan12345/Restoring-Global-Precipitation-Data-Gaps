%从季节性网格中提取固定的月份网格
%输入季节性网格数据
FolderPath1=input('请输入数据存储文件夹:','s'); %输入完整的gsmap_mvk数据
index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
Files=dir(FolderPath1);
FilesCount=length(Files);

FolderPath2=input('请输入数据存储文件夹:','s'); %输入完整的gsmap_gauge数据
index2=strfind(FolderPath2,'\');  %输出字符'\'在FolderPath的位置


FolderPath3=input('请输入数据存储文件夹:','s'); %输入完整的国家气象局数据
index3=strfind(FolderPath3,'\');  %输出字符'\'在FolderPath的位置

FolderPath4=input('请输入数据存储文件夹:','s'); %输入完整的温度数据
index4=strfind(FolderPath4,'\');  %输出字符'\'在FolderPath的位置


FolderPath5=input('请输入数据存储文件夹:','s'); %输入完整的纬度数据
index5=strfind(FolderPath5,'\');  %输出字符'\'在FolderPath的位置


%3-5 6-8 9-11  12-2 都要来一次
SaveFolder1=strcat('H:\青藏高原数据\时间预测\01_clip_data\训练\gsmap_mvk\','12-2'); %输出文件夹路径
if exist(SaveFolder1,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder1);
end

SaveFolder2=strcat('H:\青藏高原数据\时间预测\01_clip_data\训练\gsmap_gauge\','12-2'); %输出文件夹路径
if exist(SaveFolder2,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder2);
end

SaveFolder3=strcat('H:\青藏高原数据\时间预测\01_clip_data\训练\国家气象局\','12-2'); %输出文件夹路径
if exist(SaveFolder3,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder3);
end

SaveFolder4=strcat('H:\青藏高原数据\时间预测\01_clip_data\训练\温度\','12-2'); %输出文件夹路径
if exist(SaveFolder4,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder4);
end

SaveFolder5=strcat('H:\青藏高原数据\时间预测\01_clip_data\训练\纬度\','12-2'); %输出文件夹路径
if exist(SaveFolder5,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder5);
end

SaveFolder6=strcat('H:\青藏高原数据\时间预测\01_clip_data\测试\gsmap_mvk\','12-2'); %输出文件夹路径
if exist(SaveFolder6,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder6);
end

SaveFolder7=strcat('H:\青藏高原数据\时间预测\01_clip_data\测试\gsmap_gauge\','12-2'); %输出文件夹路径
if exist(SaveFolder7,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder7);
end

SaveFolder8=strcat('H:\青藏高原数据\时间预测\01_clip_data\测试\国家气象局\','12-2'); %输出文件夹路径
if exist(SaveFolder8,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder8);
end

SaveFolder9=strcat('H:\青藏高原数据\时间预测\01_clip_data\测试\温度\','12-2'); %输出文件夹路径
if exist(SaveFolder9,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder9);
end

SaveFolder10=strcat('H:\青藏高原数据\时间预测\01_clip_data\测试\纬度\','12-2'); %输出文件夹路径
if exist(SaveFolder10,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder10);
end

disp('处理中...');

% 15-16 
parfor k=3:FilesCount
      FilePath1=strcat(FolderPath1,'\',Files(k).name);  %文件路径\文件名
      FilePath2=strcat(FolderPath2,'\',Files(k).name);  %文件路径\文件名
      FilePath3=strcat(FolderPath3,'\',Files(k).name);  %文件路径\文件名
      FilePath4=strcat(FolderPath4,'\',Files(k).name);  %文件路径\文件名
      FilePath5=strcat(FolderPath5,'\',Files(k).name);  %文件路径\文件名
      
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
      
      day = 181; %15-16中每个季节的天数
      
      fid1=fopen(FilePath1,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,day);
      data1 = data1'; 
      fclose(fid1);
      
      fid2=fopen(FilePath2,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
      data2 = reshape(data2,1,day);
      data2 = data2'; 
      fclose(fid2);
      
      fid3=fopen(FilePath3,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
      data3 = reshape(data3,1,day);
      data3 = data3'; 
      fclose(fid3);
      
      fid4=fopen(FilePath4,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
      data4 = reshape(data4,1,day);
      data4 = data4'; 
      fclose(fid4);
      
      fid5=fopen(FilePath5,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data5 = cell2mat(textscan(fid5,'%f','headerlines',0));
      data5 = reshape(data5,1,day);
      data5 = data5'; 
      fclose(fid5);
%       
      
       count1=150;   %训练的天数
       count2=31;    %测试的天数（最后的一个月）
       
      data_1 =data1(1:count1,:); % 3-5 276 夏季：6-8 276 秋季：9-11  273  冬季：12-2 271
      data_2 =data2(1:count1,:); 
      data_3 =data3(1:count1,:); 
      data_4 =data4(1:count1,:); 
      data_5 =data5(1:count1,:); 
      
      data_6 =data1(count1+1:day,:);
      data_7 =data2(count1+1:day,:);
      data_8 =data3(count1+1:day,:);
      data_9 =data4(count1+1:day,:);
      data_10 =data5(count1+1:day,:);
        
     SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC输出文件夹路径
     
     outfile1=strcat(SaveFolder1,'\',SaveFiles);      
     fid1=fopen(outfile1,'w');      
     
     outfile2=strcat(SaveFolder2,'\',SaveFiles);      
     fid2=fopen(outfile2,'w');   
     
     outfile3=strcat(SaveFolder3,'\',SaveFiles);      
     fid3=fopen(outfile3,'w');   
     
     outfile4=strcat(SaveFolder4,'\',SaveFiles);      
     fid4=fopen(outfile4,'w');   
     
     outfile5=strcat(SaveFolder5,'\',SaveFiles);      
     fid5=fopen(outfile5,'w');   
     
     outfile6=strcat(SaveFolder6,'\',SaveFiles);      
     fid6=fopen(outfile6,'w');   
     
     outfile7=strcat(SaveFolder7,'\',SaveFiles);      
     fid7=fopen(outfile7,'w');   
     
     outfile8=strcat(SaveFolder8,'\',SaveFiles);      
     fid8=fopen(outfile8,'w');   
     
     outfile9=strcat(SaveFolder9,'\',SaveFiles);      
     fid9=fopen(outfile9,'w');   
     
     outfile10=strcat(SaveFolder10,'\',SaveFiles);      
     fid10=fopen(outfile10,'w');   

     for i=1:1:count1
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data_1(i,j));
                 fprintf(fid2,'%g\r\n',data_2(i,j));
                 fprintf(fid3,'%g\r\n',data_3(i,j));
                 fprintf(fid4,'%g\r\n',data_4(i,j));
                 fprintf(fid5,'%g\r\n',data_5(i,j));
             else
                fprintf(fid1,'%g ',data_1(i,j));
                fprintf(fid2,'%g ',data_2(i,j));
                fprintf(fid3,'%g ',data_3(i,j));
                fprintf(fid4,'%g ',data_4(i,j));
                fprintf(fid5,'%g ',data_5(i,j));
             end
         end   
     end
     fclose(fid1); 
     fclose(fid2); 
     fclose(fid3); 
     fclose(fid4); 
     fclose(fid5); 
     
      for i=1:1:count2
         for j=1:1:1
             if j==1
                 fprintf(fid6,'%g\r\n',data_6(i,j));
                 fprintf(fid7,'%g\r\n',data_7(i,j));
                 fprintf(fid8,'%g\r\n',data_8(i,j));
                 fprintf(fid9,'%g\r\n',data_9(i,j));
                 fprintf(fid10,'%g\r\n',data_10(i,j));
             else
                 fprintf(fid6,'%g',data_6(i,j));
                 fprintf(fid7,'%g',data_7(i,j));
                 fprintf(fid8,'%g',data_8(i,j));
                 fprintf(fid9,'%g',data_9(i,j));
                 fprintf(fid10,'%g',data_10(i,j));
             end
         end   
      end
     fclose(fid6);  
     fclose(fid7); 
     fclose(fid8); 
     fclose(fid9); 
     fclose(fid10); 
end
disp('处理完成')
