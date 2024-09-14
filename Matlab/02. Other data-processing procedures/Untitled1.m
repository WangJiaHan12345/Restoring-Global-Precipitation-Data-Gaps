% 将网格形式转为天数的形式
FolderPath1=input('请输入数据存储文件夹:','s');   %输入GPCC文件夹
index1=strfind(FolderPath1,'\');  %输出字符'\'在FolderPath的位置
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

SaveFolder=strcat('G:\全球\空间校正结果\用于和GPCP比较\','月尺度\DNN'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end

disp('处理中...');


parfor k=3:1:FilesCount1
     
      FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %文件路径\文件名
      Name=Files1(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置


      fid1=fopen(FilePath1,'rb','l');  % 'rb'以二进制方式只读类型打开文件，也可以直接'r';'l':little endian小端序打开
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,1461);
      data1 = data1'; 
      fclose(fid1);
      
     result = zeros(48,1);
    
     day = [0,31,59,90,120,151,181,212,243,273,304,334,365,396,425,456,486,517,547,578,609,639,670,700,731,762,790,821,851,882,912,943,974,1004,1035,1065,1096,1127,1155,1186,1216,1247,1277,1308,1339,1369,1400,1430,1461];
     
     for m = 1:1:48
         sum = 0;
         for n = day(m)+1:1:day(m+1)
             sum = sum + data1(n,1);
         end
         result(m,1)=sum;
     end
     
     SaveFiles=strcat(Name(1:6),'.txt'); %CPC输出文件夹路径

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end    

    fid1=fopen(outfile,'w');

     for a=1:1:48
         for b=1:1:1
             if b==1
                 fprintf(fid1,'%g\r\n',result(a,b));
             else
                 fprintf(fid1,'%g ',result(a,b));
             end
         end   
     end
    fclose(fid1); 
end
