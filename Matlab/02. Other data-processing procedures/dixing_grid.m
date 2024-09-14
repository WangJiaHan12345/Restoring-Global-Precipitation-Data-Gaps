% 将网格的形式改为日期的形式
FolderPath=input('请输入数据存储文件夹:','s');  %输入可用网格
index=strfind(FolderPath,'\');  %输出字符'\'在FolderPath的位置
Files=dir(FolderPath);
FilesCount=length(Files);

SaveFolder=strcat('G:\全球\时间预测结果\画相对图所用数据\季度\只含地面站点\','降水'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end

fid = fopen('G:\全球\时间预测结果\画相对图所用数据\季度\原始数据\rain_sum_12-2.txt','rb','l');
data = cell2mat(textscan(fid,'%f','headerlines',0));
data = reshape(data,720,240);
data = data'; 
fclose(fid); 

result = [];
a = 0;
% 将网格数据转为天数据
for k=3:FilesCount
       
      Name=Files(k).name;
      location=strfind(Name,'.');  %输出字符'.'在FilePath的位置
      
      
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      a = a + 1;
      result(a,1) = data(i,j);
       
end


 outfile=strcat(SaveFolder,'\','san_sum_12-2.txt');

 if exist(outfile,'file')~=0 
    delete(outfile);     
 end


 fid1=fopen(outfile,'w');
     
 for i=1:1:a
     for j=1:1:1
         if j==1
             fprintf(fid1,'%g\r\n',result(i,j));
         else
             fprintf(fid1,'%g ',result(i,j));
         end
     end   
 end
 fclose(fid1);         