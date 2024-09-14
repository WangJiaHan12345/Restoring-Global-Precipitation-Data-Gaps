data_rain=csvread('G:\全球\时间预测结果\画相对图所用数据\季度\原始数据_1\分气候区\干旱区\经纬度相关性\rain_sum_12-2.csv',1,1);%csvread只能读取纯数据
rain = [];
lon = [];
count = 0;

for j =1:1:720
    day = 0;
    sum = 0;
    for i =1:1:240
        if data_rain(i,j)>0
            day = day +1;
            sum = sum +data_rain(i,j);
        end
    end
    
    if day >0
        count =count +1;
        rain(count,1) = sum/day;
        lon(count,1) = j*0.5;
    end
end


SaveFolder=strcat('G:\全球\时间预测结果\画相对图所用数据\季度\原始数据_1\分气候区\干旱区\','经纬度相关性'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end

outfile1=strcat(SaveFolder,'\','12-2.txt');
outfile2=strcat(SaveFolder,'\','lon_12-2.txt');

 if exist(outfile1,'file')~=0 
    delete(outfile1);     
 end
 
  if exist(outfile2,'file')~=0 
    delete(outfile2);     
 end


 fid1=fopen(outfile1,'w');
 fid2=fopen(outfile2,'w');

 for i=1:1:count
     for j=1:1:1
         if j==1
             fprintf(fid1,'%g\r\n',rain(i,j));
         else
             fprintf(fid1,'%g ',rain(i,j));
         end
     end   
 end
 
 
  for i=1:1:count
     for j=1:1:1
         if j==1
             fprintf(fid2,'%g\r\n',lon(i,j));
         else
             fprintf(fid2,'%g ',lon(i,j));
         end
     end   
  end
 
 fclose(fid1);    
 fclose(fid2);  
          

disp('处理完成');


    