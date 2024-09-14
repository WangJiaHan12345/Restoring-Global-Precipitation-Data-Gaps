
data1 = csvread('H:\空间预测\ganhan\02_final_data\季节\Early4\rain_sum_3-5.csv',1,1);
data2 = csvread('H:\空间预测\ganhan\02_final_data\季节\Early4\rain_sum_6-8.csv',1,1);
data3 = csvread('H:\空间预测\ganhan\02_final_data\季节\Early4\rain_sum_9-11.csv',1,1);
data4 = csvread('H:\空间预测\ganhan\02_final_data\季节\Early4\rain_sum_12-2.csv',1,1);


dem = csvread('H:\空间预测\ganhan\dem\global\DEM.csv',1,1);
disp('处理中...');

a = [];
b =[];
c =[];
d =[];
e =[];

count = 0;

for i = 1:1:186
    for j =1:1:720
        if dem(i,j)~=-9999 && data1(i,j)~=-9999 && data2(i,j)~=-9999 && data3(i,j)~=-9999 && data4(i,j)~=-9999
            count = count +1;
            a(count,1) = dem(i,j);
            b(count,1) = data1(i,j);
            c(count,1) = data2(i,j);
            d(count,1) = data3(i,j);
            e(count,1) = data4(i,j);         
        end
    end
end

SaveFolder=strcat('G:\毕业论文图\全球\空间\dem与preci关系\','ganhan'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end


outfile1=strcat(SaveFolder,'\','dem.txt');
outfile2=strcat(SaveFolder,'\','rain_sum_3-5.txt');
outfile3=strcat(SaveFolder,'\','rain_sum_6-8.txt');
outfile4=strcat(SaveFolder,'\','rain_sum_9-11.txt');
outfile5=strcat(SaveFolder,'\','rain_sum_12-2.txt');


fid1=fopen(outfile1,'w');
fid2=fopen(outfile2,'w');
fid3=fopen(outfile3,'w');
fid4=fopen(outfile4,'w');
fid5=fopen(outfile5,'w');
        

 for i=1:1:count
     for j=1:1:1
         if j==1
             fprintf(fid1,'%g\r\n',a(i,j));
             fprintf(fid2,'%g\r\n',b(i,j));
             fprintf(fid3,'%g\r\n',c(i,j));
             fprintf(fid4,'%g\r\n',d(i,j));
             fprintf(fid5,'%g\r\n',e(i,j));
         else
             fprintf(fid1,'%g',a(i,j));
             fprintf(fid2,'%g',b(i,j));
             fprintf(fid3,'%g',c(i,j));
             fprintf(fid4,'%g',d(i,j));
             fprintf(fid5,'%g',e(i,j));
         end
     end   
 end
         
fclose(fid1); 
fclose(fid2); 
fclose(fid3); 
fclose(fid4); 
fclose(fid5); 