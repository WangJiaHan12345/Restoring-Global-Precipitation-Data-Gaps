SaveFolder=strcat('H:\时间预测\最初数据\温度\','2018_1'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
disp('处理中...');

ncFilePath='H:\时间预测\最初数据\温度\tmin.2018.nc';

lon=ncread(ncFilePath,'lon');%读取经度变量
lat=ncread(ncFilePath,'lat');%读取纬度变量
time=ncread(ncFilePath,'time');%读取时间变量
tmin=ncread(ncFilePath,'tmin');%获取蒸散发变量数据
for a=1:1:365
    tmin1=tmin(:,:,a);
    tmin1=rot90(tmin1,3);
    tmin1=fliplr(tmin1);

    for i=1:1:360
        for j=1:1:720
           if tmin1(i,j)== -9.969209968386869e+36
               tmin1(i,j)= -9999;
           end
        end
    end
      
      Date=datetime(2018,1,a);
      DateString = datestr(Date);

      SaveFiles=strcat(DateString,'.txt'); %CPC输出文件夹路径
     
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');

     fprintf(fid1,'NCOLS        720\r\nNROWS        240\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.5\r\nNODATA_VALUE    -9999\r\n');


     for i=61:1:300
         for j=1:1:720
             if j==720
                 fprintf(fid1,'%g\r\n',tmin1(i,j));
             else
                fprintf(fid1,'%g ',tmin1(i,j));
             end
         end   
     end
         fclose(fid1); 
end
disp('处理完');
