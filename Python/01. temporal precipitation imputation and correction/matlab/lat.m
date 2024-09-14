SaveFolder=strcat('G:\青藏高原\纬度\','2015-2016'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end

fid = fopen('G:\青藏高原\中国-青藏高原-440.txt','rb','l');
data = cell2mat(textscan(fid,'%f','headerlines',6));
data = reshape(data,700,440);
data = data'; 
fclose(fid); 
disp('处理中...');

% 可用网格是txt文件
for i = 1:1:440
    for j =1:1:700
        if data(i,j) >= 0
            
            a = 59 - (i-1)*0.1;
            rain = ones(731,1)*a;

    
            SaveFiles= [num2str(i,'%03d'),num2str(j,'%03d'),'.txt']; %CPC输出文件夹路径

             outfile=strcat(SaveFolder,'\',SaveFiles);

             if exist(outfile,'file')~=0 
                delete(outfile);     
             end
     
     
             fid1=fopen(outfile,'w');
             
             for m=1:1:731  
                 for n=1:1:1
                     if n==1
                         fprintf(fid1,'%g\r\n',rain(m,n));
                     else
                         fprintf(fid1,'%g ',rain(m,n));
                     end
                 end   
             end
             
             fclose(fid1);             
        end  
    end
end
disp('处理完成');
