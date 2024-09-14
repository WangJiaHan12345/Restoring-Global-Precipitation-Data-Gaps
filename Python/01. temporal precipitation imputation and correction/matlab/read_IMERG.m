%预处理 0.1deg
FolderPath=input('请输入数据存储文件夹:','s');
SaveFolder=strcat('H:\原始数据\','ImergFinal2015-201909-0.1deg'); %输出文件夹路径
if exist(SaveFolder,'dir')~=7  %如果路径不存在则新建路径
    mkdir(SaveFolder);
end
disp('处理中')

all_file=dir(FolderPath);

%k=3:length(all_file)
% 这里只选取2015年和2016年两年
for k=87651:122834
    %filename = all_file(k).name;
    FilePath=strcat(FolderPath,'\',all_file(k).name);  %文件路径\文件名
    Name=all_file(k).name;
    location=strfind(Name,'.'); 
    
    data = h5read(FilePath,'/Grid/precipitationCal');
    newdata=flipud(data);  %上下翻转
    newdata=circshift(newdata,1800,2); %左右平移
    newdata=newdata/2;    %单位匹配
    newdata(newdata<0)=-9999;
           
%     [n,m] = size(newdata);  %0.1 -->0.5
%     newdata1=zeros(n/5,m/5);
%     for i=1:n/5
%         for j=1:m/5
%             c=sum(sum(newdata((i*5-4:i*5),(j*5-4:j*5))));
%             if c>=0 || c==(-999*25)
%                 newdata1(i,j)=c/25;       
%             else
%                 count=ceil(c/(-999));
%                 newdata1(i,j)=(c+count*999)/(25-count);
%             end
%         end
%     end 

    newfilename=strcat(Name(24:31),'.txt');
    FilePath=strcat(SaveFolder,'\',newfilename); %文件路径\文件名
   
    fid=fopen(FilePath,'w');
    newdata=newdata(300:1500,:);  % 1800——》1200
     
    fprintf(fid,'NCOLS        3600\r\nNROWS        1200\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.100\r\nNODATA_VALUE   -9999.0000\r\n');
%     for i=1:1:240
%         for j=1:1:720
%             if j==720
%                 fprintf(fid,'%9.4f\n',newdata1(i,j));
%             else
%                 fprintf(fid,'%9.4f\t',newdata1(i,j));
%             end
%         end
%     end
    fprintf(fid,strcat(repmat('%9.4f\t',1,3600),'\r\n'),newdata'); 
    fclose(fid);  
end       
disp('处理完')
            

