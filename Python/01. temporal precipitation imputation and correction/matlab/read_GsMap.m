%%This is a program to resampling for GSMaP��from 0.1 degree to 0.5 degree
FolderPath=input('���������ݴ洢�ļ���:','s');
SaveFolder=strcat('H:\GSMAP����\hourly����\�й�����ȡ�������-0.1\','GSMaP_mvk\2017'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
disp('������')

all_file=dir(FolderPath);

% 0.5
% for k=3:length(all_file)
%    FilePath=strcat(FolderPath,'\',all_file(k).name);  %�ļ�·��\�ļ���
%    Name=all_file(k).name;
%    location=strfind(Name,'.'); 
% 
%     fid = fopen(FilePath, 'rb','l');
%     rain = fread(fid,[3600,1200],'float');
%     rain = rain';
%     rain(rain < 0) = 0;
%     
%     %rain��0.1��0.1   0.1 -->0.5
%     [n,m] = size(rain); 
%     newdata=zeros(n/5,m/5); 
%     for i=1:n/5
%         for j=1:m/5
%             newdata(i,j)=(sum(sum(rain((i*5-4:i*5),(j*5-4:j*5)))))./25;
%         end
%     end 
%     
%     % ����Ҫ��һ�µõ���Ӧ��ʱ��
%     newfilename=strcat(Name(13:20),'.txt');
%     FilePath=strcat(SaveFolder,'\',newfilename); %�ļ�·��\�ļ���
%    
%     fid1=fopen(FilePath,'w');
%    
%      
%     fprintf(fid1,'NCOLS        720\r\nNROWS        240\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.500\r\nNODATA_VALUE   -9999.0000\r\n');
%     
%      for i=1:1:240
%         for j=1:1:720
%             if j==720
%                 fprintf(fid1,'%9.4f\n',newdata(i,j));
%             else
%                 fprintf(fid1,'%9.4f\t',newdata(i,j));
%             end
%         end
%     end
%    
%     %fprintf(fid1,strcat(repmat('%9.4f\t',1,720),'\r\n'),newdata'); 
%     
%     fclose(fid);  
%     fclose(fid1);
% end
% 
% disp('�������')
% 
% 

%0.1
parfor k=3:length(all_file)
   FilePath=strcat(FolderPath,'\',all_file(k).name);  %�ļ�·��\�ļ���
   Name=all_file(k).name;
   location=strfind(Name,'.'); 

    fid = fopen(FilePath, 'rb','l');
    rain = fread(fid,[3600,1200],'float32');
    rain = rain';
    rain(rain < 0) = 0;
    fclose(fid); 
    
    % ����Ҫ��һ�µõ���Ӧ��ʱ��   gauge: 13-20  mvk:11-18
    newfilename=strcat(Name(11:18),'.txt');
    FilePath=strcat(SaveFolder,'\',newfilename); %�ļ�·��\�ļ���
   
    fid1=fopen(FilePath,'w');
   
     
    fprintf(fid1,'NCOLS        3600\r\nNROWS        1200\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.100\r\nNODATA_VALUE   -9999.0000\r\n');

    %��
    fprintf(fid1,strcat(repmat('%9.4f\t',1,3600),'\r\n'),rain'); 
     
    fclose(fid1);
end

disp('�������')



