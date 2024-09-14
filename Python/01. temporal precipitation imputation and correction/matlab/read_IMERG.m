%Ԥ���� 0.1deg
FolderPath=input('���������ݴ洢�ļ���:','s');
SaveFolder=strcat('H:\ԭʼ����\','ImergFinal2015-201909-0.1deg'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
disp('������')

all_file=dir(FolderPath);

%k=3:length(all_file)
% ����ֻѡȡ2015���2016������
for k=87651:122834
    %filename = all_file(k).name;
    FilePath=strcat(FolderPath,'\',all_file(k).name);  %�ļ�·��\�ļ���
    Name=all_file(k).name;
    location=strfind(Name,'.'); 
    
    data = h5read(FilePath,'/Grid/precipitationCal');
    newdata=flipud(data);  %���·�ת
    newdata=circshift(newdata,1800,2); %����ƽ��
    newdata=newdata/2;    %��λƥ��
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
    FilePath=strcat(SaveFolder,'\',newfilename); %�ļ�·��\�ļ���
   
    fid=fopen(FilePath,'w');
    newdata=newdata(300:1500,:);  % 1800������1200
     
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
disp('������')
            

